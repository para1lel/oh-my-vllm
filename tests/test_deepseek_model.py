import gc
import hashlib
import json
import math
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model
from oh_my_vllm.models.deepseek_v41.decode import (
    DecodeBindings,
    EngramStage,
    pause_state_export,
    sampled_token_ids,
)
from oh_my_vllm.models.deepseek_v41.engram import NgramHashState
from oh_my_vllm.models.deepseek_v41.kernel import act_quant, fp8_gemm
from oh_my_vllm.models.deepseek_v41.loader import RankLoader
from oh_my_vllm.models.deepseek_v41.model import (
    Attention,
    Compressor,
    Engram,
    Gate,
    ModelArgs,
    MoE,
    ParallelEngramEmbedding,
    RowParallelLinear,
    Transformer,
    _route_counts,
    compressed_context_width,
    context_token_bucket,
    get_dspark_topk_idxs,
    get_prefill_window_topk_idxs,
    get_window_topk_idxs,
    position_window_topk_idxs,
    select_candidate_blocks,
    shared_attn,
)
from safetensors.torch import save_file


class DeepSeekModel(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_decode_graph_owns_cached_inputs_after_lru_eviction(self):
        from oh_my_vllm.models.deepseek_v41.decode import DecodeGraph

        cached = (
            model.precompute_freqs_cis,
            get_window_topk_idxs,
            get_prefill_window_topk_idxs,
            get_dspark_topk_idxs,
        )

        class CachedModel:
            def __init__(self):
                attention = SimpleNamespace(
                    compress_ratio=0,
                    window_size=128,
                    window_kv_cache=torch.zeros((4, 128, 1)),
                )
                self.layers = [
                    SimpleNamespace(attn=attention, ffn=SimpleNamespace(), engram=None)
                ]
                self.references = ()

            def __call__(self, tokens, start_pos, **kwargs):
                values = (
                    model.precompute_freqs_cis(32, 32, 0, 10000, 1, 32, 1),
                    get_window_topk_idxs(128, 4, 6, 0),
                    get_prefill_window_topk_idxs(128, 4, 6, start_pos),
                    get_dspark_topk_idxs(128, 4, 6, start_pos),
                )
                self.references = tuple(weakref.ref(value) for value in values)
                output = torch.cat(
                    [value.contiguous().view(torch.uint8).flatten() for value in values]
                )
                return tokens[:, 0], output, None

        for operation in cached:
            operation.cache_clear()
        try:
            with torch.inference_mode(), torch.device("cuda"):
                bindings = DecodeBindings(4, (0, 1, 2, 3), "cuda", length=6)
                bindings.bind([[1] * 6] * 4, 2050)
                target = CachedModel()
                expected = target(bindings.token_ids, 2050)[1].clone()
                graph = DecodeGraph(target, bindings, 2050)
                graph.capture()
                references = target.references
                for length in (7, 8, 9):
                    model.precompute_freqs_cis(32, 32 + length, 0, 10000, 1, 32, 1)
                    get_window_topk_idxs(128, 1, length, 0)
                    get_prefill_window_topk_idxs(128, 1, length, 20)
                    get_dspark_topk_idxs(128, 1, length, 20)
                for operation in cached:
                    operation.cache_clear()
                gc.collect()
                self.assertTrue(
                    all(reference() is not None for reference in references)
                )
                for _ in range(2):
                    graph.replay()
                    self.assertTrue(torch.equal(graph.logits, expected))
                del graph
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
                bindings.close()
        finally:
            for operation in cached:
                operation.cache_clear()

    def test_cached_tensor_capture_restores_owners_after_nested_failure(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with kernel.capture_cached_tensors() as outer:
            first = get_prefill_window_topk_idxs(4, 1, 2, 2)
            with (
                self.assertRaisesRegex(RuntimeError, "capture failed"),
                kernel.capture_cached_tensors() as failed,
            ):
                second = get_window_topk_idxs(4, 1, 3, 0)
                raise RuntimeError("capture failed")
            get_prefill_window_topk_idxs(4, 1, 2, 2)
        third = get_dspark_topk_idxs(4, 1, 2, 2)
        self.assertEqual(set(outer), {id(first)})
        self.assertEqual(set(failed), {id(second)})
        self.assertNotIn(id(third), outer)
        self.assertEqual(get_prefill_window_topk_idxs.cache_parameters()["maxsize"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cached_projection_and_status_inputs_keep_tensor_owners(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with (
            torch.inference_mode(),
            patch.dict(kernel._PACKED_WEIGHTS, {}, clear=True),
            patch.dict(kernel._SPARSE_STATUS, {}, clear=True),
        ):
            weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
            scales = torch.full((4, 4), 127, dtype=torch.uint8, device="cuda").view(
                torch.float8_e8m0fnu
            )
            packed = kernel._packed_weight(weight, scales, 8)
            with kernel.capture_cached_tensors() as owners:
                self.assertIs(kernel._packed_weight(weight, scales, 8), packed)
                status = kernel._sparse_status_accumulator(weight.device)
            references = tuple(weakref.ref(value) for value in (*packed[:2], status))
            del packed, status
            kernel.invalidate_packed_weights(weight, scales)
            kernel._SPARSE_STATUS.clear()
            gc.collect()
            self.assertTrue(all(reference() is not None for reference in references))
            owners.clear()
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_prepared_expert_pointer_arrays_own_the_packed_targets(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with torch.inference_mode(), patch.dict(kernel._PACKED_WEIGHTS, {}, clear=True):
            weight = torch.zeros((128, 64), dtype=torch.uint8, device="cuda").view(
                torch.float4_e2m1fn_x2
            )
            scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda").view(
                torch.float8_e8m0fnu
            )
            pointers = kernel._static_pointer_arrays([(weight, scales)], weight.device)
            reference = weakref.ref(
                next(iter(kernel._PACKED_WEIGHTS.values())).packed[1]
            )
            kernel.invalidate_packed_weights(weight, scales)
            gc.collect()
            self.assertIsNotNone(reference())
            del pointers
            gc.collect()
            self.assertIsNone(reference())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_capture_pointer_upload_retains_its_cached_host_snapshot(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        values = [16, 32]
        with (
            torch.inference_mode(),
            patch.dict(kernel._CAPTURE_POINTER_HOSTS, {}, clear=True),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with (
                kernel.capture_cached_tensors() as owners,
                torch.cuda.graph(graph, stream=stream),
            ):
                output = kernel._upload_pointer_array(
                    values, "ownership-test", torch.device("cuda")
                ).clone()
            host = next(iter(kernel._CAPTURE_POINTER_HOSTS.values()))
            reference = weakref.ref(host)
            del host
            kernel._CAPTURE_POINTER_HOSTS.clear()
            gc.collect()
            self.assertIsNotNone(reference())
            graph.replay()
            self.assertEqual(output.tolist(), values)
            del graph
            owners.clear()
            gc.collect()
            self.assertIsNone(reference())

    def test_cuda_batch_rejects_unverified_configuration_before_binding(self):
        from oh_my_vllm.models.deepseek_v41 import batch, kernel

        released = ModelArgs(
            dim=5120,
            hc_mult=4,
            dtype="fp8",
            expert_dtype="fp4",
            n_heads=64,
            head_dim=512,
            n_routed_experts=384,
            n_activated_experts=6,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
            window_size=128,
        )
        invalid = dict(
            dim=1024,
            hc_mult=2,
            dtype="bf16",
            expert_dtype=None,
            n_heads=32,
            head_dim=128,
            n_routed_experts=128,
            n_activated_experts=3,
            index_n_heads=16,
            index_head_dim=64,
            index_topk=1024,
            window_size=1024,
        )
        for field, value in invalid.items():
            worker = SimpleNamespace(
                device="cuda", args=replace(released, **{field: value})
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                batch.target_decode_batch(worker, [[1], [2]], (2, 2), (0, 1))
            self.assertFalse(hasattr(worker, "decode_bindings"))
        for name, value in (
            ("fp8_block_size", 128),
            ("fp4_block_size", 128),
            ("scale_fmt", "float32"),
            ("scale_dtype", torch.float32),
        ):
            with (
                patch.object(model, name, value),
                self.assertRaisesRegex(ValueError, "UE8M0 K32"),
            ):
                batch.require_batch_configuration(released)
        with (
            patch.object(kernel, "_SPARSE_NATIVE_FAILED", True),
            self.assertRaisesRegex(ValueError, "native sparse"),
        ):
            batch.require_batch_configuration(released)
        with (
            patch.object(kernel, "_SPARSE_NATIVE_DISABLED", True),
            patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
        ):
            batch.require_batch_configuration(released)
        with (
            patch.object(kernel, "_SPARSE_NATIVE_DISABLED", False),
            patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
        ):
            batch.require_batch_configuration(released)

    def test_grouped_local_sum_maps_route_slots_for_each_strategy(self):
        from oh_my_vllm.models.deepseek_v41 import kernel
        from oh_my_vllm.models.deepseek_v41.dispatch import _local_expert_sum

        rows = torch.zeros((2, 32), dtype=torch.bfloat16)
        indices = torch.tensor([[5, 7, 3, 6, 4, 0], [0, 1, 2, 6, 7, 5]])
        weights = torch.ones((2, 6), dtype=torch.float32)
        contributions = torch.tensor(
            (0, 0, 0, -0.0244140625, 0.0145263671875, 2.514570951461792e-8, 0, 0),
            dtype=torch.bfloat16,
        )
        expected = torch.zeros_like(rows, dtype=torch.float32)
        expected[0] = -0.009887670166790485
        expected[1] = 2.514570951461792e-8
        original_sum = kernel.ordered_expert_sum
        for strategy in ("TP4", "TEP4", "DEP4"):
            with self.subTest(strategy=strategy), patch.object(model, "world_size", 1):
                moe = MoE(
                    0,
                    ModelArgs(
                        dim=32,
                        moe_inter_dim=32,
                        n_routed_experts=8,
                        n_activated_experts=6,
                        n_shared_experts=1,
                        dtype="bf16",
                        parallel_strategy=strategy,
                    ),
                )
                if strategy != "TP4":
                    moe.experts_start_idx, moe.experts_end_idx = 3, 6
                seen = []

                def grouped(
                    values, tokens, route_weights, counts, count_tensor, moe=moe
                ):
                    first = sum(counts[: moe.experts_start_idx])
                    last = sum(counts[: moe.experts_end_idx])
                    ids = torch.repeat_interleave(torch.arange(8), count_tensor.long())
                    result = (
                        contributions[ids[first:last], None].expand(-1, 32).contiguous()
                    )
                    return tokens[first:last], result

                def checked_sum(values, routes, output, topk, seen=seen):
                    original_sum(values, routes, output, topk)
                    seen.append(output.clone())

                with (
                    patch.object(moe, "_grouped_routed", side_effect=grouped),
                    patch.object(kernel, "ordered_expert_sum", side_effect=checked_sum),
                    patch.object(
                        moe.shared_experts,
                        "forward",
                        return_value=torch.zeros_like(rows),
                    ),
                ):
                    if strategy == "DEP4":
                        _local_expert_sum(moe, rows, indices, weights)
                    else:
                        moe._forward_rows(rows, routing=(weights, indices))
                self.assertEqual(len(seen), 1)
                self.assertTrue(
                    torch.equal(seen[0].view(torch.int32), expected.view(torch.int32))
                )

    def test_moe_prefill_configuration_and_workspace_lifetime(self):
        from oh_my_vllm.models.deepseek_v41.moe_prefill import (
            PrefillWorkspace,
            projection_rows,
        )

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(projection_rows(), 1024)
            for strategy, partitions, expected in (
                ("TEP4", 4, 8192),
                ("TP4", 4, 8192),
                ("DEP4", 4, 1024),
                ("TEP4", 1, 1024),
            ):
                self.assertEqual(
                    projection_rows(strategy=strategy, partitions=partitions), expected
                )
        with patch.dict("os.environ", {"OH_MY_VLLM_DEEPSEEK_MOE_PREFILL_ROWS": "1024"}):
            self.assertEqual(projection_rows(strategy="TEP4", partitions=4), 1024)
        for value in (1024, "4096", "8192"):
            self.assertEqual(projection_rows(value), int(value))
        for value in (0, 2048, 16384, "4096.0", 4096.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                projection_rows(value)
        with patch.dict("os.environ", {"OH_MY_VLLM_DEEPSEEK_MOE_PREFILL_ROWS": "4096"}):
            transformer = Transformer(
                ModelArgs(
                    n_layers=2,
                    dim=32,
                    moe_inter_dim=32,
                    n_heads=1,
                    q_lora_rank=32,
                    o_lora_rank=32,
                    head_dim=32,
                    rope_head_dim=16,
                    vocab_size=32,
                    max_seq_len=8,
                    max_batch_size=1,
                    compress_ratios=[1, 1],
                    kv_source_layers=[],
                    index_source_layers=[],
                    o_groups=1,
                    dtype="bf16",
                    expert_dtype=None,
                )
            )
        workspace = transformer._moe_prefill_workspace
        self.assertIsInstance(workspace, PrefillWorkspace)
        self.assertTrue(
            all(
                layer.ffn._prefill_workspace is workspace
                for layer in transformer.layers
            )
        )
        self.assertEqual(workspace.buffers, {})
        for options, length, enabled in (
            ({}, 2, True),
            ({"start_pos": 1}, 2, True),
            ({}, 1, False),
            ({"logical_rows": 1}, 2, False),
            ({"position": torch.tensor(0)}, 2, False),
            ({"execution_plan": object()}, 2, False),
            ({"valid_rows": torch.ones(2, dtype=torch.bool)}, 2, False),
            ({"dispatch_combine": object()}, 2, False),
        ):
            with self.subTest(options=options, length=length), ExitStack() as stack:
                calls = []

                def block(value, start, pre_mix, *_args, calls=calls, **kwargs):
                    calls.append(kwargs.get("prefill_eager", False))
                    return value, pre_mix

                for layer in transformer.layers:
                    stack.enter_context(
                        patch.object(layer, "forward", side_effect=block)
                    )
                transformer(
                    torch.zeros((1, length), dtype=torch.int64),
                    compute_logits=False,
                    collect_main_hidden=False,
                    reserve_cache=False,
                    check_status=False,
                    **options,
                )
                self.assertEqual(calls, [enabled, enabled])
        workspace.buffers["sentinel"] = torch.empty(1)
        transformer.layers[0].ffn._apply(lambda value: value)
        self.assertEqual(workspace.buffers, {})
        rows = torch.zeros((2048, 32), dtype=torch.bfloat16)
        self.assertIsNone(
            transformer.layers[0].ffn._prepare_prefill_workspace(rows, True)
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            cuda_rows = SimpleNamespace(
                is_cuda=True, dtype=torch.bfloat16, size=lambda _: 2048
            )
            self.assertIsNone(
                transformer.layers[0].ffn._prepare_prefill_workspace(cuda_rows, True)
            )
        workspace.partitions = 1
        workspace.buffers["gather_input"] = torch.empty((3, 2))
        product = torch.arange(6).float().view(3, 2)
        positions = torch.tensor((2, 0, 1))
        with patch(
            "torch.distributed.all_gather_into_tensor",
            side_effect=AssertionError("a single partition must not use a collective"),
        ):
            for size in (0, 3):
                selected = workspace._gather_product(product, positions[:size])
                self.assertTrue(torch.equal(selected, product[positions[:size]]))
                self.assertIs(workspace._gather_sum(selected), selected)

    def test_moe_prefill_keeps_collective_waves_and_reuses_storage(self):
        from oh_my_vllm.models.deepseek_v41 import kernel
        from oh_my_vllm.models.deepseek_v41.moe_prefill import PrefillWorkspace

        class Native:
            def __init__(self):
                self.projections = []
                self.collectives = []

            def grouped_workspace_size(self, *_args):
                return 64

            def quantize(self, values, encoded, scales, *_args):
                encoded.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
                scales.fill_(127)

            def grouped_project(
                self,
                encoded,
                scales,
                role,
                first_expert,
                counts,
                output,
                scratch,
                max_rows,
            ):
                self.projections.append((role, encoded.size(0)))
                values = encoded.view(torch.float8_e4m3fn).float()
                output.copy_(values[:, :1].expand_as(output))
                if role == "down":
                    first = 0
                    for expert, count in enumerate(counts.tolist(), first_expert):
                        output[first : first + count].add_(expert / 8)
                        first += count

            def swiglu(self, gate, up, weights, output, *_args):
                output.copy_(gate.float() * weights[:, None])

            def expert_sum(self, values, route_rows, output):
                output.zero_()
                for row, routes in enumerate(route_rows.tolist()):
                    for route in sorted(route for route in routes if route >= 0):
                        output[row].add_(values[route].float())

            def all_reduce(self, values):
                self.collectives.append(("reduce", tuple(values.shape)))
                values.add_(2)

            def all_gather_into_tensor(self, output, values):
                self.collectives.append(
                    (
                        "product" if values.dtype == torch.bfloat16 else "sum",
                        tuple(values.shape),
                    )
                )
                for rank, shard in enumerate(output.view(4, *values.shape)):
                    shard.copy_(values)
                    if values.dtype == torch.float32:
                        shard.add_(rank / 8)

        def gate(value, image_mask=None):
            ids = (value[:, 1].long()[:, None] + torch.tensor([0, 3, 6])) % 8
            weights = torch.tensor([0.25, 0.5, 0.75]).expand(value.size(0), -1)
            return weights, ids

        for strategy in ("TEP4", "TP4"):
            for wave in (4096, 8192):
                with self.subTest(strategy=strategy, wave=wave), ExitStack() as stack:
                    stack.enter_context(patch.object(model, "world_size", 4))
                    stack.enter_context(patch.object(model, "rank", 1))
                    native = Native()
                    for module in (
                        "_projection_module",
                        "_transform_module",
                        "_activation_module",
                        "_routing_module",
                    ):
                        stack.enter_context(
                            patch.object(kernel, module, return_value=native)
                        )
                    stack.enter_context(
                        patch.object(model.dist, "all_reduce", native.all_reduce)
                    )
                    stack.enter_context(
                        patch.object(
                            model.dist,
                            "all_gather_into_tensor",
                            native.all_gather_into_tensor,
                        )
                    )
                    dim = 128 if strategy == "TP4" else 32
                    moe = MoE(
                        0,
                        ModelArgs(
                            dim=dim,
                            moe_inter_dim=128,
                            n_routed_experts=8,
                            n_activated_experts=3,
                            dtype="bf16",
                            expert_dtype=None,
                            parallel_strategy=strategy,
                        ),
                    )
                    groups = moe.n_local_experts
                    projection = tuple(
                        (torch.empty((32, dim // 2)), None) for _ in range(groups)
                    )
                    down = tuple(
                        (torch.empty((32, 64 if strategy == "TP4" else 16)), None)
                        for _ in range(groups)
                    )
                    moe._routed_state.weights = SimpleNamespace(
                        gate=projection, up=projection, down=down
                    )
                    moe._routed_state.pointers = tuple(
                        (role, moe.experts_start_idx) for role in ("gate", "up", "down")
                    )
                    workspace = PrefillWorkspace(wave)
                    workspace.reserve(moe, torch.device("cpu"), 4)
                    addresses = {
                        name: value.data_ptr()
                        for name, value in workspace.buffers.items()
                    }
                    stack.enter_context(
                        patch.object(
                            moe, "_prepare_prefill_workspace", return_value=workspace
                        )
                    )
                    gate_mock = stack.enter_context(
                        patch.object(moe.gate, "forward", side_effect=gate)
                    )
                    for length in (wave + 1025, 1025):
                        native.projections.clear()
                        native.collectives.clear()
                        gate_mock.reset_mock()
                        value = torch.zeros((1, length, dim), dtype=torch.bfloat16)
                        value[0, :, 0] = torch.arange(length) % 3
                        value[0, :, 1] = torch.arange(length) % 8
                        shared = torch.full((length, dim), -0.5, dtype=torch.bfloat16)
                        with patch.object(
                            moe.shared_experts, "forward", return_value=shared
                        ):
                            actual = moe(value, prefill_eager=True)
                        weights, ids = gate(value[0])
                        expected = torch.zeros((length, dim), dtype=torch.float32)
                        for expert in range(moe.experts_start_idx, moe.experts_end_idx):
                            token, slot = (ids == expert).nonzero(as_tuple=True)
                            contribution = (
                                value[0, token, 0].float() * weights[token, slot]
                                + expert / 8
                            ).to(torch.bfloat16)
                            expected[token] += contribution[:, None].float()
                        if strategy == "TEP4":
                            expected += 2
                        else:
                            expected += (
                                torch.arange(4).repeat_interleave(32)[None, :] / 8
                            )
                        expected = (
                            (expected + shared).to(torch.bfloat16).view_as(actual)
                        )
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int16), expected.view(torch.int16)
                            )
                        )
                        collective_rows = [
                            min(1024, length - first)
                            for first in range(0, length, 1024)
                        ]
                        expected_collectives = []
                        for rows in collective_rows:
                            if strategy == "TEP4":
                                expected_collectives.append(("reduce", (rows, dim)))
                            else:
                                expected_collectives.extend(
                                    (("product", (rows * 3, 32)), ("sum", (rows, 32)))
                                )
                        self.assertEqual(native.collectives, expected_collectives)
                        self.assertEqual(
                            [call.args[0].size(0) for call in gate_mock.call_args_list],
                            collective_rows,
                        )
                        self.assertEqual(
                            max(
                                count
                                for role, count in native.projections
                                if role == "gate"
                            ),
                            min(wave, length) * 3
                            if strategy == "TP4"
                            else int(
                                (
                                    (ids[: min(wave, length)] >= 2)
                                    & (ids[: min(wave, length)] < 4)
                                ).sum()
                            ),
                        )
                        workspace.reserve(moe, torch.device("cpu"), 4)
                        self.assertEqual(
                            addresses,
                            {
                                name: value.data_ptr()
                                for name, value in workspace.buffers.items()
                            },
                        )
                    if strategy == "TEP4":
                        native.projections.clear()
                        native.collectives.clear()
                        gate_mock.reset_mock()
                        indices = torch.tensor([[0, 4, 6]]).expand(1025, -1)
                        weights = torch.ones((1025, 3))
                        routing = workspace.routing(
                            moe, value[0], None, (weights, indices)
                        )
                        actual = moe._forward_rows(
                            value[0],
                            routing=routing,
                            shared_output=shared,
                            workspace=workspace,
                        )
                        self.assertTrue(
                            torch.equal(actual, torch.full_like(actual, 1.5))
                        )
                        self.assertEqual(native.projections, [])
                        self.assertEqual(
                            native.collectives,
                            [("reduce", (1024, dim)), ("reduce", (1, dim))],
                        )
                        gate_mock.assert_not_called()

    def test_route_counts_match_bincount_without_a_host_read(self):
        indices = torch.tensor([0, 3, 3, 5, 383, 0], dtype=torch.int32)
        expected = torch.bincount(indices.to(torch.int64), minlength=384).to(
            torch.int32
        )

        actual = _route_counts(indices, 384)

        self.assertEqual(actual.dtype, torch.int32)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            _route_counts(torch.empty(0, dtype=torch.int32), 4),
            torch.zeros(4, dtype=torch.int32),
        )

    def test_rank_loader_keeps_full_down_k_and_matching_output_row_scales(self):
        config = json.loads(
            (Path(__file__).parent / "fixtures/deepseek_v41/config.json").read_text()
        )
        layout, source, expected = {}, {}, {}
        for prefix in ("layers.0", "mtp.2"):
            name = f"{prefix}.ffn.experts.0.w2"
            weight = (torch.arange(128)[None, :] + torch.arange(128)[:, None] * 3).to(
                torch.int8
            )
            scales = (torch.arange(128) % 15 + 120).to(torch.uint8)
            scales = scales[:, None].expand(128, 8).contiguous()
            for suffix, value, dtype in (
                ("weight", weight, "I8"),
                ("scale", scales.view(torch.float8_e8m0fnu), "F8_E8M0"),
            ):
                key = f"{name}.{suffix}"
                layout[key] = {"dtype": dtype, "shape": list(value.shape)}
                source[key] = value
                expected[key] = value.view(torch.uint8)
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(source, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, "weights.safetensors")})
            )
            for rank in range(4):
                with patch(
                    "oh_my_vllm.models.deepseek_v41.loader.tensor_layout",
                    return_value=layout,
                ):
                    loader = RankLoader(root, config, rank, strategy="TP4")
                try:
                    for name, value in expected.items():
                        for device in devices:
                            with self.subTest(rank=rank, name=name, device=device):
                                destination = torch.empty(
                                    (32, value.size(1)),
                                    dtype=(
                                        torch.float4_e2m1fn_x2
                                        if name.endswith("weight")
                                        else torch.float8_e8m0fnu
                                    ),
                                    device=device,
                                )
                                loader._copy_parameter(name, destination)
                                self.assertTrue(
                                    torch.equal(
                                        destination.cpu().view(torch.uint8),
                                        value[rank * 32 : (rank + 1) * 32],
                                    )
                                )
                finally:
                    loader.close()

    def test_shared_expert_forward_uses_full_weights_after_routed_exchange(self):
        args = ModelArgs(
            dim=4,
            moe_inter_dim=8,
            n_layers=1,
            n_routed_experts=4,
            n_activated_experts=1,
            dspark_n_routed_experts=8,
            expert_dtype=None,
        )
        values = torch.ones((1, 1, 4), dtype=torch.bfloat16)
        gate_weight = torch.tensor([[0.5, 0.5, 0, 0]] * 8, dtype=torch.bfloat16)
        up_weight = torch.tensor([[1.0, 0.0, 0, 0]] * 8, dtype=torch.bfloat16)
        down_weight = torch.tensor([[0.125] * 8] * 4, dtype=torch.bfloat16)
        product = torch.nn.functional.silu(torch.ones(1, 8)).to(torch.bfloat16)
        shared_output = torch.nn.functional.linear(product, down_weight)
        expected = (shared_output.float() + 3).to(torch.bfloat16).view_as(values)
        for strategy in ("TP4", "TEP4", "DEP4"):
            for rank in range(4):
                for layer in (0, 1):
                    with (
                        self.subTest(strategy=strategy, rank=rank, layer=layer),
                        patch("oh_my_vllm.models.deepseek_v41.model.world_size", 4),
                        patch("oh_my_vllm.models.deepseek_v41.model.rank", rank),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model.default_dtype",
                            torch.bfloat16,
                        ),
                    ):
                        moe = MoE(layer, replace(args, parallel_strategy=strategy))
                        if strategy == "DEP4":
                            moe.dep4_exchange = lambda _layer, rows, _routing: (
                                torch.full_like(rows, 3, dtype=torch.float32)
                            )
                        shared = moe.shared_experts
                        self.assertEqual(tuple(shared.w1.weight.shape), (8, 4))
                        self.assertEqual(tuple(shared.w2.weight.shape), (4, 8))
                        self.assertEqual(tuple(shared.w3.weight.shape), (8, 4))
                        with torch.no_grad():
                            for parameter in moe.parameters():
                                parameter.zero_()
                            shared.w1.weight.copy_(gate_weight)
                            shared.w2.weight.copy_(down_weight)
                            shared.w3.weight.copy_(up_weight)

                        def reduce_routed(value):
                            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
                            value.fill_(3)

                        def gather_routed(output, value):
                            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
                            output.fill_(3 if value.dtype == torch.float32 else 0)

                        with (
                            patch(
                                "oh_my_vllm.models.deepseek_v41.model.dist.all_reduce",
                                side_effect=reduce_routed,
                            ),
                            patch(
                                "oh_my_vllm.models.deepseek_v41.model.dist.all_gather_into_tensor",
                                side_effect=gather_routed,
                            ),
                        ):
                            actual = moe(values)
                        self.assertTrue(torch.equal(actual, expected))

    def test_rank_loader_preserves_full_shared_fp8_values_and_scales(self):
        config = json.loads(
            (Path(__file__).parent / "fixtures/deepseek_v41/config.json").read_text()
        )
        shapes = {
            "w1.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w1.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
            "w2.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w2.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
            "w3.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w3.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
        }
        layout, source, expected = {}, {}, {}
        for namespace in ("layers", "mtp"):
            for index, (suffix, (shape, dtype, torch_dtype)) in enumerate(
                shapes.items()
            ):
                name = f"{namespace}.0.ffn.shared_experts.{suffix}"
                layout[name] = {"shape": list(shape), "dtype": dtype}
                raw = (torch.arange(math.prod(shape)) % 16 + 112 + index).to(
                    torch.uint8
                )
                expected[name] = raw.reshape(shape)
                source_name = name.replace("ffn", "mlp").replace(
                    ".scale", ".weight_scale_inv"
                )
                source[f"model.{source_name}"] = raw.view(torch_dtype).reshape(shape)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model-00001-of-00048.safetensors"
            save_file(source, shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, shard.name)})
            )
            before = hashlib.sha256(shard.read_bytes()).digest()
            for strategy in ("TP4", "TEP4", "DEP4"):
                for rank in range(4):
                    with (
                        self.subTest(strategy=strategy, rank=rank),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.loader.tensor_layout",
                            return_value=layout,
                        ),
                        patch("oh_my_vllm.models.deepseek_v41.model.world_size", 4),
                        patch("oh_my_vllm.models.deepseek_v41.model.rank", rank),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model.default_dtype",
                            torch.float8_e4m3fn,
                        ),
                    ):
                        model = torch.nn.Module()
                        for namespace in ("layers", "mtp"):
                            block = torch.nn.Module()
                            block.ffn = torch.nn.Module()
                            block.ffn.shared_experts = MoE(
                                0,
                                ModelArgs(
                                    dim=128,
                                    moe_inter_dim=128,
                                    n_routed_experts=4,
                                    parallel_strategy=strategy,
                                ),
                            ).shared_experts
                            model.add_module(namespace, torch.nn.ModuleList([block]))
                        loader = RankLoader(
                            root, config, rank, strategy=strategy, tile_rows=17
                        )
                        try:
                            for name, value in expected.items():
                                self.assertEqual(
                                    loader.specs[name]["shape"], tuple(value.shape)
                                )
                            info = loader.load_into(model)
                            self.assertEqual(info["parameter_count"], 12)
                            for name, value in model.named_parameters():
                                self.assertTrue(
                                    torch.equal(
                                        value.view(torch.uint8), expected[name]
                                    ),
                                    name,
                                )
                            for blocks in (model.layers, model.mtp):
                                for projection in ("w1", "w2", "w3"):
                                    linear = getattr(
                                        blocks[0].ffn.shared_experts, projection
                                    )
                                    self.assertIs(linear.weight.scale, linear.scale)
                        finally:
                            loader.close()
            self.assertEqual(hashlib.sha256(shard.read_bytes()).digest(), before)

    def test_dense_owner_keeps_small_rank_term_before_bf16_rounding(self):
        from oh_my_vllm.models.deepseek_v41.local_tp import LocalRowProjection

        # Captured TEP4 values at layer 21, query 158, channel 348.
        partials = torch.tensor(
            [-0.04150390625, -2.2351741790771484e-08, -0.462890625, 0.11669921875],
            dtype=torch.bfloat16,
        )
        calls = []

        def project(values, weight):
            rank = int(weight[0, 0])
            calls.append((rank, tuple(values.shape), values.dtype))
            return partials[rank].expand(values.size(0), 1).clone()

        with torch.device("cpu"):
            inputs = torch.ones((2, 4), dtype=torch.bfloat16)
            weights = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
            actual = LocalRowProjection(4)(inputs, weights, None, project)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                actual, torch.full((2, 1), -0.3876953423023224, dtype=torch.float32)
            )
        )
        self.assertTrue(
            torch.equal(
                actual.to(torch.bfloat16),
                torch.full((2, 1), -0.388671875, dtype=torch.bfloat16),
            )
        )
        self.assertEqual(calls, [(rank, (2, 1), torch.bfloat16) for rank in range(4)])

    def test_attention_output_rounds_before_tensor_parallel_reduction(self):
        with patch("oh_my_vllm.models.deepseek_v41.model.world_size", 4):
            attention = RowParallelLinear(8, 2, dtype=torch.bfloat16)
            with torch.no_grad():
                attention.weight.copy_(torch.tensor([[1, 1 / 256]] * 2))

            def reduce(value):
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, torch.ones((1, 2)), rtol=0, atol=0)
                value.sub_(1)

            with patch(
                "oh_my_vllm.models.deepseek_v41.model.dist.all_reduce",
                side_effect=reduce,
            ):
                actual = attention(torch.ones((1, 2), dtype=torch.bfloat16))
            torch.testing.assert_close(
                actual, torch.zeros((1, 2), dtype=torch.bfloat16), rtol=0, atol=0
            )

    def test_router_keeps_float32_projection_before_expert_selection(self):
        args = ModelArgs(
            dim=2,
            n_routed_experts=6,
            n_activated_experts=3,
            route_scale=1.5,
            score_func="sqrtsoftplus",
            vision_n_layers=0,
        )
        gate = Gate(0, args)
        gate.weight = torch.nn.Parameter(
            torch.tensor(
                [[1, 1 / 256], [1, 0], [2, 0], [3, 0], [-1, 0], [-2, 0]],
                dtype=torch.bfloat16,
            )
        )
        with torch.no_grad():
            gate.bias.zero_()
        weights, indices = gate(torch.ones((1, 2), dtype=torch.bfloat16))
        scores = [math.sqrt(math.log1p(math.exp(value))) for value in (3, 2, 257 / 256)]
        expected = torch.tensor(
            [[value / sum(scores) * 1.5 for value in scores]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(indices, torch.tensor([[3, 2, 0]])))
        torch.testing.assert_close(weights, expected, rtol=2e-7, atol=2e-7)

    def test_candidate_selection_keeps_a_compact_causal_block_mask(self):
        scores = torch.tensor(
            [[[1.0, 2, 3, 4, 5], [3.0, 2, 1, -torch.inf, -torch.inf]]]
        )
        lengths = torch.tensor([[[5], [3]]])
        actual = select_candidate_blocks(scores, lengths, 1, 2)
        expected = torch.tensor([[[False, False, True], [False, True, False]]])
        self.assertTrue(torch.equal(actual, expected))

    def test_engram_hash_reads_only_the_requested_lookback_positions(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        state = NgramHashState.__new__(NgramHashState)
        torch.nn.Module.__init__(state)
        state.layout = SimpleNamespace(max_ngram_size=2)
        state.pad_id = 0
        state.cache = torch.zeros(3, 1024, dtype=torch.int64)
        state.cache[2, 511] = 7
        state.cache[0, 511] = 3
        state.token_map = torch.arange(8)
        state.multipliers = torch.tensor([[1, 3]])
        state.primes = torch.tensor([[[11]]])
        state.offsets = torch.tensor([[0]])

        class ActiveRowsOnly(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                if (
                    function == torch.ops.aten.index.Tensor
                    and args[0] is state.cache
                    and len(args[1]) < 2
                ):
                    raise AssertionError(
                        "Engram selected the full context before lookback"
                    )
                return function(*args, **(kwargs or {}))

        with ActiveRowsOnly():
            actual = state(
                torch.tensor([[1, 2], [4, 5]]),
                start_pos=512,
                cache_slots=torch.tensor([2, 0]),
            )
        self.assertTrue(
            torch.equal(actual, torch.tensor([[[[9]], [[1]]], [[[2]], [[9]]]]))
        )
        self.assertEqual(torch.count_nonzero(state.cache[1]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_lazy_cache_maps_only_selected_slots_and_releases_finished_state(self):
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
        from oh_my_vllm.operators.runtime.paged import mapped_bytes

        previous_device, previous_dtype = (
            torch.get_default_device(),
            torch.get_default_dtype(),
        )
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(2,),
                kv_source_layers=(0,),
                index_source_layers=(0,),
                index_n_heads=4,
                index_head_dim=32,
                index_topk=4,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=64,
                max_seq_len=1048576,
                lazy_cache=True,
            )
            before = mapped_bytes()
            model = Transformer(args)
            self.assertEqual(mapped_bytes(), before)
            reference = Transformer(
                replace(args, max_batch_size=2, max_seq_len=16, lazy_cache=False)
            )
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0, std=0.02)
            reference.load_state_dict(model.state_dict())
            cache = DeepSeekCache(model, block_tokens=2, max_seq_len=args.max_seq_len)
            row = {"fa_block_table": [1, 2], "mamba_block_table": [0, 4]}
            cache.begin_step(row, 63, start_pos=0, end_pos=4)
            cache.begin_step(
                {"fa_block_table": [5, 6], "mamba_block_table": [0, 7]},
                2,
                start_pos=0,
                end_pos=4,
            )
            tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            output = model(
                tokens,
                cache_slots=torch.tensor([63, 2]),
                cache_slots_cpu=(63, 2),
                full_logits=True,
            )
            expected = reference(tokens, full_logits=True)
            torch.testing.assert_close(output[1], expected[1], atol=2e-4, rtol=2e-3)
            active = mapped_bytes() - before
            self.assertGreater(active, 0)
            self.assertLess(active, 64 * 1024 * 1024)
            attention = model.layers[0].attn
            saved = attention.compress_kv_cache[2, :2].clone()
            restored = attention.compress_kv_cache[63, :2].clone()
            restored_window = attention.window_kv_cache[63].clone()
            cache.commit(7, row, 63, accepted_end=4)
            cache.clear_slot(63)
            self.assertEqual(mapped_bytes() - before, active)
            torch.testing.assert_close(attention.compress_kv_cache[2, :2], saved)
            cache.restore(7, 10, row, start_pos=4)
            torch.testing.assert_close(attention.compress_kv_cache[10, :2], restored)
            torch.testing.assert_close(attention.window_kv_cache[10], restored_window)
            self.assertEqual(mapped_bytes() - before, active)
            cache.clear_slot(10)
            cache.clear_slot(2)
            self.assertEqual(mapped_bytes() - before, active)
            cache.close()
            self.assertEqual(mapped_bytes(), before)
        finally:
            shared_attn.reset_query_state()
            shared_attn.compress_kv = shared_attn.index_k = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_compressed_cache_publishes_current_groups_for_selected_slots(self):
        attention = Attention.__new__(Attention)
        torch.nn.Module.__init__(attention)
        attention.compress_ratio = 2
        attention.rope_head_dim = 16
        attention.is_kv_source = True
        attention.is_index_source = False
        attention.compress_kv_cache = torch.zeros(4, 8, 32)
        attention.compress_kv_cache[2, 0].fill_(3)
        attention.compress_kv_cache[0, 0].fill_(1)
        attention.freqs_cis = torch.ones(16, 8, dtype=torch.complex64)
        attention.compressor = Compressor(
            ModelArgs(dim=64, head_dim=32, compress_ratios=(2,), max_batch_size=4),
            0,
        )
        for parameter in attention.compressor.parameters():
            torch.nn.init.constant_(parameter, 0.01)
        slots = torch.tensor([2, 0])
        value = torch.arange(1, 257, dtype=torch.float32).reshape(2, 2, 64) / 256
        previous_cache, previous_indices = (
            shared_attn.compress_kv,
            shared_attn.topk_idxs,
        )
        try:
            shared_attn.topk_idxs = torch.tensor([[[0]], [[0]]])
            current, indices = attention._compress_kv(value, value, 2, 4, slots)
            expected = attention.compress_kv_cache[slots, :2]
            self.assertTrue(torch.count_nonzero(expected[:, 1]).item())
            torch.testing.assert_close(current, expected)
            self.assertEqual(
                tuple(shared_attn.compress_kv.shape), tuple(expected.shape)
            )
            self.assertIs(indices, shared_attn.topk_idxs)
            attention.is_kv_source = False
            reused, _ = attention._compress_kv(value, value, 2, 4, slots)
            torch.testing.assert_close(reused, expected)
            self.assertEqual(
                torch.count_nonzero(attention.compress_kv_cache[1]).item(), 0
            )
        finally:
            shared_attn.compress_kv, shared_attn.topk_idxs = (
                previous_cache,
                previous_indices,
            )

    def test_moe_merges_routed_experts_and_shared_output(self):
        class FixedGate(torch.nn.Module):
            def forward(self, value, image_mask=None):
                weights = torch.tensor(
                    (
                        (0.25, 0.75),
                        (0.5, 0.5),
                        (0.75, 0.25),
                        (0.6, 0.4),
                    ),
                    dtype=value.dtype,
                )
                indices = torch.tensor(
                    (
                        (0, 2),
                        (1, 3),
                        (2, 0),
                        (3, 1),
                    ),
                    dtype=torch.long,
                )
                return weights[: value.size(0)], indices[: value.size(0)]

        class FixedExpert(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, value, weights):
                return (value + self.offset) * weights

        class ZeroExpert(torch.nn.Module):
            def forward(self, value):
                return torch.zeros_like(value)

        args = ModelArgs(
            dim=2,
            moe_inter_dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype=None,
            parallel_strategy="TP4",
        )
        moe = MoE(0, args)
        moe.gate = FixedGate()
        moe.experts = torch.nn.ModuleList(FixedExpert(index + 1) for index in range(4))
        moe.shared_experts = ZeroExpert()
        value = torch.tensor(
            (
                ((1.0, 2.0),),
                ((3.0, 4.0),),
                ((5.0, 6.0),),
                ((7.0, 8.0),),
            )
        )

        result = moe(value)

        expected = torch.tensor(
            (
                ((3.5, 4.5),),
                ((6.0, 7.0),),
                ((7.5, 8.5),),
                ((10.2, 11.2),),
            )
        )
        torch.testing.assert_close(result, expected)

    def test_fp8_expert_quantizes_shared_up_input_once(self):
        expert = model.Expert(32, 32, dtype=torch.float8_e4m3fn)
        with torch.no_grad():
            for projection, value in (
                (expert.w1, 1.0),
                (expert.w3, 0.5),
                (expert.w2, 0.25),
            ):
                projection.weight.copy_(
                    torch.full(projection.weight.shape, value).to(torch.float8_e4m3fn)
                )
                projection.scale.view(torch.uint8).fill_(127)
        values = torch.ones((4, 32), dtype=torch.bfloat16)
        with patch.object(model, "act_quant", wraps=model.act_quant) as quantize:
            actual = expert(values)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 4096), rtol=0, atol=0
        )
        self.assertEqual(quantize.call_count, 2)

    def test_expert_keeps_replaced_projection_calls(self):
        values = torch.ones((4, 32), dtype=torch.bfloat16)
        expert = model.Expert(32, 32, dtype=torch.bfloat16)
        expert.w1, expert.w3, expert.w2 = (torch.nn.Identity() for _ in range(3))
        torch.testing.assert_close(
            expert(values), torch.full_like(values, 0.73046875), rtol=0, atol=0
        )

        class ZeroLinear(model.Linear):
            def forward(self, values):
                return torch.zeros_like(values)

        for mode in ("subclass", "instance", "foreign_instance", "hook", "global_hook"):
            with self.subTest(mode=mode), ExitStack() as stack:
                expert = model.Expert(32, 32, dtype=torch.float8_e4m3fn)
                if mode == "subclass":
                    expert.w1 = ZeroLinear(32, 32, dtype=torch.float8_e4m3fn)
                with torch.no_grad():
                    for projection in (expert.w1, expert.w3, expert.w2):
                        projection.weight.copy_(
                            torch.ones(projection.weight.shape).to(torch.float8_e4m3fn)
                        )
                        projection.scale.view(torch.uint8).fill_(127)

                def zero_gate(module, inputs, output, gate=expert.w1):
                    return torch.zeros_like(output) if module is gate else None

                if mode == "instance":
                    stack.enter_context(
                        patch.object(
                            expert.w1, "forward", return_value=torch.zeros_like(values)
                        )
                    )
                elif mode == "foreign_instance":
                    replacement = model.Linear(32, 32, dtype=torch.float8_e4m3fn)
                    with torch.no_grad():
                        replacement.weight.view(torch.uint8).zero_()
                        replacement.scale.view(torch.uint8).fill_(127)
                    stack.enter_context(
                        patch.object(expert.w1, "forward", replacement.forward)
                    )
                elif mode == "hook":
                    stack.enter_context(expert.w1.register_forward_hook(zero_gate))
                elif mode == "global_hook":
                    stack.enter_context(
                        torch.nn.modules.module.register_module_forward_hook(zero_gate)
                    )
                actual = expert(values)
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), rtol=0, atol=0
                )

    def test_moe_bounds_routed_rows_without_changing_outputs(self):
        class FixedGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value, image_mask=None):
                self.calls += 1
                weights = torch.tensor(
                    ((0.25, 0.75),),
                    dtype=value.dtype,
                    device=value.device,
                ).expand(value.size(0), -1)
                indices = torch.tensor(
                    ((0, 1),),
                    dtype=torch.long,
                    device=value.device,
                ).expand(value.size(0), -1)
                return weights, indices

        class FixedExpert(torch.nn.Module):
            def __init__(self, output):
                super().__init__()
                self.output = output

            def forward(self, value, weights):
                return torch.full_like(value, self.output)

        class SharedExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rows = []

            def forward(self, value):
                self.rows.append(value.size(0))
                output = value.clone()
                output[:, 0] = -1
                return output

        args = ModelArgs(
            dim=2,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype=None,
            parallel_strategy="TP4",
        )
        moe = MoE(0, args)
        gate = FixedGate()
        moe.gate = gate
        moe.experts = torch.nn.ModuleList(
            FixedExpert(output) for output in (1, 1 / 256)
        )
        moe.shared_experts = SharedExpert()
        value = torch.zeros((1025, 1, 2), dtype=torch.bfloat16)
        value[:, 0, 1] = torch.arange(1025) % 3

        result = moe(value)

        expected = torch.full_like(value, 1 / 256)
        expected[:, 0, 1] = value[:, 0, 1] + 1
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertGreater(gate.calls, 1)
        self.assertEqual(moe.shared_experts.rows, [1025])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_moe_grouped_fp4_path_matches_expert_reference(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            torch.manual_seed(0)
            args = ModelArgs(
                dim=32,
                moe_inter_dim=32,
                n_routed_experts=4,
                n_activated_experts=2,
                n_shared_experts=1,
                dtype="bf16",
                expert_dtype="fp4",
                parallel_strategy="TP4",
                swiglu_limit=10,
            )
            moe = MoE(0, args).eval()
            for expert in moe.experts:
                for linear in (expert.w1, expert.w2, expert.w3):
                    raw = torch.randint(
                        0, 256, linear.weight.shape, device="cuda", dtype=torch.uint8
                    )
                    linear.weight.data.copy_(raw.view(torch.float4_e2m1fn_x2))
                    linear.scale.data.fill_(1)

            class FixedGate(torch.nn.Module):
                def forward(self, value, image_mask=None):
                    weights = torch.tensor(
                        (
                            (0.25, 0.75),
                            (0.5, 0.5),
                            (0.75, 0.25),
                            (0.6, 0.4),
                        ),
                        device=value.device,
                        dtype=torch.float32,
                    )
                    indices = torch.tensor(
                        ((0, 2), (1, 3), (2, 0), (3, 1)),
                        device=value.device,
                        dtype=torch.long,
                    )
                    return weights[: value.size(0)], indices[: value.size(0)]

            class ZeroExpert(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.gate = FixedGate()
            moe.shared_experts = ZeroExpert()
            value = torch.randn((4, 1, 32), device="cuda", dtype=torch.bfloat16)
            actual = moe(value)

            rows = value.view(-1, 32)
            weights, indices = moe.gate(rows)
            flat_indices = indices.flatten()
            order = torch.argsort(flat_indices)
            sorted_tokens = (
                torch.arange(rows.size(0), device="cuda")
                .view(-1, 1)
                .expand_as(indices)
                .flatten()[order]
            )
            sorted_weights = weights.flatten()[order]
            sorted_rows = rows.index_select(0, sorted_tokens)
            counts = torch.bincount(flat_indices, minlength=4).tolist()
            expected_rows = torch.zeros_like(rows, dtype=torch.float32)
            first = 0
            for index, count in enumerate(counts):
                last = first + count
                if count:
                    expected_rows.index_add_(
                        0,
                        sorted_tokens[first:last],
                        moe.experts[index](
                            sorted_rows[first:last],
                            sorted_weights[first:last].unsqueeze(-1),
                        ).float(),
                    )
                first = last
            expected = expected_rows.to(value.dtype).view_as(value)
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_moe_device_counts_avoid_the_host_read_for_many_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        world_size_patch = patch.object(model, "world_size", 1)
        world_size_patch.start()
        try:
            torch.manual_seed(1)
            args = ModelArgs(
                dim=32,
                moe_inter_dim=32,
                n_routed_experts=4,
                n_activated_experts=2,
                n_shared_experts=1,
                dtype="bf16",
                expert_dtype="fp4",
                parallel_strategy="TP4",
                swiglu_limit=10,
            )
            moe = MoE(0, args).eval()
            for expert in moe.experts:
                for linear in (expert.w1, expert.w2, expert.w3):
                    raw = torch.randint(
                        0, 256, linear.weight.shape, device="cuda", dtype=torch.uint8
                    )
                    linear.weight.data.copy_(raw.view(torch.float4_e2m1fn_x2))
                    linear.scale.data.fill_(1)

            rows_count = 10  # two activated experts give 20 routed rows
            positions = torch.arange(rows_count, device="cuda")

            class FixedGate(torch.nn.Module):
                def forward(self, value, image_mask=None):
                    weights = torch.linspace(
                        0.1, 0.9, rows_count * 2, device=value.device
                    ).view(rows_count, 2)
                    indices = torch.stack(
                        (positions % 4, (positions + 1) % 4), dim=1
                    ).long()
                    return weights, indices

            class ZeroExpert(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.gate = FixedGate()
            moe.shared_experts = ZeroExpert()
            value = torch.randn(
                (rows_count, 1, 32), device="cuda", dtype=torch.bfloat16
            )

            rows = value.view(-1, 32)
            weights, indices = moe.gate(rows)
            flat_indices = indices.flatten()
            order = torch.argsort(flat_indices)
            sorted_tokens = (
                torch.arange(rows.size(0), device="cuda")
                .view(-1, 1)
                .expand_as(indices)
                .flatten()[order]
            )
            sorted_weights = weights.flatten()[order]
            sorted_rows = rows.index_select(0, sorted_tokens)
            counts = torch.bincount(flat_indices, minlength=4).tolist()
            expected_rows = torch.zeros_like(rows, dtype=torch.float32)
            first = 0
            for index, count in enumerate(counts):
                last = first + count
                if count:
                    expected_rows.index_add_(
                        0,
                        sorted_tokens[first:last],
                        moe.experts[index](
                            sorted_rows[first:last],
                            sorted_weights[first:last].unsqueeze(-1),
                        ).float(),
                    )
                first = last
            expected = expected_rows.to(value.dtype).view_as(value)

            original = torch.Tensor.tolist

            def guarded(tensor):
                if tensor.is_cuda:
                    raise AssertionError("host read of a CUDA routed count")
                return original(tensor)

            with patch.object(torch.Tensor, "tolist", guarded):
                actual = moe(value)
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)
        finally:
            world_size_patch.stop()
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp8_projection_accepts_long_prefill_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            torch.manual_seed(11)
            values = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            weight_scale = torch.ones((1, 4), device="cuda", dtype=torch.float8_e8m0fnu)
            encoded, scales = act_quant(
                values,
                block_size=32,
                scale_fmt="ue8m0",
                scale_dtype=torch.float8_e8m0fnu,
            )

            result = fp8_gemm(
                encoded,
                scales,
                weight,
                weight_scale,
                block_size=32,
            )

            self.assertEqual(tuple(result.shape), (512, 32))
            self.assertTrue(torch.isfinite(result).all())
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_transformer_can_skip_vocab_projection_for_prefill(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=0,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
                dspark_target_layer_ids=(0,),
            )
            model = Transformer(args)
            _, logits, hidden = model(
                torch.randint(0, 128, (1, 2)),
                compute_logits=False,
            )
            self.assertIsNone(logits)
            self.assertEqual(tuple(hidden.shape), (1, 2, 64))
            _, logits, hidden = model(
                torch.randint(0, 128, (1, 2)),
                compute_logits=False,
                collect_main_hidden=False,
            )
            self.assertIsNone(logits)
            self.assertIsNone(hidden)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_transformer_clears_shared_attention_state_before_forward(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        shared_attn.topk_idxs = torch.ones(4, 1, 1, dtype=torch.int32)
        shared_attn.candidates = torch.ones(4, 1, dtype=torch.bool)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=0,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            Transformer(args)(torch.randint(0, 128, (1, 1)))
            self.assertIsNone(shared_attn.topk_idxs)
            self.assertIsNone(shared_attn.candidates)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_index_cache_tracks_cache_slots_when_a_group_is_incomplete(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(2, 2),
                kv_source_layers=(0,),
                index_source_layers=(0, 1),
                index_topk=4,
                candidate_source_layer=0,
                candidate_topk_blocks=1,
                candidate_block_size=2,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=4,
                max_seq_len=32,
            )
            model = Transformer(args)
            model(torch.randint(0, 128, (4, 4)), cache_slots=torch.arange(4))
            output = model(
                torch.randint(0, 128, (1, 1)),
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )
            self.assertEqual(tuple(output[0].shape), (1,))
            self.assertEqual(
                tuple(shared_attn.index_k.shape), (1, 2, args.index_head_dim)
            )
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_device_window_indices_match_the_integer_path(self):
        for window_size in (1, 4, 8):
            for position in range(1, 2 * window_size + 3):
                eager = get_window_topk_idxs(window_size, 2, 1, position)
                device = position_window_topk_idxs(
                    window_size, 2, torch.tensor(position, dtype=torch.int64)
                )
                self.assertTrue(torch.equal(eager, device))

    def test_device_position_decode_matches_the_integer_path(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=3,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0, 2, 1),
                kv_source_layers=(1, 2),
                index_source_layers=(1, 2),
                index_n_heads=4,
                index_head_dim=32,
                index_topk=4,
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=1,
                max_seq_len=16,
            )
            model = Transformer(args)
            # Equal top-k cutoff scores can select different keys after context padding.
            generator = torch.Generator(device="cpu").manual_seed(0)
            for parameter in model.parameters():
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            initial = {name: value.clone() for name, value in model.named_buffers()}
            tokens = torch.randint(0, 128, (1, 12), generator=generator)

            def decode(with_device_position):
                model(tokens[:, :5], start_pos=0)
                logits, ids = [], []
                for position in range(5, 12):
                    cursor = (
                        torch.tensor(position, dtype=torch.int64)
                        if with_device_position
                        else None
                    )
                    output, step_logits, _ = model(
                        tokens[:, position : position + 1],
                        start_pos=position,
                        position=cursor,
                    )
                    logits.append(step_logits)
                    ids.append(output)
                return logits, ids

            eager_logits, eager_ids = decode(False)
            eager_buffers = {
                name: value.clone() for name, value in model.named_buffers()
            }
            for name, value in model.named_buffers():
                value.copy_(initial[name])
            device_logits, device_ids = decode(True)
            for index in range(len(eager_logits)):
                self.assertTrue(
                    torch.equal(eager_logits[index], device_logits[index]),
                    f"decode step {index} logits differ",
                )
                self.assertTrue(torch.equal(eager_ids[index], device_ids[index]))
            for name, value in model.named_buffers():
                self.assertTrue(
                    torch.equal(eager_buffers[name], value),
                    f"{name} differs after the device position path",
                )
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_varlen_target_calls_keep_per_request_logits_and_recurrent_state(self):
        previous_device, previous_dtype = (
            torch.get_default_device(),
            torch.get_default_dtype(),
        )
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=3,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0, 2, 1),
                kv_source_layers=(1, 2),
                index_source_layers=(1, 2),
                index_n_heads=4,
                index_head_dim=32,
                index_topk=4,
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=2,
                max_seq_len=32,
            )
            transformer = Transformer(args)
            generator = torch.Generator(device="cpu").manual_seed(0)
            for parameter in transformer.parameters():
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            starts, slots = (9, 15), torch.tensor([1, 0])
            input_generator = torch.Generator(device="cpu").manual_seed(0)
            for slot, start in zip(slots.tolist(), starts, strict=True):
                transformer(
                    torch.randint(0, 128, (1, start), generator=input_generator),
                    cache_slots=torch.tensor([slot]),
                )
            initial = {
                name: value.clone() for name, value in transformer.named_buffers()
            }
            for length in (1, 6):
                tokens = torch.randint(0, 128, (2, length), generator=input_generator)
                for name, value in transformer.named_buffers():
                    value.copy_(initial[name])
                outputs = []
                for index, start in enumerate(starts):
                    outputs.append(
                        transformer(
                            tokens[index : index + 1],
                            start_pos=start,
                            cache_slots=slots[index : index + 1],
                            full_logits=length > 1,
                            sample_output=False,
                            collect_main_hidden=False,
                        )[1]
                    )
                expected = {
                    name: value.clone() for name, value in transformer.named_buffers()
                }
                for name, value in transformer.named_buffers():
                    value.copy_(initial[name])
                plan = object()
                valid_rows = torch.ones(tokens.numel(), dtype=torch.bool)
                calls = []

                def dispatch(*, layer, rows, routing, valid_rows, plan, calls=calls):
                    calls.append((layer.layer_id, valid_rows, plan))
                    return layer._forward_rows(rows, routing=routing)

                actual = transformer(
                    tokens,
                    start_pos=max(starts),
                    position=torch.tensor(starts).view(2, 1),
                    cache_slots=slots,
                    cache_slots_cpu=slots.tolist(),
                    cache_positions_cpu=starts,
                    full_logits=length > 1,
                    sample_output=False,
                    collect_main_hidden=False,
                    logical_rows=1,
                    execution_plan=plan,
                    valid_rows=valid_rows,
                    dispatch_combine=dispatch,
                )[1]
                self.assertEqual([call[0] for call in calls], [0, 1, 2])
                self.assertTrue(
                    all(call[1] is valid_rows and call[2] is plan for call in calls)
                )
                with self.assertRaisesRegex(ValueError, "one device boolean"):
                    transformer(tokens, valid_rows=valid_rows[:-1])
                torch.testing.assert_close(actual, torch.cat(outputs), rtol=0, atol=0)
                for name, value in transformer.named_buffers():
                    torch.testing.assert_close(
                        value, expected[name], rtol=0, atol=0, msg=name
                    )

                from oh_my_vllm.models.deepseek_v41.batch import target_decode_batch

                uniform_starts = (max(starts),) * len(starts)
                for name, value in transformer.named_buffers():
                    value.copy_(initial[name])
                outputs = [
                    transformer(
                        tokens[index : index + 1],
                        start_pos=start,
                        cache_slots=slots[index : index + 1],
                        full_logits=length > 1,
                        sample_output=False,
                        collect_main_hidden=False,
                    )[1]
                    for index, start in enumerate(uniform_starts)
                ]
                expected = {
                    name: value.clone() for name, value in transformer.named_buffers()
                }
                for name, value in transformer.named_buffers():
                    value.copy_(initial[name])
                from tests.test_deepseek_speculation import _worker

                worker = _worker()
                worker.model = transformer
                worker.args = args
                worker.decode_cycle = 2
                worker.decode_graph_limit = 0
                result = target_decode_batch(
                    worker, tokens.tolist(), uniform_starts, slots.tolist()
                )
                torch.testing.assert_close(
                    result.logits, torch.cat(outputs), rtol=0, atol=0
                )
                for name, value in transformer.named_buffers():
                    torch.testing.assert_close(
                        value, expected[name], rtol=0, atol=0, msg=name
                    )
                self.assertEqual(shared_attn.index_k.size(1), max(starts) + length)
                self.assertEqual(shared_attn.compress_kv.size(1), max(starts) + length)
                for binding in worker.decode_bindings.values():
                    binding.close()
        finally:
            shared_attn.reset_query_state()
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_engram_hash_reads_a_device_position(self):
        state = NgramHashState.__new__(NgramHashState)
        torch.nn.Module.__init__(state)
        state.layout = SimpleNamespace(max_ngram_size=2)
        state.pad_id = 0
        state.cache = torch.zeros(2, 16, dtype=torch.int64)
        state.cache[0, :6] = torch.tensor([3, 4, 5, 6, 7, 1])
        state.token_map = torch.arange(8)
        state.multipliers = torch.tensor([[1, 3]])
        state.primes = torch.tensor([[[11]]])
        state.offsets = torch.tensor([[0]])
        tokens = torch.tensor([[2]])
        slots = torch.tensor([0])
        initial = state.cache.clone()

        eager = state(tokens, start_pos=6, cache_slots=slots)
        eager_cache = state.cache.clone()
        state.cache.copy_(initial)
        device = state(
            tokens,
            start_pos=6,
            cache_slots=slots,
            position=torch.tensor(6, dtype=torch.int64),
        )
        self.assertTrue(torch.equal(eager, device))
        self.assertTrue(torch.equal(eager_cache, state.cache))

    def test_padded_context_widths_never_expose_padding_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=3,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0, 2, 1),
                kv_source_layers=(1, 2),
                index_source_layers=(1, 2),
                index_n_heads=4,
                index_head_dim=32,
                index_topk=16,
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=1,
                max_seq_len=32,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            model(torch.randint(0, 128, (1, 5)), start_pos=0)
            widths = {}
            for position in range(5, 13):
                model(
                    torch.randint(0, 128, (1, 1)),
                    start_pos=position,
                    position=torch.tensor(position, dtype=torch.int64),
                )
                # The last index source (layer 2, ratio 1) owns the published state.
                idxs = shared_attn.topk_idxs
                width = shared_attn.index_k.size(1)
                compress_len = position + 1
                self.assertGreaterEqual(width, compress_len)
                self.assertEqual(
                    width,
                    compressed_context_width(compress_len, 1, args.max_seq_len),
                )
                compressed = idxs[idxs >= args.window_size]
                self.assertTrue(
                    bool((compressed < args.window_size + compress_len).all()),
                    "an emitted index selected a padding row",
                )
                extent = context_token_bucket(compress_len)
                widths.setdefault(extent, set()).add((idxs.size(-1), width))
            for extent, shapes in widths.items():
                self.assertEqual(
                    len(shapes), 1, f"bucket {extent} changed shape: {shapes}"
                )
            self.assertEqual(compressed_context_width(13, 1, args.max_seq_len), 16)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_decode_bindings_write_fixed_device_inputs(self):
        bindings = DecodeBindings(2, [3, 1], "cpu")
        bindings.bind(torch.tensor([7, 9]), 12)
        self.assertTrue(torch.equal(bindings.token_ids, torch.tensor([[7], [9]])))
        self.assertEqual(int(bindings.position), 12)
        self.assertTrue(torch.equal(bindings.cache_slots, torch.tensor([3, 1])))
        bindings.bind(torch.tensor([4, 5]), 13)
        self.assertTrue(torch.equal(bindings.token_ids, torch.tensor([[4], [5]])))
        self.assertEqual(int(bindings.position), 13)
        with self.assertRaises(ValueError):
            DecodeBindings(2, [1, 1], "cpu")
        with self.assertRaises(ValueError):
            DecodeBindings(2, [1], "cpu")
        single = DecodeBindings(1, [0], "cpu")
        single.bind(6, 4)
        self.assertTrue(torch.equal(single.token_ids, torch.tensor([[6]])))
        self.assertEqual(sampled_token_ids(torch.tensor([[1.0, 3.0, 2.0]])), [1])

    def test_decode_graph_validates_every_request_before_capture_and_replay(self):
        from oh_my_vllm.models.deepseek_v41.decode import DecodeGraph

        bindings = DecodeBindings(2, (3, 1), "cpu", length=6, varlen=True)
        graph = DecodeGraph.__new__(DecodeGraph)
        graph.bindings = bindings
        graph.start_pos = 14
        graph.context_bucket = 32
        graph.cycle = 2
        graph.residue = 0
        graph.window_prefixes = ((8, 8),)
        graph.routed_bindings = ()
        graph.closed = False
        graph.cache_lease = None
        graph.output = None
        graph.model = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "bind"):
            graph._validate_positions()
        for positions in ((12, 14), (14, 18), (18, 26)):
            bindings.bind([[1] * 6] * 2, positions)
            graph._validate_positions()
        for positions in ((8, 14), (13, 14), (6, 14), (14, 28)):
            bindings.bind([[1] * 6] * 2, positions)
            with (
                self.subTest(positions=positions),
                self.assertRaisesRegex(ValueError, "outside"),
            ):
                graph.replay()
        bindings.bind([[1] * 6] * 2, (8, 14))
        with (
            patch.object(graph, "close") as close,
            self.assertRaisesRegex(ValueError, "outside"),
        ):
            graph.capture()
        close.assert_called_once_with()
        graph.context_bucket = 16
        graph.start_pos = 6
        graph.window_prefixes = ((8, 6),)
        bindings.bind([[1] * 6] * 2, (6, 6))
        graph._validate_positions()
        bindings.bind([[1] * 6] * 2, (4, 6))
        with self.assertRaisesRegex(ValueError, "outside"):
            graph.replay()
        bindings.close()

    def test_capture_pauses_host_state_export(self):
        export = SimpleNamespace(steps={0: "step"}, batch=(0,))
        model = SimpleNamespace(_state_export=export)
        with pause_state_export(model):
            self.assertEqual(export.steps, {})
        self.assertEqual(export.steps, {0: "step"})
        self.assertEqual(export.batch, (0,))
        plain = SimpleNamespace()
        with pause_state_export(plain):
            pass
        self.assertFalse(hasattr(plain, "_state_export"))

    def test_captured_forward_defers_the_sparse_status_check(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=64,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=2,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            ids = torch.randint(0, 64, (1, 2))
            with self.assertRaises(ValueError):
                model(
                    ids,
                    start_pos=0,
                    position=torch.tensor([[0, 1]], dtype=torch.int64),
                )
            with patch(
                "oh_my_vllm.models.deepseek_v41.model.require_sparse_status"
            ) as status:
                model(ids, start_pos=0, check_status=False)
                status.assert_not_called()
                model(ids, start_pos=0)
                status.assert_called_once()
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_engram_accepts_a_staged_embedding(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layout = SimpleNamespace(
                max_ngram_size=2,
                n_heads=1,
                head_dim=32,
                layer_ids=(0,),
                num_embeddings=(8,),
            )
            args = ModelArgs(dim=8, hc_mult=1, norm_eps=1e-20)
            engram = Engram(args, 0, layout)
            for parameter in engram.parameters():
                if parameter.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                    parameter.data.copy_(
                        torch.full(parameter.shape, 0.02, dtype=torch.float32).to(
                            parameter.dtype
                        )
                    )
                else:
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            x = torch.randn(1, 1, 1, 8, dtype=torch.bfloat16)
            hash_ids = torch.randint(0, 8, (1, 1, 1))
            eager = engram(x, hash_ids)
            staged = engram.embed(hash_ids).flatten(-2)
            mixed = engram(x, None, embedding=staged)
            self.assertTrue(torch.equal(eager, mixed))
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_compressed_context_width_covers_each_bucket(self):
        for ratio in (1, 2):
            capacity = 512
            widths = {}
            for end_pos in range(1, capacity * ratio + 1):
                width = compressed_context_width(end_pos, ratio, capacity)
                self.assertGreaterEqual(width, end_pos // ratio)
                self.assertLessEqual(width, capacity)
                extent = context_token_bucket(end_pos)
                widths.setdefault(extent, set()).add(width)
            for extent, values in widths.items():
                self.assertEqual(
                    len(values), 1, f"bucket {extent} ratio {ratio}: {values}"
                )
            self.assertEqual(
                compressed_context_width(capacity * ratio, ratio, capacity), capacity
            )
            self.assertEqual(
                compressed_context_width(capacity * ratio + 1, ratio, capacity),
                capacity,
            )

    def test_prefill_window_indices_include_previous_cache_and_causal_chunk(self):
        indices = get_prefill_window_topk_idxs(4, 1, 3, 4)
        self.assertEqual(tuple(indices.shape), (1, 3, 4))
        self.assertTrue(torch.equal(indices[0, 0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(indices[0, 1], torch.tensor([2, 3, 4, 5])))
        self.assertTrue(torch.equal(indices[0, 2], torch.tensor([3, 4, 5, 6])))

        indices = get_prefill_window_topk_idxs(8, 2, 6, 4)
        expected = torch.tensor(
            [
                [-1, -1, -1, 0, 1, 2, 3, 4],
                [-1, -1, 0, 1, 2, 3, 4, 5],
                [-1, 0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(indices, expected.expand(2, -1, -1)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_table_stays_on_host_for_cuda_lookup(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            table = ParallelEngramEmbedding(8, 32)
            table.weight.data.copy_(
                torch.arange(8 * 32, device="cpu", dtype=torch.float32)
                .reshape(8, 32)
                .to(torch.float8_e4m3fn)
            )
            table.scale.data.fill_(1)

            result = table(torch.tensor([[1, 3]], device="cuda"))
            original = result.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                repeated = table(
                    torch.tensor([[3, 1]], dtype=torch.int32, device="cuda")
                )
            torch.cuda.current_stream().wait_stream(stream)
            self.assertTrue(torch.equal(repeated, original.flip(1)))
            self.assertTrue(torch.equal(result, original))

            self.assertEqual(table.weight.device.type, "cpu")
            self.assertFalse(table.weight.is_pinned())
            self.assertEqual(result.device.type, "cuda")
            expected = (
                torch.arange(32, 64, device="cpu", dtype=torch.float32)
                .to(torch.float8_e4m3fn)
                .float()
                .to("cuda")
            )
            self.assertTrue(torch.equal(result[0, 0].float(), expected))
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def _engram_staging_table(self):
        table = ParallelEngramEmbedding(8, 32)
        values = torch.arange(8 * 32, dtype=torch.float32, device="cpu")
        table.weight.data.copy_((values.reshape(8, 32) / 16).to(table.weight.dtype))
        table.scale.data.copy_(
            torch.tensor([125, 126, 127, 128, 129, 127, 126, 125], dtype=torch.uint8)
            .reshape(8, 1)
            .view(torch.float8_e8m0fnu)
        )
        return table

    def _engram_lookup_reference(self, table, ids):
        ids = ids.cpu().long()
        mask = (ids < table.vocab_start_idx) | (ids >= table.vocab_end_idx)
        local = (ids - table.vocab_start_idx).masked_fill(mask, 0)
        values = torch.nn.functional.embedding(local, table.weight)
        scales = torch.nn.functional.embedding(local, table.scale)
        values = (
            (values.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1))
            .flatten(-2)
            .to(torch.bfloat16)
        )
        return values.masked_fill(mask.unsqueeze(-1), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_staging_waits_before_third_chunk_and_keeps_tail(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        table = self._engram_staging_table()
        ids = (torch.arange(2 * 65536 + 7, device="cuda") % 10) - 1
        native = kernel._transform_module()
        events = []
        stages = []
        submissions = []
        event_type = torch.cuda.Event

        class TrackedEvent:
            def __init__(self):
                self.event = event_type()
                self.ready = False
                self.records = 0
                events.append(self)

            def synchronize(self):
                self.event.synchronize()
                self.ready = True

            def record(self, stream):
                self_index = events.index(self)
                submissions.append(("record", self_index))
                self.event.record(stream)
                self.records += 1
                self.ready = False

        def stage(weight, scales, indices, packed, start):
            event = events[len(stages) % 2]
            self.assertTrue(event.ready)
            if event.records:
                self.assertTrue(event.event.query())
            stages.append((packed.data_ptr(), indices.numel()))
            event.ready = False
            native.engram_lookup_stage(weight, scales, indices, packed, start)

        def decode(packed, output):
            native.engram_lookup_decode(packed, output)
            submissions.append(("decode", (len(stages) - 1) % 2))

        module = SimpleNamespace(
            engram_lookup_stage=stage,
            engram_lookup_decode=decode,
        )
        with (
            patch.object(torch.cuda, "Event", TrackedEvent),
            patch.object(kernel, "_transform_module", return_value=module),
        ):
            actual = table(ids)
        expected = self._engram_lookup_reference(table, ids)
        self.assertTrue(
            torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        )
        self.assertEqual([length for _, length in stages], [65536, 65536, 7])
        self.assertEqual(stages[0][0], stages[2][0])
        self.assertNotEqual(stages[0][0], stages[1][0])
        self.assertEqual([event.records for event in events], [2, 1])
        self.assertEqual(
            submissions,
            [
                ("decode", 0),
                ("record", 0),
                ("decode", 1),
                ("record", 1),
                ("decode", 0),
                ("record", 0),
            ],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_staging_recovers_after_copy_or_decode_failure(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        native = kernel._transform_module()
        copy = torch.Tensor.copy_
        synchronize = torch.cuda.Stream.synchronize
        for failure in ("copy", "decode"):
            with self.subTest(failure=failure):
                table = self._engram_staging_table()
                ids = torch.tensor([1, 7, 3, -1], device="cuda")
                table(ids)
                synchronized = []

                def sync(stream, calls=synchronized):
                    synchronize(stream)
                    calls.append(stream.cuda_stream)

                def copy_then_fail(destination, source, *args, **kwargs):
                    result = copy(destination, source, *args, **kwargs)
                    if destination.is_cuda and destination.dtype == torch.uint8:
                        raise RuntimeError("injected Engram copy failure")
                    return result

                def decode_then_fail(packed, output):
                    native.engram_lookup_decode(packed, output)
                    raise RuntimeError("injected Engram decode failure")

                module = SimpleNamespace(
                    engram_lookup_stage=native.engram_lookup_stage,
                    engram_lookup_decode=decode_then_fail,
                )
                fault = (
                    patch.object(torch.Tensor, "copy_", copy_then_fail)
                    if failure == "copy"
                    else patch.object(kernel, "_transform_module", return_value=module)
                )
                fault_stream = torch.cuda.Stream()
                fault_stream.wait_stream(torch.cuda.current_stream())
                with (
                    torch.cuda.stream(fault_stream),
                    patch.object(torch.cuda.Stream, "synchronize", sync),
                    fault,
                    self.assertRaisesRegex(RuntimeError, "injected Engram"),
                ):
                    table(ids)
                self.assertEqual(synchronized, [fault_stream.cuda_stream])
                updated = ids.flip(0)
                actual = table(updated)
                expected = self._engram_lookup_reference(table, updated)
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int16), expected.view(torch.int16)
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_stage_refreshes_graph_replay_inputs(self):
        table = self._engram_staging_table()
        engram = SimpleNamespace(
            q_weight=torch.empty(1, device="cuda"),
            n_hash_cols=1,
            head_dim=32,
            layer_hash_index=0,
            embed=table,
        )
        staged_model = SimpleNamespace(
            layers=[SimpleNamespace(engram=engram)],
            engram_hash=lambda tokens, *args, **kwargs: tokens[:, :, None, None],
        )
        bindings = SimpleNamespace(
            batch=1,
            token_ids=torch.tensor([[1]], device="cuda"),
            cache_slots=None,
            position=None,
        )
        stage = EngramStage(staged_model, 1)
        stage.stage(bindings, 0)
        address = stage.buffers[0].data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = stage.buffers[0].clone()
        torch.cuda.current_stream().wait_stream(stream)
        for token in (1, 6, 3):
            bindings.token_ids.fill_(token)
            stage.stage(bindings, token)
            graph.replay()
            expected = self._engram_lookup_reference(table, bindings.token_ids)
            self.assertEqual(stage.buffers[0].data_ptr(), address)
            self.assertTrue(
                torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            )

    def test_dspark_draft_attention_keeps_the_full_proposal_block_visible(self):
        indices = get_dspark_topk_idxs(8, 2, 5, 4)

        expected = torch.tensor([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=torch.int32)
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(indices.shape, (2, 5, 10))
        self.assertTrue(torch.equal(indices, expected.expand(2, 5, 10)))
        self.assertTrue(indices.is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_tiny_cuda_fp8_projection_path_returns_finite_logits(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=8,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=4,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="fp8",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            model = Transformer(args).eval()
            for parameter in model.parameters():
                if parameter.dtype.is_floating_point:
                    values = torch.randn(
                        parameter.shape, device=parameter.device, dtype=torch.float32
                    )
                    parameter.data.copy_((values * 0.02).to(parameter.dtype))
            with torch.inference_mode():
                _, logits, _ = model(torch.randint(0, 128, (1, 4)))
            self.assertTrue(torch.isfinite(logits).all())
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_cache_slots_do_not_cross_contaminate_decode(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=8,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=4,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=2,
                max_seq_len=8,
            )
            baseline = Transformer(args)
            mixed = Transformer(args)
            mixed.load_state_dict(baseline.state_dict())
            for parameter in baseline.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            mixed.load_state_dict(baseline.state_dict())
            first = torch.randint(0, 128, (1, 4))
            second = torch.randint(0, 128, (1, 4))
            decode = torch.randint(0, 128, (1, 1))
            baseline(first, cache_slots=torch.tensor([0]))
            expected = baseline(
                decode,
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )[1]
            mixed(first, cache_slots=torch.tensor([0]))
            mixed(second, cache_slots=torch.tensor([1]))
            actual = mixed(
                decode,
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )[1]
            self.assertTrue(torch.allclose(expected, actual, atol=0, rtol=0))
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_nonzero_prefill_chunk_matches_sequential_decode(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=8,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=4,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=1,
                max_seq_len=8,
            )
            sequential = Transformer(args)
            chunked = Transformer(args)
            for parameter in sequential.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            chunked.load_state_dict(sequential.state_dict())
            prompt = torch.tensor([[1, 2, 3, 4]])
            continuation = torch.tensor([[5, 6]])
            sequential(prompt, cache_slots=torch.tensor([0]))
            sequential_logits = [
                sequential(
                    continuation[:, offset : offset + 1],
                    start_pos=4 + offset,
                    cache_slots=torch.tensor([0]),
                    full_logits=True,
                )[1]
                for offset in range(continuation.size(1))
            ]
            chunked(prompt, cache_slots=torch.tensor([0]))
            chunked_logits = chunked(
                continuation,
                start_pos=4,
                cache_slots=torch.tensor([0]),
                full_logits=True,
            )[1]
            torch.testing.assert_close(
                chunked_logits,
                torch.cat(sequential_logits, dim=1),
                atol=1e-5,
                rtol=1e-3,
            )
            selected = Transformer(args)
            selected.load_state_dict(sequential.state_dict())
            selected(prompt, cache_slots=torch.tensor([0]))
            selected_logits = selected(
                continuation,
                start_pos=4,
                cache_slots=torch.tensor([0]),
                full_logits=True,
                logit_start=1,
            )[1]
            torch.testing.assert_close(
                selected_logits,
                chunked_logits[:, 1:],
                atol=1e-5,
                rtol=1e-3,
            )
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_nonzero_prefill_chunk_matches_sequential_compressed_attention(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(2, 2),
                kv_source_layers=(0,),
                index_source_layers=(0, 1),
                index_n_heads=4,
                index_head_dim=32,
                index_topk=4,
                candidate_source_layer=0,
                candidate_topk_blocks=1,
                candidate_block_size=2,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=2,
                max_seq_len=32,
                dspark_target_layer_ids=(0, 1),
            )
            for prefix_length, chunk_length in ((3, 5), (9, 13)):
                with self.subTest(prefix=prefix_length, chunk=chunk_length):
                    torch.manual_seed(71)
                    sequential = Transformer(args).float()
                    chunked = Transformer(args).float()
                    for parameter in sequential.parameters():
                        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
                    chunked.load_state_dict(sequential.state_dict())
                    slots = torch.tensor([1])
                    prompt = torch.arange(1, prefix_length + 1).unsqueeze(0)
                    continuation = torch.arange(
                        prefix_length + 1, prefix_length + chunk_length + 1
                    ).unsqueeze(0)
                    sequential(prompt, cache_slots=slots)
                    expected = [
                        sequential(
                            continuation[:, offset : offset + 1],
                            start_pos=prefix_length + offset,
                            cache_slots=slots,
                            full_logits=True,
                        )
                        for offset in range(chunk_length)
                    ]
                    chunked(prompt, cache_slots=slots)
                    with patch(
                        "oh_my_vllm.models.deepseek_v41.model.INDEX_SCORE_WORK_BYTES",
                        1024,
                    ):
                        actual = chunked(
                            continuation,
                            start_pos=prefix_length,
                            cache_slots=slots,
                            full_logits=True,
                        )
                    torch.testing.assert_close(
                        actual[1],
                        torch.cat([item[1] for item in expected], dim=1),
                        atol=2e-4,
                        rtol=2e-3,
                    )
                    torch.testing.assert_close(
                        actual[2],
                        torch.cat([item[2] for item in expected], dim=1),
                        atol=2e-4,
                        rtol=2e-3,
                    )
                    for left, right in zip(
                        chunked.layers, sequential.layers, strict=True
                    ):
                        torch.testing.assert_close(
                            left.attn.window_kv_cache,
                            right.attn.window_kv_cache,
                            atol=3e-4,
                            rtol=3e-3,
                        )
                    compressed = (prefix_length + chunk_length) // 2
                    torch.testing.assert_close(
                        chunked.layers[0].attn.compress_kv_cache[1, :compressed],
                        sequential.layers[0].attn.compress_kv_cache[1, :compressed],
                        atol=3e-4,
                        rtol=3e-3,
                    )
                    torch.testing.assert_close(
                        chunked.layers[0].attn.indexer.k_cache[1, :compressed],
                        sequential.layers[0].attn.indexer.k_cache[1, :compressed],
                        atol=3e-4,
                        rtol=3e-3,
                    )
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_dspark_seed_stores_target_history_without_running_draft_queries(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=1,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                dspark_n_routed_experts=4,
                dspark_n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=8,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=4,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                dspark_block_size=5,
                dspark_noise_token_id=0,
                dspark_target_layer_ids=(0,),
                dspark_markov_rank=8,
                max_batch_size=1,
                max_seq_len=16,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            slots = torch.tensor([0])
            _, _, hidden = model(torch.randint(0, 128, (1, 4)), cache_slots=slots)
            model.forward_spec_seed(hidden, start_pos=0, cache_slots=slots)
            first = model.mtp[0].attn.window_kv_cache[0, :4].clone()
            self.assertTrue(torch.any(first != 0))

            _, _, hidden = model(
                torch.randint(0, 128, (1, 1)),
                start_pos=4,
                cache_slots=slots,
            )
            model.forward_spec_seed(hidden, start_pos=4, cache_slots=slots)
            self.assertTrue(torch.any(model.mtp[0].attn.window_kv_cache[0, 4] != 0))
            proposal = model.forward_spec(
                torch.tensor([[1]]),
                hidden,
                start_pos=4,
                cache_slots=slots,
            )
            self.assertIsNotNone(proposal)
            self.assertEqual(tuple(proposal[0].shape), (1, 6))
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_tiny_target_forward_returns_finite_logits(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=8,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=4,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            output_ids, logits, main_hidden = model(torch.randint(0, 128, (1, 8)))
            self.assertEqual(tuple(output_ids.shape), (1,))
            self.assertEqual(tuple(logits.shape), (1, 128))
            self.assertIsNone(main_hidden)
            self.assertTrue(torch.isfinite(logits).all())
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    unittest.main()
