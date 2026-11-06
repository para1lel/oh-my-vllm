import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model_context
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    MoE,
    Transformer,
)

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelPrefill(DeepSeekModelFixture, unittest.TestCase):
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
                patch.object(model_context, name, value),
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
            with (
                self.subTest(strategy=strategy),
                patch.object(model_context, "world_size", 1),
            ):
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

        self._check_prefill_row_configuration(projection_rows)
        transformer = self._prefill_transformer()
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

    def _prefill_transformer(self):
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
        return transformer

    def _check_prefill_row_configuration(self, projection_rows):
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
