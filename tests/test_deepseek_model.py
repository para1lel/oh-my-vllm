import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.engram import NgramHashState
from oh_my_vllm.models.deepseek_v41.kernel import act_quant, fp8_gemm
from oh_my_vllm.models.deepseek_v41.loader import RankLoader
from oh_my_vllm.models.deepseek_v41.model import (
    Attention,
    Compressor,
    Expert,
    Gate,
    ModelArgs,
    MoE,
    ParallelEngramEmbedding,
    RowParallelLinear,
    Transformer,
    get_dspark_topk_idxs,
    get_prefill_window_topk_idxs,
    select_candidate_blocks,
    shared_attn,
)
from safetensors.torch import save_file


class DeepSeekModel(unittest.TestCase):
    def test_shared_expert_forward_uses_full_weights_after_routed_reduction(self):
        args = ModelArgs(
            dim=2,
            moe_inter_dim=8,
            n_layers=1,
            n_routed_experts=4,
            n_activated_experts=1,
            dspark_n_routed_experts=8,
            expert_dtype=None,
        )
        values = torch.ones((1, 1, 2), dtype=torch.bfloat16)
        gate_weight = torch.tensor([[0.5, 0.5]] * 8, dtype=torch.bfloat16)
        up_weight = torch.tensor([[1.0, 0.0]] * 8, dtype=torch.bfloat16)
        down_weight = torch.tensor([[0.125] * 8] * 2, dtype=torch.bfloat16)
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
                        shared = moe.shared_experts
                        self.assertEqual(tuple(shared.w1.weight.shape), (8, 2))
                        self.assertEqual(tuple(shared.w2.weight.shape), (2, 8))
                        self.assertEqual(tuple(shared.w3.weight.shape), (8, 2))
                        with torch.no_grad():
                            for parameter in moe.parameters():
                                parameter.zero_()
                            shared.w1.weight.copy_(gate_weight)
                            shared.w2.weight.copy_(down_weight)
                            shared.w3.weight.copy_(up_weight)

                        reductions = []

                        def reduce_routed(
                            value, reductions=reductions, strategy=strategy
                        ):
                            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
                            reductions.append(value)
                            if strategy != "TP4" or len(reductions) == 2:
                                value.fill_(3)

                        with patch(
                            "oh_my_vllm.models.deepseek_v41.model.dist.all_reduce",
                            side_effect=reduce_routed,
                        ):
                            actual = moe(values)
                        self.assertEqual(len(reductions), 2 if strategy == "TP4" else 1)
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

    def test_expert_down_projection_rounds_after_tensor_parallel_reduction(self):
        with patch("oh_my_vllm.models.deepseek_v41.model.world_size", 4):
            expert = Expert(2, 8, dtype=torch.bfloat16, tensor_parallel=True)
            attention = RowParallelLinear(8, 2, dtype=torch.bfloat16)
            for layer, partial, result in (
                (expert.w2, 257 / 256, 1 / 256),
                (attention, 1, 0),
            ):
                with self.subTest(expert=layer is expert.w2):
                    with torch.no_grad():
                        layer.weight.copy_(torch.tensor([[1, 1 / 256]] * 2))

                    def reduce(value, partial=partial):
                        self.assertEqual(value.dtype, torch.float32)
                        torch.testing.assert_close(
                            value,
                            torch.full((1, 2), partial, dtype=torch.float32),
                            rtol=0,
                            atol=0,
                        )
                        value.sub_(1)

                    with patch(
                        "oh_my_vllm.models.deepseek_v41.model.dist.all_reduce",
                        side_effect=reduce,
                    ):
                        actual = layer(torch.ones((1, 2), dtype=torch.bfloat16))
                    torch.testing.assert_close(
                        actual,
                        torch.full((1, 2), result, dtype=torch.bfloat16),
                        rtol=0,
                        atol=0,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_lazy_cache_keeps_empty_context_storage_dense(self):
        from oh_my_vllm.models.deepseek_v41.storage import sequence_cache

        previous_device = torch.get_default_device()
        torch.set_default_device("cuda")
        try:
            value = sequence_cache(
                SimpleNamespace(lazy_cache=True), (2, 0, 32), dtype=torch.bfloat16
            )
            self.assertEqual(value.shape, (2, 0, 32))
            self.assertEqual(value.device.type, "cuda")
            self.assertFalse(hasattr(value, "_paged_storage"))
        finally:
            torch.set_default_device(previous_device)

    def test_mapped_models_reject_storage_conversion_before_access(self):
        model = Transformer.__new__(Transformer)
        torch.nn.Module.__init__(model)
        model._sequence_caches = (torch.empty(0),)
        with self.assertRaisesRegex(RuntimeError, "mapped.*conversion"):
            model.float()

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
            self.assertEqual(mapped_bytes() - before, active // 2)
            torch.testing.assert_close(attention.compress_kv_cache[2, :2], saved)
            cache.restore(7, 10, row, start_pos=4)
            torch.testing.assert_close(attention.compress_kv_cache[10, :2], restored)
            torch.testing.assert_close(attention.window_kv_cache[10], restored_window)
            self.assertEqual(mapped_bytes() - before, active)
            cache.clear_slot(10)
            cache.clear_slot(2)
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
        moe.experts = torch.nn.ModuleList(FixedExpert(index + 1) for index in range(2))
        moe.shared_experts = ZeroExpert()
        value = torch.tensor((1.0, 2.0)).expand(1025, 1, 2)

        result = moe(value)

        expected = torch.tensor((2.75, 3.75)).expand_as(result)
        torch.testing.assert_close(result, expected)
        self.assertGreater(gate.calls, 1)

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
