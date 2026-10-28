import unittest

import torch
from oh_my_vllm.models.deepseek_v41.kernel import act_quant, fp8_gemm
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    MoE,
    ParallelEngramEmbedding,
    Transformer,
    get_dspark_topk_idxs,
    get_prefill_window_topk_idxs,
    shared_attn,
)


class DeepSeekModel(unittest.TestCase):
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_moe_grouped_fp4_path_matches_expert_reference(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
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

    def test_dspark_draft_attention_masks_future_draft_positions(self):
        indices = get_dspark_topk_idxs(8, 1, 5, 4)

        self.assertEqual(indices.shape, (1, 5, 10))
        self.assertTrue(
            torch.equal(indices[0, 0, 5:], torch.tensor([8, -1, -1, -1, -1]))
        )
        self.assertTrue(
            torch.equal(indices[0, 4, 5:], torch.tensor([8, 9, 10, 11, 12]))
        )

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
