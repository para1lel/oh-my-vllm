import unittest

import torch
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    ParallelEngramEmbedding,
    Transformer,
    get_dspark_topk_idxs,
)


class DeepSeekModel(unittest.TestCase):
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
            sequential_logits = sequential(
                continuation[:, :1],
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )[1]
            sequential_logits = sequential(
                continuation[:, 1:],
                start_pos=5,
                cache_slots=torch.tensor([0]),
            )[1]
            chunked(prompt, cache_slots=torch.tensor([0]))
            chunked_logits = chunked(
                continuation,
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )[1]
            torch.testing.assert_close(chunked_logits, sequential_logits)
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
