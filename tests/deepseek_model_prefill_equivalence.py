import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    Transformer,
    get_dspark_topk_idxs,
)

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelPrefillEquivalence(DeepSeekModelFixture, unittest.TestCase):
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
            args = self._sequential_prefill_args()
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
            args = self._compressed_prefill_args()
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
                        "oh_my_vllm.models.deepseek_v41.model_indexer.INDEX_SCORE_WORK_BYTES",
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

    def _sequential_prefill_args(self):
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
        return args

    def _compressed_prefill_args(self):
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
        return args
