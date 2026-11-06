import unittest

import torch
from oh_my_vllm.models.deepseek_v41.decode import DecodeBindings
from oh_my_vllm.models.deepseek_v41.decode_state import (
    GraphStateExports,
    redirect_state_exports,
)
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    Transformer,
)
from oh_my_vllm.models.deepseek_v41.state import StateExports

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelTarget(DeepSeekModelFixture, unittest.TestCase):
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

    def test_proposal_device_position_keeps_logits_confidence_and_state(self):
        with torch.device("cpu"):
            model = self._proposal_model()
        with torch.device("cpu"), torch.inference_mode():
            generator = torch.Generator(device="cpu").manual_seed(7)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.02, generator=generator)
            bindings = DecodeBindings(1, (0,), "cpu")
            hidden = torch.randn(1, 1, 64, generator=generator).bfloat16()
            inputs = torch.tensor([[3]])
            for position in (1, 6, 7, 8, 15, 16, 23):
                with self.subTest(position=position):
                    windows = [layer.attn.window_kv_cache for layer in model.mtp]
                    initial = [
                        torch.randn(value.shape, generator=generator).bfloat16()
                        for value in windows
                    ]
                    expected_export, actual_export = StateExports(), StateExports()
                    for export in (expected_export, actual_export):
                        export.begin(0, position, (position + 1,))
                    model._state_export = expected_export
                    for layer, value in zip(model.mtp, initial, strict=True):
                        layer.attn._state_export = expected_export
                        layer.attn.window_kv_cache.copy_(value)
                    expected = model.forward_spec(inputs, hidden, start_pos=position)
                    expected_windows = [value.clone() for value in windows]
                    for value, original in zip(windows, initial, strict=True):
                        value.copy_(original)
                    bindings.bind(inputs, position)
                    branch = min(position, 7)
                    captured = GraphStateExports(bindings, branch)
                    with redirect_state_exports(model, captured):
                        actual = model.forward_spec(
                            inputs,
                            hidden,
                            start_pos=branch,
                            cache_slots=bindings.cache_slots,
                            cache_slots_cpu=bindings.cache_slots_cpu,
                            position=bindings.position,
                        )
                    captured.publish(actual_export, preserve_existing=True)
                    for want, got in zip(expected, actual, strict=True):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    for value, want in zip(windows, expected_windows, strict=True):
                        torch.testing.assert_close(value, want, rtol=0, atol=0)
                        torch.testing.assert_close(
                            actual_export.window_state(0, position + 1, value),
                            expected_export.window_state(0, position + 1, value),
                            rtol=0,
                            atol=0,
                        )
            bindings.close()

    @staticmethod
    def _proposal_model():
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            return Transformer(
                ModelArgs(
                    vocab_size=128,
                    dim=64,
                    moe_inter_dim=64,
                    n_layers=1,
                    n_mtp_layers=3,
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
                    compress_ratios=(0, 0, 0, 0),
                    kv_source_layers=(),
                    index_source_layers=(),
                    candidate_source_layer=-1,
                    hc_mult=2,
                    dtype="bf16",
                    expert_dtype=None,
                    dspark_block_size=5,
                    dspark_noise_token_id=0,
                    dspark_target_layer_ids=(0,),
                    dspark_markov_rank=8,
                    max_batch_size=1,
                    max_seq_len=32,
                    temperature=0,
                )
            )
        finally:
            torch.set_default_dtype(previous)

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
