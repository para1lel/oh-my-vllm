import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.engram import NgramHashState
from oh_my_vllm.models.deepseek_v41.model import (
    Attention,
    Compressor,
    ModelArgs,
    Transformer,
    select_candidate_blocks,
)
from oh_my_vllm.models.deepseek_v41.model_context import shared_attn

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelCache(DeepSeekModelFixture, unittest.TestCase):
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
            args = self._lazy_cache_args()
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

    def _lazy_cache_args(self):
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
        return args
