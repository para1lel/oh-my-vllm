import unittest
from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.packed_cache import PackedHistory, TokenHistory


class FakeModel:
    def __init__(self, *, batch_size=2, context=8):
        args = SimpleNamespace(
            max_batch_size=batch_size, max_seq_len=context, lazy_cache=False
        )
        self.layers = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    is_kv_source=True,
                    window_kv_cache=torch.zeros(batch_size, 4, 3),
                    compress_kv_cache=PackedHistory(
                        args, 32, 2, 16, torch.float8_e4m3fn
                    ),
                    compress_ratio=2,
                    indexer=SimpleNamespace(
                        owns_k=True,
                        k_cache=PackedHistory(args, 32, 2, 32, torch.float8_e8m0fnu),
                    ),
                    compressor=SimpleNamespace(
                        kv_state=torch.zeros(batch_size, 2, 3),
                        score_state=torch.zeros(batch_size, 2, 3),
                    ),
                )
            )
        ]
        self.mtp = [
            SimpleNamespace(
                attn=SimpleNamespace(window_kv_cache=torch.zeros(batch_size, 4, 3))
            )
        ]
        self.engram_hash = SimpleNamespace(cache=TokenHistory(args))


def cache_for(model, **options):
    return DeepSeekCache(model, fa_capacity=128, **options)


def row(fa, mamba):
    return {
        "request_id": 7,
        "fa_block_table": fa,
        "mamba_block_table": mamba,
    }


class DeepSeekCacheBehavior(unittest.TestCase):
    def test_real_model_restores_sources_with_and_without_compression_tails(self):
        from oh_my_vllm.models.deepseek_v41.model import ModelArgs, Transformer

        with torch.device("cpu"), torch.inference_mode():
            model = Transformer(
                ModelArgs(
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
                    compress_ratios=(2, 1),
                    kv_source_layers=(0, 1),
                    index_source_layers=(0, 1),
                    index_n_heads=4,
                    index_head_dim=32,
                    index_topk=4,
                    candidate_source_layer=-1,
                    hc_mult=2,
                    dtype="bf16",
                    expert_dtype=None,
                    temperature=0,
                    max_batch_size=2,
                    max_seq_len=8,
                    lazy_cache=False,
                )
            )
            cache = cache_for(model, block_tokens=4, max_seq_len=8)
            self.addCleanup(cache.close)
            source = row([1], [3])
            cache.activate_row(source, 0, end_pos=4)
            for index, layer in enumerate(model.layers):
                layer.attn.window_kv_cache[0].fill_(index + 2)
            compressor = model.layers[0].attn.compressor
            compressor.kv_state[0].fill_(5)
            compressor.score_state[0].fill_(7)
            cache.commit(7, source, 0, accepted_end=4)
            cache.clear_slot(0)
            cache.restore(7, 1, source, start_pos=4)
            for index, layer in enumerate(model.layers):
                window = layer.attn.window_kv_cache[1]
                torch.testing.assert_close(window, torch.full_like(window, index + 2))
            torch.testing.assert_close(
                compressor.kv_state[1], torch.full_like(compressor.kv_state[1], 5)
            )
            torch.testing.assert_close(
                compressor.score_state[1], torch.full_like(compressor.score_state[1], 7)
            )

    def test_verification_commits_exact_windows_tails_and_private_checkpoints(self):
        windows = (
            (1, 2, 3, 0),
            (1, 2, 3, 4),
            (5, 2, 3, 4),
            (5, 6, 3, 4),
            (5, 6, 7, 4),
            (5, 6, 7, 8),
        )
        for accepted, want in enumerate(windows):
            with self.subTest(accepted_drafts=accepted):
                model = FakeModel()
                cache = cache_for(model, block_tokens=4, max_seq_len=8)
                cache.activate_row(row([1, 2], [3, 4]), 0, end_pos=8)
                attention = model.layers[0].attn
                mtp = model.mtp[0].attn.window_kv_cache
                attention.window_kv_cache[0, :2] = torch.tensor([[1.0], [2.0]])
                mtp[0, :2] = torch.tensor([[101.0], [102.0]])
                model.engram_hash.cache[0, :2] = torch.tensor([20, 21])
                cache.commit(7, row([1], [3]), 0, accepted_end=2, include_mtp=True)
                prefix_window = cache.mamba_pages[3].window[0].clone()
                cache.copy_prefix([(1, 7, 2)])
                current = {
                    **row([7, 2], [5, 6]),
                    "prefix_state_slot": 3,
                    "cache_checkpoint": (3, 9),
                }
                cache.begin_step(current, 0, start_pos=2, end_pos=8, minimum_end=3)
                cache.exports.bind((0,))
                values = torch.arange(3, 9).float().view(1, 6, 1).expand(1, 6, 3)
                cache.exports.window(attention.window_kv_cache, values, 2)
                cache.exports.window(mtp, values + 100, 2)
                compressor = attention.compressor
                cache.exports.tail(
                    compressor.kv_state, compressor.score_state, values, values + 100, 2
                )
                for position in range(2, 8):
                    attention.window_kv_cache[0, position % 4] = position + 1
                    mtp[0, position % 4] = position + 101
                compressor.kv_state.fill_(999)
                compressor.score_state.fill_(999)
                model.engram_hash.cache[0, 2:8] = torch.arange(22, 28)
                endpoint = 3 + accepted
                cache.commit(
                    7, current, 0, start_pos=2, accepted_end=endpoint, include_mtp=True
                )
                cache.restore(7, 1, current, start_pos=endpoint, use_prefix_state=False)
                torch.testing.assert_close(
                    attention.window_kv_cache[1, :, 0], torch.tensor(want).float()
                )
                torch.testing.assert_close(
                    mtp[1, :, 0],
                    torch.tensor(
                        [value + 100 if value else 0 for value in want]
                    ).float(),
                )
                kv, score = compressor.kv_state[1], compressor.score_state[1]
                torch.testing.assert_close(
                    kv[0], torch.full((3,), float(endpoint if endpoint % 2 else 0))
                )
                torch.testing.assert_close(kv[1], torch.zeros(3))
                torch.testing.assert_close(
                    score[0],
                    torch.full(
                        (3,), float(endpoint + 100) if endpoint % 2 else -torch.inf
                    ),
                )
                torch.testing.assert_close(score[1], torch.full((3,), -torch.inf))
                torch.testing.assert_close(
                    cache.mamba_pages[3].window[0], prefix_window
                )
                self.assertEqual(cache.mamba_pages[3].endpoint, 2)
                self.assertEqual(cache.mamba_pages[9].endpoint, 3)
                torch.testing.assert_close(
                    cache.mamba_pages[9].window[0][:, 0], torch.tensor([1.0, 2, 3, 0])
                )

    def test_restore_requires_the_exact_state_endpoint_before_clearing(self):
        model = FakeModel()
        cache = cache_for(model, block_tokens=4, max_seq_len=8)
        source = row([1, 2], [3, 0])
        cache.activate_row(source, 0, end_pos=4)
        cache.commit(7, source, 0, accepted_end=4)
        for endpoint in (2, 3, 5):
            with self.subTest(endpoint=endpoint):
                model.layers[0].attn.window_kv_cache[0].fill_(71)
                with self.assertRaisesRegex(ValueError, "endpoint"):
                    cache.restore(7, 0, source, start_pos=endpoint)
                torch.testing.assert_close(
                    model.layers[0].attn.window_kv_cache[0], torch.full((4, 3), 71.0)
                )

    def test_commit_merges_prefix_state_into_a_new_mamba_page(self):
        model = FakeModel()
        cache = cache_for(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        source = row([1], [3])
        cache.activate_row(source, 0, end_pos=4)
        model.layers[0].attn.window_kv_cache[0] = 1
        model.engram_hash.cache[0, :2] = torch.tensor([104, 105])
        cache.commit(7, source, 0, accepted_end=2)
        cache.restore(
            7,
            0,
            {**row([1], [4]), "prefix_state_slot": 3},
            start_pos=2,
        )
        model.engram_hash.cache[0, 2:4] = torch.tensor([106, 107])
        cache.commit(
            7,
            {**row([1], [4]), "prefix_state_slot": 3},
            0,
            start_pos=2,
            accepted_end=4,
        )

        self.assertTrue(
            torch.equal(
                cache.mamba_pages[4].engram[0],
                torch.tensor([104, 105, 106, 107]),
            )
        )

    def test_restore_rejects_missing_accepted_mamba_state(self):
        model = FakeModel()
        cache = cache_for(model, block_tokens=4, max_seq_len=8)
        cache._fa_page(1)
        with self.assertRaisesRegex(ValueError, "state page"):
            cache.restore(7, 0, row([1], [0]), start_pos=4)
        with self.assertRaisesRegex(ValueError, "state page"):
            cache.restore(7, 0, row([1], [3]), start_pos=4)

    def test_decoded_export_snapshots_the_post_replay_state(self):
        model = FakeModel()
        cache = cache_for(model, block_tokens=4, max_seq_len=8)
        attention = model.layers[0].attn
        source = row([1, 2], [3, 4])
        cache.begin_step(source, 0, start_pos=4, end_pos=5, minimum_end=5)
        attention.window_kv_cache[0] = torch.tensor(
            [[4.0, 40.0, 44.0], [5.0, 50.0, 54.0], [6.0, 60.0, 64.0], [7.0, 70.0, 74.0]]
        )
        attention.compressor.kv_state[0, 0] = torch.tensor([9.0, 90.0, 99.0])
        attention.compressor.score_state[0, 0] = torch.tensor([1.0, 2.0, 3.0])

        cache.export_decoded_step((0,), 4, 5)

        torch.testing.assert_close(
            cache.exports.window_state(0, 5, attention.window_kv_cache),
            attention.window_kv_cache[0],
        )
        tail_kv, tail_score = cache.exports.tail_state(
            0, 5, attention.compressor.kv_state
        )
        torch.testing.assert_close(tail_kv[0], torch.tensor([9.0, 90.0, 99.0]))
        self.assertTrue(bool((tail_kv[1:] == 0).all()))
        torch.testing.assert_close(tail_score[0], torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(bool(torch.isneginf(tail_score[1:]).all()))

    def test_decoded_export_requires_one_token(self):
        model = FakeModel()
        cache = cache_for(model, block_tokens=4, max_seq_len=8)
        cache.begin_step(row([1, 2], [3, 4]), 0, start_pos=5, end_pos=7, minimum_end=7)
        with self.assertRaisesRegex(ValueError, "one token"):
            cache.export_decoded_step((0,), 5, 7)

    def test_host_transfers_require_a_live_resident_request(self):
        model = FakeModel()
        requests = {7: SimpleNamespace(cache_slot=0)}
        cache = cache_for(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
            requests=requests,
            free_slots=set(),
        )
        cache.commit(7, row([1], [3]), 0, accepted_end=4)
        operation = {
            "transfer_id": 1,
            "snapshot_id": 9,
            "kind": "store",
            "request_id": 7,
            "fa_blocks": [1],
            "mamba_blocks": [3],
        }
        cache.handle({"operations": [operation]})
        requests.pop(7)
        with self.assertRaisesRegex(ValueError, "unknown"):
            cache.handle(
                {
                    "operations": [
                        {
                            "transfer_id": 2,
                            "snapshot_id": 9,
                            "kind": "load",
                            "request_id": 7,
                            "fa_blocks": [2],
                            "mamba_blocks": [4],
                        }
                    ]
                }
            )

    def test_invalid_later_operation_has_no_batch_side_effect(self):
        model = FakeModel()
        cache = cache_for(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        cache.commit(7, row([1], [3]), 0, accepted_end=4)
        with self.assertRaisesRegex(ValueError, "unknown DeepSeek cache snapshot"):
            cache.handle(
                {
                    "operations": [
                        {
                            "transfer_id": 1,
                            "snapshot_id": 9,
                            "kind": "store",
                            "request_id": None,
                            "fa_blocks": [1],
                            "mamba_blocks": [3],
                        },
                        {
                            "transfer_id": 2,
                            "snapshot_id": 99,
                            "kind": "load",
                            "request_id": None,
                            "fa_blocks": [2],
                            "mamba_blocks": [4],
                        },
                    ]
                }
            )
        self.assertNotIn(9, cache.snapshots)
        self.assertEqual(cache.seen, set())

    def test_store_load_and_prefix_copy_restore_logical_blocks(self):
        model = FakeModel()
        cache = cache_for(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        source = row([1, 2], [3, 4])
        cache.activate_row(source, 0, end_pos=6)
        model.layers[0].attn.window_kv_cache[0] = torch.arange(12).reshape(4, 3)
        model.layers[0].attn.compressor.kv_state[0] = 11
        model.layers[0].attn.compressor.score_state[0] = 13
        model.mtp[0].attn.window_kv_cache[0] = 17
        model.engram_hash.cache[0, :6] = torch.arange(6) + 20
        cache.commit(7, source, 0, accepted_end=4, include_mtp=True)
        cache.commit(7, source, 0, start_pos=4, accepted_end=6, include_mtp=True)

        result = cache.handle(
            {
                "operations": [
                    {
                        "transfer_id": 1,
                        "snapshot_id": 9,
                        "kind": "store",
                        "request_id": 7,
                        "fa_blocks": [1, 2],
                        "mamba_blocks": [3, 4],
                    }
                ]
            }
        )
        self.assertEqual(result[0]["status"], "complete")
        self.assertGreater(result[0]["bytes"], 0)

        cache.clear_slot(0)
        result = cache.handle(
            {
                "operations": [
                    {
                        "transfer_id": 2,
                        "snapshot_id": 9,
                        "kind": "load",
                        "request_id": 7,
                        "fa_blocks": [5, 6],
                        "mamba_blocks": [7, 8],
                    }
                ]
            }
        )
        self.assertEqual(result[0]["status"], "complete")
        restored = row([5, 6], [7, 8])
        cache.restore(7, 0, restored, start_pos=6)
        torch.testing.assert_close(
            model.layers[0].attn.window_kv_cache[0, :4],
            torch.arange(12, dtype=torch.float32).reshape(4, 3),
        )
        torch.testing.assert_close(
            model.mtp[0].attn.window_kv_cache[0],
            torch.full((4, 3), 17.0),
        )
        self.assertTrue(
            torch.equal(model.engram_hash.cache[0, :6], torch.arange(6) + 20)
        )

        cache.clear_slot(0)
        partial = {
            **row([5, 6], [0, 8]),
            "prefix_state_slot": 8,
        }
        cache.restore(7, 0, partial, start_pos=6)
        self.assertTrue(
            torch.equal(model.engram_hash.cache[0, 4:6], torch.arange(4, 6) + 20)
        )

        cache.mamba_pages[7].tails[0][0].zero_()
        cache.mamba_pages[7].tails[0][1].fill_(-torch.inf)
        cache.restore(7, 0, {**partial, "mamba_block_table": [0, 7]}, start_pos=6)
        self.assertTrue(
            torch.equal(
                model.layers[0].attn.compressor.kv_state[0], torch.full((2, 3), 11)
            )
        )
        self.assertTrue(
            torch.equal(
                model.layers[0].attn.compressor.score_state[0],
                torch.full((2, 3), 13),
            )
        )
        model.layers[0].attn.compressor.kv_state[0].fill_(22)
        model.layers[0].attn.compressor.score_state[0].fill_(23)
        accepted = {**partial, "mamba_block_table": [0, 7]}
        cache.commit(7, accepted, 0, start_pos=6, accepted_end=6)
        cache.restore(
            7,
            0,
            {**accepted, "prefix_state_slot": 8},
            start_pos=6,
            use_prefix_state=False,
        )
        self.assertTrue(
            torch.equal(
                model.layers[0].attn.compressor.kv_state[0],
                torch.full((2, 3), 22),
            )
        )

        with self.assertRaisesRegex(ValueError, "request does not match"):
            cache.handle(
                {
                    "operations": [
                        {
                            "transfer_id": 3,
                            "snapshot_id": 9,
                            "kind": "load",
                            "request_id": 8,
                            "fa_blocks": [5, 6],
                            "mamba_blocks": [7, 8],
                        }
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown physical page"):
            cache.handle(
                {
                    "operations": [
                        {
                            "transfer_id": 4,
                            "snapshot_id": 10,
                            "kind": "store",
                            "request_id": None,
                            "fa_blocks": [99],
                            "mamba_blocks": [7],
                        }
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "shorter"):
            cache.commit(7, row([], []), 0, accepted_end=1)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            cache.commit(
                7,
                {
                    **accepted,
                    "cache_checkpoint": ("invalid", 8),
                },
                0,
                accepted_end=6,
            )

        cache.copy_prefix([(5, 9, 2)])
        self.assertTrue(
            torch.equal(cache.fa_pages[9].engram[0][:2], torch.tensor([20, 21]))
        )


if __name__ == "__main__":
    unittest.main()
