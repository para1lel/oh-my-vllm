import unittest
from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache


class FakeModel:
    def __init__(self):
        self.layers = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_kv_cache=torch.zeros(2, 4, 3),
                    compress_kv_cache=torch.zeros(2, 4, 3),
                    compress_ratio=2,
                    indexer=SimpleNamespace(
                        k_cache=torch.zeros(2, 4, 2),
                    ),
                    compressor=SimpleNamespace(
                        kv_state=torch.zeros(2, 2, 3),
                        score_state=torch.zeros(2, 2, 3),
                    ),
                )
            )
        ]
        self.mtp = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_kv_cache=torch.zeros(2, 4, 3),
                )
            )
        ]
        self.engram_hash = SimpleNamespace(cache=torch.zeros(2, 8, dtype=torch.int64))


def row(fa, mamba):
    return {
        "request_id": 7,
        "fa_block_table": fa,
        "mamba_block_table": mamba,
    }


class DeepSeekCacheBehavior(unittest.TestCase):
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
                cache = DeepSeekCache(model, block_tokens=4, max_seq_len=8)
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
        cache = DeepSeekCache(model, block_tokens=4, max_seq_len=8)
        source = row([1, 2], [3, 0])
        cache.ensure_row(source)
        cache.commit(7, source, 0, accepted_end=4)
        for endpoint in (2, 3, 5):
            with self.subTest(endpoint=endpoint):
                model.layers[0].attn.window_kv_cache[0].fill_(71)
                with self.assertRaisesRegex(ValueError, "endpoint"):
                    cache.restore(7, 0, source, start_pos=endpoint)
                torch.testing.assert_close(
                    model.layers[0].attn.window_kv_cache[0], torch.full((4, 3), 71.0)
                )

    def test_fa_page_growth_does_not_replicate_window_or_tail_storage(self):
        def storage_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, (list, tuple)):
                return sum(storage_bytes(item) for item in value)
            if hasattr(value, "__dataclass_fields__"):
                return sum(storage_bytes(item) for item in vars(value).values())
            return 0

        cache = DeepSeekCache(FakeModel(), block_tokens=4, max_seq_len=64)
        cache.ensure_row(row([1], [31]))
        initial = storage_bytes(list(cache.fa_pages.values()))
        cache.ensure_row(row(list(range(1, 17)), [0] * 15 + [31]))
        added = storage_bytes(list(cache.fa_pages.values())) - initial
        self.assertEqual(added, 15 * 72)
        self.assertEqual(
            cache.stats()["gpu_logical_page_bytes"],
            storage_bytes([*cache.fa_pages.values(), *cache.mamba_pages.values()]),
        )

    def test_commit_merges_prefix_state_into_a_new_mamba_page(self):
        model = FakeModel()
        cache = DeepSeekCache(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        source = row([1], [3])
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

    def test_commit_keeps_compressed_rows_completed_by_decode(self):
        model = FakeModel()
        cache = DeepSeekCache(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        source = row([1, 2], [0, 4])
        cache.commit(7, source, 0, accepted_end=6)

        model.layers[0].attn.compress_kv_cache[0, 3] = torch.full((3,), 31)
        model.layers[0].attn.indexer.k_cache[0, 3] = torch.full((2,), 37)
        cache.commit(7, source, 0, start_pos=7, accepted_end=8)

        cache.clear_slot(0)
        cache.restore(7, 0, source, start_pos=8)
        torch.testing.assert_close(
            model.layers[0].attn.compress_kv_cache[0, 3],
            torch.full((3,), 31.0),
        )
        torch.testing.assert_close(
            model.layers[0].attn.indexer.k_cache[0, 3],
            torch.full((2,), 37.0),
        )

    def test_restore_rejects_missing_accepted_mamba_state(self):
        model = FakeModel()
        cache = DeepSeekCache(model, block_tokens=4, max_seq_len=8)
        cache._fa_page(1)
        with self.assertRaisesRegex(ValueError, "state page"):
            cache.restore(7, 0, row([1], [0]), start_pos=4)
        with self.assertRaisesRegex(ValueError, "state page"):
            cache.restore(7, 0, row([1], [3]), start_pos=4)

    def test_host_transfers_require_a_live_resident_request(self):
        model = FakeModel()
        requests = {7: SimpleNamespace(cache_slot=0)}
        cache = DeepSeekCache(
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
        cache = DeepSeekCache(
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
        cache = DeepSeekCache(
            model,
            block_tokens=4,
            max_seq_len=8,
            host_limit=1 << 20,
        )
        model.layers[0].attn.window_kv_cache[0] = torch.arange(12).reshape(4, 3)
        model.layers[0].attn.compress_kv_cache[0, :2] = torch.arange(6).reshape(2, 3)
        model.layers[0].attn.indexer.k_cache[0, :2] = torch.arange(4).reshape(2, 2)
        model.layers[0].attn.compressor.kv_state[0] = 11
        model.layers[0].attn.compressor.score_state[0] = 13
        model.mtp[0].attn.window_kv_cache[0] = 17
        model.engram_hash.cache[0, :6] = torch.arange(6) + 20
        source = row([1, 2], [3, 4])
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
            model.layers[0].attn.compress_kv_cache[0, :2],
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
        )
        torch.testing.assert_close(
            model.layers[0].attn.indexer.k_cache[0, :2],
            torch.arange(4, dtype=torch.float32).reshape(2, 2),
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
        torch.testing.assert_close(
            cache.fa_pages[9].compressed[0][0], torch.arange(3).float()
        )
        torch.testing.assert_close(
            cache.fa_pages[9].index[0][0], torch.arange(2).float()
        )
        self.assertTrue(
            torch.equal(cache.fa_pages[9].engram[0][:2], torch.tensor([20, 21]))
        )


if __name__ == "__main__":
    unittest.main()
