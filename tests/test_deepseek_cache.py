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
        cache.commit(7, source, 0, accepted_end=4)
        cache.mamba_pages[3].engram[0].copy_(torch.tensor([104, 105, 0, 0]))
        cache.mamba_pages[3].valid_tokens = 2

        model.engram_hash.cache[0, :2] = torch.tensor([104, 105])
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
        source = row([1, 2], [3, 4])
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
        cache.commit(7, source, 0, accepted_end=6, include_mtp=True)

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
        cache.commit(7, accepted, 0, accepted_end=6)
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
        cache.clear_slot(0)
        copied = row([9, 6], [7, 8])
        cache.restore(7, 0, copied, start_pos=2)
        torch.testing.assert_close(
            model.layers[0].attn.window_kv_cache[0, :2],
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
        )


if __name__ == "__main__":
    unittest.main()
