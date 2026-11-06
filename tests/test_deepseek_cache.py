import copy
import unittest
from unittest.mock import Mock

import torch

from tests.deepseek_cache_fixtures import FakeModel, cache_for, row


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

    def test_slot_release_and_prefix_copy_restore_logical_blocks(self):
        model = FakeModel()
        cache = cache_for(
            model,
            block_tokens=4,
            max_seq_len=8,
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

        cache.clear_slot(0)
        cache.restore(7, 0, source, start_pos=6)
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

        accepted = self._check_cache_prefix_state(cache, model)

        self._check_cache_restore_errors(cache, accepted)

        cache.copy_prefix([(1, 9, 2)])
        self.assertTrue(
            torch.equal(cache.fa_pages[9].engram[0][:2], torch.tensor([20, 21]))
        )

    def _check_cache_restore_errors(self, cache, accepted):
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

    def _check_cache_prefix_state(self, cache, model):
        cache.clear_slot(0)
        partial = {
            **row([1, 2], [0, 4]),
            "prefix_state_slot": 4,
        }
        cache.restore(7, 0, partial, start_pos=6)
        self.assertTrue(
            torch.equal(model.engram_hash.cache[0, 4:6], torch.arange(4, 6) + 20)
        )

        cache.mamba_pages[3].tails[0][0].zero_()
        cache.mamba_pages[3].tails[0][1].fill_(-torch.inf)
        cache.restore(7, 0, {**partial, "mamba_block_table": [0, 3]}, start_pos=6)
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
        accepted = {**partial, "mamba_block_table": [0, 3]}
        cache.commit(7, accepted, 0, start_pos=6, accepted_end=6)
        cache.restore(
            7,
            0,
            {**accepted, "prefix_state_slot": 4},
            start_pos=6,
            use_prefix_state=False,
        )
        self.assertTrue(
            torch.equal(
                model.layers[0].attn.compressor.kv_state[0],
                torch.full((2, 3), 22),
            )
        )
        return accepted


class DeepSeekRankReplies(unittest.TestCase):
    def _replicated_replies(self):
        from oh_my_vllm.models.deepseek_v41.launcher import GroupWorker

        from tests.deepseek_speculation_fixtures import (
            _chain_row,
            _chain_worker,
            _wire_message,
        )

        workers = [
            _chain_worker(prompts=((10, 11, 12), (20, 21, 22))) for _ in range(4)
        ]
        for rank, worker in enumerate(workers):
            self.addCleanup(worker.cache.close)
            worker.rank = rank
        message = _wire_message(
            workers[0],
            dict(
                rpc_id=31,
                scheduled=[
                    _chain_row(workers[0], request.history, key=key)
                    for key, request in workers[0].requests.items()
                ],
                finished_request_ids=[],
                preempted_request_ids=[],
            ),
        )
        replies = {rank: worker.execute(message) for rank, worker in enumerate(workers)}
        bridge = GroupWorker.__new__(GroupWorker)
        bridge.commands, bridge._collect, bridge.close = [], Mock(), Mock()
        return bridge, message, replies

    @staticmethod
    def _invalid_replies(replies):
        cases = []
        for field in ("rpc_id", "step_id", "worker_epoch"):
            bad = copy.deepcopy(replies)
            bad[3][field] += 1
            cases.append((field, bad))
        for field, value in (
            ("lease_id", 999),
            ("purpose", "finalize"),
            ("action", "head_only"),
            ("request_id", 999),
            ("computed_end", 99),
            ("accepted_end", 99),
            ("token_ids", [99]),
            ("new_draft_token_ids", [99]),
            ("text", "different"),
            ("reasoning_tokens", 99),
            ("finish_reason", "stop"),
        ):
            bad = copy.deepcopy(replies)
            bad[3]["outputs"][0][field] = value
            cases.append((field, bad))
        for mode in (
            "rank_missing",
            "extra_rank",
            "row_missing",
            "row_duplicate",
            "type",
            "error",
            "row_error",
        ):
            bad = copy.deepcopy(replies)
            if mode == "rank_missing":
                bad.pop(3)
            elif mode == "extra_rank":
                bad[4] = copy.deepcopy(bad[3])
            elif mode == "row_missing":
                bad[3]["outputs"].pop()
            elif mode == "row_duplicate":
                bad[3]["outputs"].append(copy.deepcopy(bad[3]["outputs"][0]))
            elif mode == "type":
                bad[3]["type"] = "ack"
            elif mode == "error":
                bad[3] = dict(type="error", message="rank failed")
            else:
                bad[3]["outputs"][0].update(
                    error="rank failed",
                    token_ids=[],
                    new_draft_token_ids=[],
                    num_accepted_draft_tokens=0,
                )
            cases.append((mode, bad))
        return cases

    def test_tp_tep_validate_real_replies_and_ignore_rank_private_extensions(self):
        bridge, message, replies = self._replicated_replies()
        replies[3]["rank_private_debug"] = 3
        replies[3]["outputs"].reverse()
        replies[3]["outputs"][0]["rank_private_debug"] = 3
        bridge._collect.return_value = replies
        for strategy in ("tp4", "tep4"):
            bridge.strategy = strategy
            self.assertEqual(bridge.handle(message), replies[0])
        bridge.close.assert_not_called()

    def test_duplicate_rank_cannot_complete_the_reply_group(self):
        from oh_my_vllm.models.deepseek_v41.launcher import GroupWorker

        bridge, message, replies = self._replicated_replies()
        bridge.strategy = "tp4"
        bridge._collect.side_effect = lambda _timeout: GroupWorker._store_reply(
            dict(replies), 3, replies[3]
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            bridge.handle(message)
        bridge.close.assert_called_once()

    def test_tp_tep_reject_nonzero_rank_identity_data_errors_and_incomplete_sets(self):
        bridge, message, replies = self._replicated_replies()
        for strategy in ("tp4", "tep4"):
            bridge.strategy = strategy
            bridge._collect.return_value = replies
            self.assertEqual(bridge.handle(message), replies[0])
            for name, bad in self._invalid_replies(replies):
                bridge._collect.return_value = bad
                bridge.close.reset_mock()
                with self.subTest(strategy=strategy, mutation=name):
                    with self.assertRaises((ValueError, RuntimeError)):
                        bridge.handle(message)
                    bridge.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
