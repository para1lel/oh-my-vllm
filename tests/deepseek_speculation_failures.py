import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tests.deepseek_speculation_fixtures import (
    DeepSeekSpeculationFixture,
    _chain_batch,
    _chain_row,
    _chain_worker,
    _PositionModel,
    _verification_worker,
    _worker,
)
from tests.test_deepseek_cache import cache_for


class DeepSeekSpeculationFailures(DeepSeekSpeculationFixture, unittest.TestCase):
    def test_first_request_failure_before_commit_keeps_pages_unpublished(self):
        for batched in (False, True):
            for failure in ("seed", "proposal", "export", "commit"):
                with self.subTest(batched=batched, failure=failure):
                    worker = _chain_worker(prompts=((10,),))
                    scheduled = _chain_row(worker, [10, 1, 2, 3, 4, 5])
                    owner, name = {
                        "seed": (worker.model, "forward_spec_seed"),
                        "proposal": (worker.model, "forward_spec"),
                        "export": (worker.cache.exports, "window_state"),
                        "commit": (worker.cache, "commit"),
                    }[failure]
                    with (
                        patch.object(owner, name, side_effect=ValueError(failure)),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.batch.target_decode_batch",
                            side_effect=_chain_batch,
                        ),
                        self.assertRaisesRegex(ValueError, failure),
                    ):
                        worker._execute_decode_group(
                            [scheduled]
                        ) if batched else worker._execute_row(scheduled)
                    self.assertTrue(
                        all(
                            page.valid_tokens == 0
                            for page in worker.cache.fa_pages.values()
                        )
                    )
                    self.assertTrue(
                        all(
                            page.endpoint == 0
                            for page in worker.cache.mamba_pages.values()
                        )
                    )
                    self.assertEqual(worker.cache.exports.steps, {})
                    self.assertIsNone(worker.requests[1].cache_slot)
                    self.assertEqual(worker.requests[1].state_position, 0)
                    self.assertEqual(worker.slots, {0, 1})

    def test_later_batch_proposal_failure_discards_the_group_after_an_earlier_commit(
        self,
    ):
        from oh_my_vllm.models.deepseek_v41.launcher import GroupWorker

        worker = _chain_worker(prompts=((10, 11, 12), (20, 21, 22)), acceptance=2)
        worker.batch_decode_enabled, worker.decode_cycle = True, 2
        outputs = {
            key: worker._execute_row(_chain_row(worker, request.history, key=key))
            for key, request in worker.requests.items()
        }
        rows = [
            _chain_row(
                worker,
                [request.history[-1], *outputs[key]["new_draft_token_ids"]],
                key=key,
            )
            for key, request in worker.requests.items()
        ]
        histories = {
            key: list(request.history) for key, request in worker.requests.items()
        }
        failed_slot = worker.requests[2].cache_slot
        proposal = worker._proposal

        def fail_second(token, hidden, start, slot):
            if slot == failed_slot:
                raise RuntimeError("second proposal failed")
            return proposal(token, hidden, start, slot)

        message = dict(
            type="execute",
            rpc_id=3,
            scheduled=rows,
            finished_request_ids=[],
            preempted_request_ids=[],
        )
        bridge = GroupWorker.__new__(GroupWorker)
        bridge.commands = []
        bridge.close = Mock()
        bridge._collect = Mock(
            side_effect=lambda _timeout: {0: worker.execute(message)}
        )
        with (
            patch.object(worker, "_proposal", side_effect=fail_second) as propose,
            patch.object(worker.cache, "commit", wraps=worker.cache.commit) as commit,
            patch(
                "oh_my_vllm.models.deepseek_v41.batch.target_decode_batch",
                side_effect=_chain_batch,
            ),
            self.assertRaisesRegex(RuntimeError, "second proposal failed"),
        ):
            bridge.handle(message)
        self.assertEqual(propose.call_count, 2)
        commit.assert_called_once()
        self.assertEqual(commit.call_args.args[0], 1)
        bridge.close.assert_called_once()
        # Rust must receive no successful Execute result from this discarded group.
        self.assertEqual(worker.cache.fa_pages[1].valid_tokens, 4)
        self.assertEqual(worker.cache.mamba_pages[2].endpoint, 6)
        self.assertEqual(worker.cache.fa_pages[17].valid_tokens, 3)
        self.assertEqual(worker.cache.mamba_pages[17].endpoint, 3)
        self.assertEqual(worker.cache.fa_pages[18].valid_tokens, 0)
        for key, request in worker.requests.items():
            self.assertEqual(request.generated, 4)
            self.assertNotEqual(request.history, histories[key])
            self.assertIsNone(request.cache_slot)
            self.assertEqual(request.state_position, 0)
        self.assertEqual(worker.cache.exports.steps, {})
        self.assertEqual(worker.slots, {0, 1, 2})

    def test_stopped_verification_commits_only_the_visible_accepted_prefix(self):
        worker, scheduled = _verification_worker([1, 2, 3, 4, 5, 9])
        request = worker.requests[1]

        def stop(tokens):
            request.finish = "stop"
            return tokens[:2], "ok"

        request.append = stop
        result = worker._execute_row(scheduled)
        self.assertEqual(result["token_ids"], [1, 2])
        self.assertEqual(result["num_accepted_draft_tokens"], 1)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(request.state_position, 21)
        worker.cache.restore(1, 1, {**scheduled, "prefix_state_slot": 6}, start_pos=21)
        for window in (
            worker.model.layers[0].attn.window_kv_cache,
            worker.model.mtp[0].attn.window_kv_cache,
        ):
            torch.testing.assert_close(
                window[1, :, 0], torch.tensor([20.0, 17, 18, 19]), rtol=0, atol=0
            )

    def test_missing_mtp_export_fails_before_physical_page_publication(self):
        worker, scheduled = _verification_worker([1, 2, 9, 9, 9, 9])
        worker.model.forward_spec_seed = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ValueError, "window state at endpoint"):
            worker._execute_row(scheduled)
        self.assertTrue(
            all(page.valid_tokens == 0 for page in worker.cache.fa_pages.values())
        )
        self.assertTrue(
            all(page.endpoint == 0 for page in worker.cache.mamba_pages.values())
        )
        self.assertEqual(worker.cache.exports.steps, {})
        self.assertIsNone(worker.requests[1].cache_slot)
        self.assertEqual(worker.slots, {0, 1})

    def test_prefill_subbatches_keep_different_checkpoint_plans_and_slot_order(self):
        worker = _worker()
        worker.model = _PositionModel(batch_size=9)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.prefill_rows = 8
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = {8}
        worker.requests = {
            slot: SimpleNamespace(
                cache_slot=slot,
                history=list(range(1, 33)),
                prompt_len=32,
                state_position=0,
                finish=None,
                thinking_count=0,
            )
            for slot in range(8)
        }
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = cache_for(worker.model, block_tokens=4, max_seq_len=32)
        scheduled = [
            {
                "request_id": slot,
                "token_ids": list(range(1, 17)),
                "num_computed_tokens": 0,
                "fa_block_table": list(range(slot * 4 + 1, slot * 4 + 5)),
                "mamba_block_table": [0, 0, 0, 20 + slot],
            }
            for slot in range(8)
        ]
        scheduled[0]["cache_checkpoint"] = (3, 50)
        scheduled[5]["cache_checkpoint"] = (7, 55)
        worker._execute_prefill_group(scheduled)
        for slot, endpoint, source, expected in (
            (0, 3, 50, [1.0, 2, 3, 0]),
            (5, 7, 55, [505.0, 506, 507, 504]),
        ):
            with self.subTest(slot=slot):
                worker.cache.restore(
                    slot,
                    8,
                    {**scheduled[slot], "prefix_state_slot": source},
                    start_pos=endpoint,
                )
                for window in (
                    worker.model.layers[0].attn.window_kv_cache,
                    worker.model.mtp[0].attn.window_kv_cache,
                ):
                    torch.testing.assert_close(
                        window[8, :, 0], torch.tensor(expected), rtol=0, atol=0
                    )

    def test_verification_preserves_lookback_for_both_mtp_checkpoint_and_tail(self):
        worker, scheduled = _verification_worker([1, 2, 9, 9, 9, 9])
        worker._execute_row(scheduled)
        for endpoint, source, expected in (
            (20, 5, [16.0, 17, 18, 19]),
            (22, 6, [20.0, 21, 18, 19]),
        ):
            with self.subTest(endpoint=endpoint):
                worker.cache.restore(
                    1, 1, {**scheduled, "prefix_state_slot": source}, start_pos=endpoint
                )
                actual = worker.model.mtp[0].attn.window_kv_cache[1, :, 0]
                torch.testing.assert_close(
                    actual, torch.tensor(expected), rtol=0, atol=0
                )

    def test_prefill_export_failure_releases_the_new_request_slot(self):

        from tests.test_deepseek_cache import FakeModel, cache_for

        worker = _worker()
        worker.requests = {
            1: SimpleNamespace(cache_slot=None, state_position=0, history=[1])
        }
        worker.slots = {0}
        worker.cache = cache_for(FakeModel(), block_tokens=4, max_seq_len=8)
        scheduled = {
            "request_id": 1,
            "token_ids": [1],
            "num_computed_tokens": 0,
            "fa_block_table": [1],
            "mamba_block_table": [0],
        }
        with self.assertRaisesRegex(ValueError, "endpoint"):
            worker._execute_prefill_group([scheduled])
        self.assertIsNone(worker.requests[1].cache_slot)
        self.assertEqual(worker.slots, {0})

    def test_long_prefill_exports_mtp_checkpoint_before_discarding_hidden_rows(self):

        worker = _worker()
        worker.model = _PositionModel()
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.prefill_rows = 4
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = {1}
        worker.requests = {
            1: SimpleNamespace(
                cache_slot=0,
                history=list(range(1, 33)),
                prompt_len=32,
                state_position=0,
                finish=None,
                thinking_count=0,
            )
        }
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = cache_for(worker.model, block_tokens=4, max_seq_len=16)
        scheduled = {
            "request_id": 1,
            "token_ids": list(range(1, 17)),
            "num_computed_tokens": 0,
            "fa_block_table": [1, 2, 3, 4],
            "mamba_block_table": [0, 0, 0, 5],
            "cache_checkpoint": (3, 9),
        }
        worker._execute_prefill_group([scheduled])
        worker.cache.restore(1, 1, {**scheduled, "prefix_state_slot": 9}, start_pos=3)
        for window in (
            worker.model.layers[0].attn.window_kv_cache,
            worker.model.mtp[0].attn.window_kv_cache,
        ):
            torch.testing.assert_close(window[1, :, 0], torch.tensor([1.0, 2, 3, 0]))

    def test_worker_rejection_restores_rows_overwritten_by_future_drafts(self):

        from tests.test_deepseek_cache import FakeModel, cache_for

        worker = _worker()
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.model = FakeModel()
        request = SimpleNamespace(
            cache_slot=0,
            state_position=2,
            history=[10, 20, 30],
            prompt_len=3,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
        worker.requests = {1: request}
        worker.slots = {1}
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = cache_for(worker.model, block_tokens=4, max_seq_len=8)
        attention = worker.model.layers[0].attn
        attention.window_kv_cache[0, :2] = torch.tensor([[1.0], [2.0]])
        worker.cache.commit(
            1, {"fa_block_table": [1], "mamba_block_table": [3]}, 0, accepted_end=2
        )
        worker.cache.copy_prefix([(1, 7, 2)])

        def target(tokens, start_pos, slot, **kwargs):
            exports = worker.cache.exports
            exports.bind((slot,))
            values = torch.arange(3, 9).float().view(1, 6, 1).expand(1, 6, 3)
            exports.window(attention.window_kv_cache, values, start_pos)
            exports.tail(
                attention.compressor.kv_state,
                attention.compressor.score_state,
                values,
                values + 100,
                start_pos,
            )
            for position in range(2, 8):
                attention.window_kv_cache[slot, position % 4] = position + 1
            logits = torch.nn.functional.one_hot(torch.full((1, 6), 10), 16).float()
            return None, logits, None

        worker._target = target
        result = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [30, 5, 6, 7, 8, 9],
                "num_computed_tokens": 2,
                "fa_block_table": [7, 2],
                "mamba_block_table": [5, 6],
                "prefix_state_slot": 3,
            }
        )
        self.assertEqual(result["token_ids"], [10])
        self.assertEqual(request.state_position, 3)
        torch.testing.assert_close(
            attention.window_kv_cache[0, :, 0], torch.tensor([1.0, 2, 3, 0])
        )
        torch.testing.assert_close(
            attention.compressor.kv_state[0, 0], torch.full((3,), 3.0)
        )
        self.assertEqual(worker.cache.mamba_pages[3].endpoint, 2)
