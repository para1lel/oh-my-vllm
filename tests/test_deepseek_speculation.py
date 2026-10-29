import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.inference import (
    DEFAULT_PREFILL_ROWS,
    MAX_PREFILL_ROWS,
    Worker,
    prefill_row_limit,
)
from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts

from tests.test_deepseek_cache import FakeModel


class _PositionModel(FakeModel):
    def __init__(self, *, value_offset=1, predictions=None, batch_size=2):
        super().__init__()
        self.value_offset, self.predictions = value_offset, predictions
        self.engram_hash.cache = torch.zeros(batch_size, 32, dtype=torch.int64)
        attention = self.layers[0].attn
        attention.window_kv_cache = torch.zeros(batch_size, 4, 3)
        attention.compress_kv_cache = torch.zeros(batch_size, 16, 3)
        attention.indexer.k_cache = torch.zeros(batch_size, 16, 2)
        attention.compressor.kv_state = torch.zeros(batch_size, 2, 3)
        attention.compressor.score_state = torch.zeros(batch_size, 2, 3)
        self.mtp[0].attn.window_kv_cache = torch.zeros(batch_size, 4, 3)

    def __call__(self, inputs, *, start_pos, cache_slots_cpu, **kwargs):
        self._state_export.bind(cache_slots_cpu)
        attention = self.layers[0].attn
        positions = torch.arange(start_pos, start_pos + inputs.size(1))
        offsets = torch.tensor(cache_slots_cpu).view(-1, 1, 1) * 100
        values = (
            (positions + self.value_offset).float().view(1, -1, 1) + offsets
        ).expand(-1, -1, 3)
        self._state_export.window(attention.window_kv_cache, values, start_pos)
        self._state_export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            values,
            values + 100,
            start_pos,
        )
        for index, slot in enumerate(cache_slots_cpu):
            attention.window_kv_cache[slot, positions[-4:].remainder(4)] = values[
                index, -4:
            ]
            self.engram_hash.cache[slot, positions] = positions
        logits = (
            None
            if self.predictions is None
            else torch.nn.functional.one_hot(
                torch.tensor([self.predictions]), 16
            ).float()
        )
        return None, logits, values

    def forward_spec_seed(self, hidden, *, start_pos, cache_slots):
        self._state_export.bind(cache_slots.tolist())
        window = self.mtp[0].attn.window_kv_cache
        self._state_export.window(window, hidden, start_pos)
        positions = torch.arange(start_pos, start_pos + hidden.size(1))
        for index, slot in enumerate(cache_slots.tolist()):
            window[slot, positions[-4:].remainder(4)] = hidden[index, -4:]


def _verification_worker(predictions):
    worker = Worker.__new__(Worker)
    worker.model = _PositionModel(value_offset=0, predictions=predictions)
    worker.device = torch.device("cpu")
    worker.dspark = True
    worker.args = SimpleNamespace(window_size=4)
    worker.slots = {1}
    worker.requests = {
        1: SimpleNamespace(
            cache_slot=0,
            history=[10] * 20,
            prompt_len=20,
            state_position=0,
            generated=0,
            limit=8,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
    }
    worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
    worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=32)
    return worker, {
        "request_id": 1,
        "token_ids": [10] * 20 + [1, 2, 3, 4, 5],
        "num_computed_tokens": 0,
        "fa_block_table": [1, 2, 3, 4, 5, 6, 7],
        "mamba_block_table": [0, 0, 0, 3, 5, 6, 7],
    }


class DeepSeekSpeculation(unittest.TestCase):
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
        self.assertEqual(worker.cache.fa_pages, {})
        self.assertTrue(
            all(page.endpoint == 0 for page in worker.cache.mamba_pages.values())
        )
        self.assertEqual(worker.cache.exports.steps, {})
        self.assertIsNone(worker.requests[1].cache_slot)
        self.assertEqual(worker.slots, {0, 1})

    def test_prefill_subbatches_keep_different_checkpoint_plans_and_slot_order(self):
        worker = Worker.__new__(Worker)
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
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=32)
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
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        from tests.test_deepseek_cache import FakeModel

        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=None, state_position=0, history=[1])
        }
        worker.slots = {0}
        worker.cache = DeepSeekCache(FakeModel(), block_tokens=4, max_seq_len=8)
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
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        worker = Worker.__new__(Worker)
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
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=16)
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

    def test_long_verification_seeds_mtp_from_the_accepted_hidden_tail(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = set()
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        request = SimpleNamespace(
            cache_slot=0,
            history=[10] * 20,
            prompt_len=20,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
        worker.requests = {1: request}
        ring = torch.zeros(4)

        class Model:
            def __call__(self, inputs, *, start_pos, **kwargs):
                hidden = torch.arange(start_pos, start_pos + inputs.size(1)).float()
                predictions = torch.tensor([[1, 2, 9, 9, 9, 9]])
                return (
                    None,
                    torch.nn.functional.one_hot(predictions, 16).float(),
                    hidden.view(1, -1, 1),
                )

            def forward_spec_seed(self, hidden, *, start_pos, cache_slots):
                positions = torch.arange(start_pos, start_pos + hidden.size(1))
                ring[positions.remainder(4)] = hidden.flatten()

        worker.model = Model()
        output = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10] * 20 + [1, 2, 3, 4, 5],
                "num_computed_tokens": 0,
            }
        )
        self.assertEqual(output["num_accepted_draft_tokens"], 2)
        self.assertEqual(request.state_position, 22)
        torch.testing.assert_close(ring, torch.tensor([20.0, 21, 18, 19]))

    def test_worker_rejection_restores_rows_overwritten_by_future_drafts(self):
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        from tests.test_deepseek_cache import FakeModel

        worker = Worker.__new__(Worker)
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
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=8)
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

    def test_execute_rejects_a_resident_position_mismatch(self):
        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=0, state_position=4),
        }
        with self.assertRaisesRegex(ValueError, "position"):
            worker.execute(
                {
                    "finished_request_ids": [],
                    "preempted_request_ids": [],
                    "scheduled": [
                        {
                            "request_id": 1,
                            "token_ids": [],
                            "num_computed_tokens": 3,
                        }
                    ],
                    "rpc_id": 1,
                }
            )

    def test_execute_rejects_unknown_lifecycle_request(self):
        worker = Worker.__new__(Worker)
        worker.requests = {}
        with self.assertRaisesRegex(ValueError, "unknown request"):
            worker.execute(
                {
                    "finished_request_ids": [1],
                    "preempted_request_ids": [],
                    "scheduled": [],
                    "rpc_id": 1,
                }
            )

    def test_restore_failure_releases_the_new_request_slot(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(cache_slot=None, history=[10])
        worker.requests = {1: request}
        worker.slots = {0}
        worker.cache = SimpleNamespace(
            restore=Mock(side_effect=ValueError("bad state")),
            clear_slot=Mock(),
            ensure_mamba_row=Mock(),
            begin_step=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "bad state"):
            worker._execute_row(
                {
                    "request_id": 1,
                    "token_ids": [10],
                    "num_computed_tokens": 0,
                }
            )
        self.assertIsNone(request.cache_slot)
        self.assertEqual(worker.slots, {0})

    def test_commit_failure_releases_the_request_slot(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            cache_slot=None,
            state_position=0,
            history=[10],
            prompt_len=1,
            limit=1,
            generated=0,
            finish=None,
            thinking_count=0,
            append=Mock(return_value=([11], "ok")),
        )
        worker.requests = {1: request}
        worker.slots = {0}
        worker.device = torch.device("cpu")
        worker.dspark = False
        worker.model = Mock(
            return_value=(
                None,
                torch.tensor([[1.0]]),
                None,
            )
        )
        worker.cache = SimpleNamespace(
            restore=Mock(),
            ensure_mamba_row=Mock(),
            begin_step=Mock(),
            commit=Mock(side_effect=ValueError("commit failed")),
            clear_slot=Mock(),
        )
        worker.trace = SimpleNamespace(record=Mock())
        with self.assertRaisesRegex(ValueError, "commit failed"):
            worker._execute_row(
                {
                    "request_id": 1,
                    "token_ids": [10],
                    "num_computed_tokens": 0,
                }
            )
        self.assertIsNone(request.cache_slot)
        self.assertEqual(worker.slots, {0})
        worker.cache.clear_slot.assert_called_once_with(0)

    def test_register_rejects_a_duplicate_request_identifier(self):
        worker = Worker.__new__(Worker)
        worker.requests = {1: object()}
        worker.runtime_max_context = 8
        with (
            patch("oh_my_vllm.models.deepseek_v41.inference.Request"),
            self.assertRaisesRegex(ValueError, "already owned"),
        ):
            worker.handle(
                {
                    "type": "register",
                    "request_id": 1,
                    "prompt_token_ids": [3],
                    "max_tokens": 1,
                }
            )

    def test_worker_emits_bonus_token_after_full_draft_acceptance(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            history=[10, 11],
            prompt_len=2,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([5, 6, 7], "ok")),
        )
        worker.requests = {1: request}
        worker.dspark = True
        worker._slot = lambda _: 0
        worker._proposal = Mock(return_value=[])
        worker.trace = SimpleNamespace(record=Mock())
        logits = [
            torch.nn.functional.one_hot(torch.tensor([5]), 8).float(),
            torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
            torch.nn.functional.one_hot(torch.tensor([7]), 8).float(),
        ]
        worker._target = Mock(return_value=(None, torch.stack(logits, dim=1), None))

        result = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 11, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(result["token_ids"], [5, 6, 7])
        self.assertEqual(result["num_accepted_draft_tokens"], 2)
        worker._target.assert_called_once()
        self.assertTrue(worker._target.call_args.kwargs["full_logits"])
        self.assertEqual(worker._target.call_args.kwargs["logit_start"], 1)

    def test_worker_proposes_from_the_last_target_token_in_a_prefill_chunk(self):
        worker = Worker.__new__(Worker)
        hidden = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
        request = SimpleNamespace(
            history=[10, 11],
            prompt_len=2,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([5], "ok")),
        )
        worker.requests = {1: request}
        worker.dspark = True
        worker._slot = lambda _: 0
        worker._proposal = Mock(return_value=[6, 7])
        worker.trace = SimpleNamespace(record=Mock())
        worker._target = Mock(
            return_value=(
                None,
                torch.nn.functional.one_hot(torch.tensor([[5]]), 8).float(),
                hidden,
            )
        )

        worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 11],
                "num_computed_tokens": 0,
            }
        )

        worker._proposal.assert_called_once()
        token, proposal_hidden, proposal_start, slot = worker._proposal.call_args.args
        self.assertEqual(token, 5)
        torch.testing.assert_close(proposal_hidden, hidden[:, -1:])
        self.assertEqual(proposal_start, 1)
        self.assertEqual(slot, 0)

    def test_verification_seeds_mtp_state_for_accepted_target_positions(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            history=[10, 5],
            prompt_len=1,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([6, 7], "ok")),
        )
        hidden = torch.arange(9, dtype=torch.float32).view(1, 3, 3)
        worker.requests = {1: request}
        worker.dspark = True
        worker.device = torch.device("cpu")
        worker._slot = lambda _: 0
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        worker._proposal = Mock(return_value=[])
        worker.trace = SimpleNamespace(record=Mock())
        worker._target = Mock(
            return_value=(
                None,
                torch.stack(
                    [
                        torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
                        torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
                    ],
                    dim=1,
                ),
                hidden,
            )
        )

        worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(worker.model.forward_spec_seed.call_count, 1)
        self.assertEqual(
            [
                call.kwargs["start_pos"]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [0],
        )
        self.assertEqual(
            worker.model.forward_spec_seed.call_args.args[0].shape[1],
            3,
        )

    def test_target_keeps_long_prefill_in_one_native_call(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.model = Mock(
            side_effect=[
                ("last", "last logits", "last hidden"),
            ]
        )

        result = worker._target([1] * 385, 7, 3)

        self.assertEqual(result, ("last", "last logits", "last hidden"))
        self.assertEqual(worker.model.call_count, 1)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7],
        )

    def test_target_default_prefill_limit_bounds_model_chunks(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.model = Mock(
            side_effect=[
                ("first", None, None),
                ("last", None, None),
            ]
        )

        worker._target([1] * (DEFAULT_PREFILL_ROWS + 1), 7, 3, compute_logits=False)

        self.assertEqual(worker.model.call_count, 2)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7, 7 + DEFAULT_PREFILL_ROWS],
        )

    def test_prefill_row_limit_keeps_explicit_operator_bound(self):
        self.assertEqual(prefill_row_limit(None), DEFAULT_PREFILL_ROWS)
        self.assertEqual(prefill_row_limit(str(MAX_PREFILL_ROWS)), MAX_PREFILL_ROWS)
        with self.assertRaisesRegex(ValueError, "prefill row limit"):
            prefill_row_limit(str(MAX_PREFILL_ROWS + 1))

    def test_target_releases_completed_prefill_hidden_chunks(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.args = SimpleNamespace(window_size=2)
        references = []

        def model(_inputs, **_):
            hidden = torch.ones(1, 1, 2)
            if references:
                gc.collect()
                self.assertIsNone(references[0]())
            references.append(weakref.ref(hidden))
            return torch.empty(1, dtype=torch.long), None, hidden

        worker.model = Mock(side_effect=model)
        result = worker._target(
            [1] * (MAX_PREFILL_ROWS + 1), 7, 3, compute_logits=False
        )

        self.assertEqual(tuple(result[2].shape), (1, 2, 2))
        self.assertEqual(worker.model.call_count, 2)

    def test_target_uses_the_bounded_prefill_row_limit(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = 128
        worker.model = Mock(
            side_effect=[
                ("first", None, None),
                ("second", None, None),
                ("last", None, None),
            ]
        )

        worker._target([1] * 257, 7, 3, compute_logits=False)

        self.assertEqual(worker.model.call_count, 3)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7, 135, 263],
        )

    def test_target_batch_keeps_total_rows_in_one_native_call(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS

        def model(inputs, **_):
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 200 for index in range(4)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(4),
        )

        self.assertEqual(worker.model.call_count, 1)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7],
        )

    def test_mtp_seed_keeps_total_rows_at_native_limit(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        hidden = torch.zeros(4, 200, 8)
        cache_slots = torch.arange(4)

        self.assertTrue(worker._seed_mtp_batch(hidden, 7, cache_slots))

        self.assertEqual(worker.model.forward_spec_seed.call_count, 3)
        self.assertEqual(
            [
                call.args[0].shape[1]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [96, 96, 8],
        )
        self.assertEqual(
            [
                call.kwargs["start_pos"]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [7, 103, 199],
        )

    def test_target_batch_uses_four_request_prefill_subbatches(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS

        def model(inputs, **_):
            self.assertLessEqual(inputs.size(0), 4)
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 100 for index in range(8)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(8),
        )

        self.assertEqual(worker.model.call_count, 2)

    def test_mtp_seed_keeps_only_the_target_window(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.args = SimpleNamespace(window_size=8)
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        hidden = torch.zeros(1, 200, 8)

        self.assertTrue(
            worker._seed_mtp_batch(
                hidden,
                7,
                torch.tensor([0]),
                source_length=200,
            )
        )

        worker.model.forward_spec_seed.assert_called_once()
        call = worker.model.forward_spec_seed.call_args
        self.assertEqual(tuple(call.args[0].shape), (1, 8, 8))
        self.assertEqual(call.kwargs["start_pos"], 199)

    def test_execute_groups_complete_initial_prefills(self):
        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=None, history=[10, 11, 12]),
            2: SimpleNamespace(cache_slot=None, history=[20, 21, 22]),
        }
        worker._execute_prefill_group = Mock(
            return_value={
                1: {"request_id": 1},
                2: {"request_id": 2},
            }
        )
        worker._execute_row = Mock()

        result = worker.execute(
            {
                "finished_request_ids": [],
                "preempted_request_ids": [],
                "scheduled": [
                    {
                        "request_id": 1,
                        "token_ids": [10, 11, 12],
                        "num_computed_tokens": 0,
                    },
                    {
                        "request_id": 2,
                        "token_ids": [20, 21, 22],
                        "num_computed_tokens": 0,
                    },
                ],
                "rpc_id": 1,
            }
        )

        worker._execute_prefill_group.assert_called_once()
        self.assertEqual(
            worker._execute_prefill_group.call_args.kwargs,
            {"final": True},
        )
        worker._execute_row.assert_not_called()
        self.assertEqual(result["outputs"], [{"request_id": 1}, {"request_id": 2}])

    def test_rejection_keeps_only_the_matching_prefix_and_target_correction(self):
        result = verify_greedy_drafts(
            drafts=[11, 12, 13],
            target_predictions=[11, 99, 14, 15],
        )

        self.assertEqual(result.accepted_draft_tokens, 1)
        self.assertEqual(result.token_ids, (11, 99))
        self.assertTrue(result.rejected)

    def test_full_acceptance_returns_the_target_bonus_token(self):
        result = verify_greedy_drafts(
            drafts=[11, 12, 13],
            target_predictions=[11, 12, 13, 17],
        )

        self.assertEqual(result.accepted_draft_tokens, 3)
        self.assertEqual(result.token_ids, (11, 12, 13, 17))
        self.assertFalse(result.rejected)

    def test_empty_draft_is_ordinary_target_decode(self):
        result = verify_greedy_drafts(
            drafts=[],
            target_predictions=[23],
        )

        self.assertEqual(result.accepted_draft_tokens, 0)
        self.assertEqual(result.token_ids, (23,))
        self.assertFalse(result.rejected)


if __name__ == "__main__":
    unittest.main()
