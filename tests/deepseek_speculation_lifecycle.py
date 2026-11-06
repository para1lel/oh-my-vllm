import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.inference import prefill_row_limit
from oh_my_vllm.models.deepseek_v41.worker_limits import (
    DEFAULT_PREFILL_ROWS,
    MAX_PREFILL_ROWS,
)

from tests.deepseek_speculation_fixtures import (
    DeepSeekSpeculationFixture,
    _chain_row,
    _chain_worker,
    _verification_worker,
    _wire_message,
    _worker,
)


class DeepSeekSpeculationLifecycle(DeepSeekSpeculationFixture, unittest.TestCase):
    def test_abort_releases_graph_lease_before_slot_reuse(self):
        worker = _chain_worker()
        worker.slots.intersection_update({0})
        self.addCleanup(worker.cache.close)
        request = worker.requests[1]
        scheduled = _chain_row(worker, request.history)
        worker._execute_row(scheduled)
        slot = request.cache_slot
        lease = worker.cache.page_table.acquire()
        worker.proposal_graphs = {slot: SimpleNamespace(close=lease.close)}
        worker.handle(dict(type="abort", request_id=1))
        self.assertIsNone(request.cache_slot)
        self.assertEqual(worker.cache.page_table.leases, 0)
        worker.tokenizer = request.tokenizer
        worker.handle(
            dict(
                type="register",
                request_id=2,
                prompt_token_ids=[20, 21, 22],
                max_tokens=4,
            )
        )
        self.assertEqual(worker._slot(2), slot)
        worker.cache.reset()
        self.assertEqual(worker.cache.page_table.leases, 0)

    def test_execute_rejects_a_resident_position_mismatch(self):
        worker = _worker()
        worker.requests = {
            1: SimpleNamespace(cache_slot=0, state_position=4, history=[10] * 5),
        }
        with self.assertRaisesRegex(ValueError, "position"):
            worker.execute(
                _wire_message(
                    worker,
                    {
                        "finished_request_ids": [],
                        "preempted_request_ids": [],
                        "scheduled": [
                            {
                                "request_id": 1,
                                "token_ids": [10],
                                "num_computed_tokens": 3,
                            }
                        ],
                        "rpc_id": 1,
                    },
                )
            )

    def test_execute_restarts_a_preempted_request_in_the_same_step(self):
        worker, scheduled = _verification_worker([1])
        request = worker.requests[1]
        request.state_position = 4
        attention = worker.model.layers[0].attn
        attention.window_kv_cache[0].fill_(99)
        scheduled["token_ids"] = [10, 10, 10]
        scheduled["mamba_block_table"] = [1, 2, 3, 4, 5, 6, 7]

        result = worker.execute(
            _wire_message(
                worker,
                {
                    "rpc_id": 7,
                    "finished_request_ids": [],
                    "preempted_request_ids": [1],
                    "scheduled": [scheduled],
                },
            )
        )

        self.assertEqual(result["outputs"][0]["token_ids"], [])
        self.assertEqual(request.state_position, 3)
        self.assertIsNotNone(request.cache_slot)
        torch.testing.assert_close(
            attention.window_kv_cache[request.cache_slot, :, 0],
            torch.tensor([0.0, 1, 2, 0]),
            rtol=0,
            atol=0,
        )

    def test_execute_rejects_finished_request_reuse_before_state_changes(self):
        row = {"request_id": 1, "token_ids": [10], "num_computed_tokens": 4}
        for scheduled, preempted in (([row], []), ([], [1])):
            with self.subTest(preempted=preempted):
                worker = _worker()
                request = SimpleNamespace(
                    cache_slot=0, state_position=4, history=[10] * 5
                )
                worker.requests = {1: request}
                worker.slots = {1}
                with self.assertRaisesRegex(ValueError, "overlap"):
                    worker.execute(
                        _wire_message(
                            worker,
                            {
                                "rpc_id": 7,
                                "finished_request_ids": [1],
                                "preempted_request_ids": preempted,
                                "scheduled": scheduled,
                            },
                        )
                    )
                self.assertIs(worker.requests[1], request)
                self.assertEqual(request.cache_slot, 0)
                self.assertEqual(worker.slots, {1})

    def test_execute_rejects_unknown_lifecycle_request(self):
        worker = _worker()
        worker.requests = {}
        with self.assertRaisesRegex(ValueError, "unknown request"):
            worker.execute(
                _wire_message(
                    worker,
                    {
                        "finished_request_ids": [1],
                        "preempted_request_ids": [],
                        "scheduled": [],
                        "rpc_id": 1,
                    },
                )
            )

    def test_restore_failure_releases_the_new_request_slot(self):
        worker = _worker()
        request = SimpleNamespace(cache_slot=None, history=[10])
        worker.requests = {1: request}
        worker.slots = {0}
        worker.cache = SimpleNamespace(
            restore=Mock(side_effect=ValueError("bad state")),
            clear_slot=Mock(),
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
        worker = _worker()
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
        worker = _worker()
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

    def test_target_keeps_long_prefill_in_one_native_call(self):
        worker = _worker()
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
        worker = _worker()
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
        worker = _worker()
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
        worker = _worker()
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
