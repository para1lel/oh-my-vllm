import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.inference import (
    DEFAULT_PREFILL_ROWS,
    MAX_PREFILL_ROWS,
    Worker,
    prefill_row_limit,
)
from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts


class DeepSeekSpeculation(unittest.TestCase):
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
        worker._target = Mock(
            side_effect=[
                (None, logits[0], torch.ones(1)),
                (None, logits[1], None),
                (None, logits[2], None),
            ]
        )

        result = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 11, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(result["token_ids"], [5, 6, 7])
        self.assertEqual(result["num_accepted_draft_tokens"], 2)

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
        hidden = [torch.ones(1, 1, 3), torch.full((1, 1, 3), 2)]
        worker.requests = {1: request}
        worker.dspark = True
        worker.device = torch.device("cpu")
        worker._slot = lambda _: 0
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        worker._proposal = Mock(return_value=[])
        worker.trace = SimpleNamespace(record=Mock())
        worker._target = Mock(
            side_effect=[
                (
                    None,
                    torch.nn.functional.one_hot(torch.tensor([[6]]), 8).float(),
                    hidden[0],
                ),
                (
                    None,
                    torch.nn.functional.one_hot(torch.tensor([[6]]), 8).float(),
                    hidden[1],
                ),
            ]
        )

        worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(worker.model.forward_spec_seed.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["start_pos"]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [0, 2],
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
        result = worker._target([1] * 8193, 7, 3, compute_logits=False)

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
            self.assertLessEqual(inputs.size(0) * inputs.size(1), 8192)
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
            self.assertLessEqual(inputs.size(0) * inputs.size(1), 8192)
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
