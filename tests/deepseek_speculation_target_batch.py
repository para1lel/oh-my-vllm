import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts
from oh_my_vllm.models.deepseek_v41.worker_limits import (
    DEFAULT_PREFILL_ROWS,
    MAX_PREFILL_ROWS,
)

from tests.deepseek_speculation_fixtures import DeepSeekSpeculationFixture, _worker


class DeepSeekSpeculationTargetBatch(DeepSeekSpeculationFixture, unittest.TestCase):
    def test_target_graph_requires_vocab_logits_and_one_token(self):
        worker = _worker()
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.decode_graph_enabled = True
        worker.model = Mock(return_value=(None, None, None))
        with patch.object(worker, "_decode_graph_step", return_value=None) as graph:
            worker._target([1], 5, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target([1], 0, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target([1], 5, 3, compute_logits=False)
            self.assertEqual(graph.call_count, 1)
            worker._target([1, 2], 5, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target_batch_group([[1]], 5, torch.tensor([3]))
            self.assertEqual(graph.call_count, 2)
            worker._target_batch_group(
                [[1]], 5, torch.tensor([3]), compute_logits=False
            )
            self.assertEqual(graph.call_count, 2)

    def test_target_batch_keeps_total_rows_in_one_native_call(self):
        worker = _worker()
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
        worker = _worker()
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        worker.args = SimpleNamespace(window_size=200)
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
        worker = _worker()
        worker.device = torch.device("cpu")
        worker.prefill_rows = DEFAULT_PREFILL_ROWS

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

    def test_target_batch_uses_wide_prefill_subbatches(self):
        worker = _worker()
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        calls = []

        def model(inputs, **kwargs):
            self.assertLessEqual(inputs.size(0), 16)
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            calls.append((inputs[:, 0].tolist(), kwargs["cache_slots_cpu"]))
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 100 for index in range(32)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(32),
        )

        self.assertEqual(worker.model.call_count, 2)
        self.assertEqual(
            calls,
            [
                (list(range(16)), list(range(16))),
                (list(range(16, 32)), list(range(16, 32))),
            ],
        )

    def test_mtp_seed_keeps_only_the_target_window(self):
        worker = _worker()
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
