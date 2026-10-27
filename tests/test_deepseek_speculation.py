import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from oh_my_vllm.models.deepseek_v41.inference import Worker
from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts


class DeepSeekSpeculation(unittest.TestCase):
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

    def test_target_splits_prefill_rows_at_native_limit(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.model = Mock(
            side_effect=[
                ("first", "first logits", "first hidden"),
                ("last", "last logits", "last hidden"),
            ]
        )

        result = worker._target([1] * 385, 7, 3)

        self.assertEqual(result, ("last", "last logits", "last hidden"))
        self.assertEqual(worker.model.call_count, 2)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7, 391],
        )

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
