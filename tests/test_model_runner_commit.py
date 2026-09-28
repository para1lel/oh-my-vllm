"""Request-local step commits use the real runner path without a GPU forward pass."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from oh_my_vllm.worker.model_runner import OhMyVllmWorker
from oh_my_vllm.worker.mtp import MTP
from oh_my_vllm.worker.protocol import ScheduledRequest, SchedulerOutput


class CommitTests(unittest.TestCase):
    def test_reported_cache_capacities_come_from_allocated_tensors(self):
        worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
        worker.model = SimpleNamespace(
            kinds=["full_attention", "full_attention", "gated_delta_net"]
        )
        worker.caches = [
            torch.empty(7, 1),
            torch.empty(7, 1),
            (torch.empty(3, 1), torch.empty(3, 1)),
        ]
        worker.logical_num_blocks = 99
        worker.mamba_blocks = 99
        self.assertEqual(worker.cache_capacities(), (7, 3))
        worker.caches[1] = torch.empty(6, 1)
        with self.assertRaisesRegex(RuntimeError, "inconsistent allocated"):
            worker.cache_capacities()

    def make_worker(self):
        worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
        worker.histories = {1: [3], 2: [4]}
        worker.samplers = {1: Mock(), 2: Mock()}
        worker.sources = {}
        worker.computed = {}
        worker.mtp = None
        generations = {}
        for rid in (1, 2):
            generation = Mock(finished=None, reasoning_tokens=0)
            generation.consume.side_effect = lambda tokens, _tokenizer: (tokens, "")
            generations[rid] = generation
        worker.serving = SimpleNamespace(generations=generations, tokenizer=object())
        return worker

    def test_failed_draft_rows_do_not_shift_next_request(self):
        worker = self.make_worker()
        failed = SimpleNamespace(
            request=SimpleNamespace(request_id=1, num_computed_tokens=10),
            sample_indices=[0, 1],
            drafts=[5],
            commit=Mock(),
        )
        healthy = SimpleNamespace(
            request=SimpleNamespace(request_id=2, num_computed_tokens=20),
            sample_indices=[0, 1],
            drafts=[7],
            commit=Mock(return_value=(2, 11, [(10, 12)])),
        )
        rows = [[0, 0], [0, 1], [7, 1], [9, 1]]
        result = worker._commit_plans(
            [failed, healthy], [0, 2, 4], torch.empty(4, 1), rows, {}
        )
        outputs, copies, plans, starts, counts, successes = result
        self.assertEqual(outputs[0].request_id, 1)
        self.assertIn("no valid token", outputs[0].error)
        self.assertEqual(outputs[1].token_ids, [7, 9])
        self.assertEqual(outputs[1].num_accepted_draft_tokens, 1)
        self.assertEqual((plans, starts, counts), ([healthy], [2], [2]))
        self.assertEqual(successes, [outputs[1]])
        self.assertEqual(copies, [(10, 12)])
        failed.commit.assert_not_called()
        healthy.commit.assert_called_once_with(2)
        self.assertNotIn(1, worker.histories)
        self.assertNotIn(1, worker.serving.generations)
        self.assertEqual(worker.histories[2], [4, 7, 9])

    def test_cuda_out_of_memory_remains_fatal(self):
        worker = self.make_worker()
        worker.samplers[1].sample.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")
        plan = SimpleNamespace(
            request=SimpleNamespace(request_id=1, num_computed_tokens=0),
            sample_indices=[0],
            drafts=[],
        )
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            worker._commit_plans([plan], [0, 1], torch.empty(1, 1), None, {})

    def test_grammar_failure_is_isolated_before_sampling(self):
        worker = self.make_worker()
        plans = [
            SimpleNamespace(
                request=SimpleNamespace(request_id=rid),
                writes=[1],
                drafts=[],
                sample_indices=[0],
            )
            for rid in (1, 2)
        ]

        def masks(scheduled):
            rid = next(iter(scheduled.num_scheduled_tokens))
            if rid == "1":
                raise ValueError("invalid grammar state")
            return SimpleNamespace(grammar_bitmask=np.array([[1]], dtype=np.int32))

        worker.serving.masks = masks
        masks, errors = worker._build_masks(plans)
        self.assertEqual(errors, {1: "invalid grammar state"})
        np.testing.assert_array_equal(masks[2], [[1]])

    def test_missing_registration_returns_request_error(self):
        worker = self.make_worker()
        scheduled = SchedulerOutput(
            scheduled=[ScheduledRequest(3, [2], 0, [], [])],
            num_batched_tokens=1,
        )
        output = worker.execute_model(scheduled)
        self.assertEqual(len(output.outputs), 1)
        self.assertEqual(output.outputs[0].request_id, 3)
        self.assertEqual(output.outputs[0].error, "request registration failed")

    def test_bad_mtp_state_aborts_only_its_request(self):
        worker = self.make_worker()
        worker.mtp = Mock()
        worker.mtp.validate_state.side_effect = [ValueError("bad MTP state"), None]
        plans = [
            SimpleNamespace(
                request=SimpleNamespace(request_id=rid, num_computed_tokens=10),
                sample_indices=[0],
                drafts=[],
                commit=Mock(return_value=(1, 11, [])),
            )
            for rid in (1, 2)
        ]
        rows = [[5, 1], [6, 1]]
        outputs, _, kept, starts, counts, _ = worker._commit_plans(
            plans, [0, 1, 2], torch.empty(2, 1), rows, {}
        )
        self.assertEqual(outputs[0].error, "bad MTP state")
        self.assertEqual(outputs[1].token_ids, [6])
        self.assertEqual((kept, starts, counts), ([plans[1]], [1], [1]))

    def test_mtp_state_validation_is_request_local(self):
        mtp = MTP.__new__(MTP)
        mtp.next_position = {1: 99}
        plan = SimpleNamespace(
            request=SimpleNamespace(request_id=1, num_computed_tokens=10)
        )
        with self.assertRaisesRegex(ValueError, "MTP cache position"):
            mtp.validate_state(plan, 1)


if __name__ == "__main__":
    unittest.main()
