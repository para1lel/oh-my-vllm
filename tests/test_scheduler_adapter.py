"""Exercise current installed vLLM dataclasses without a model or CUDA kernels."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from oh_my_vllm.worker.model_runner import (
    OhMyVllmWorker,
    ScheduledRequest,
    SchedulerAdapter,
    SchedulerOutput,
)
from vllm import SamplingParams


class AdapterTest(unittest.TestCase):
    def test_finished_notification_clears_all_worker_request_state(self):
        worker = object.__new__(OhMyVllmWorker)
        worker.config = object()
        worker._num_speculative_tokens = 0
        worker._worker = Mock()
        worker._worker.execute_model.return_value = None
        worker.adapter = SchedulerAdapter(["fa", "mamba"])
        worker.prompt_token_ids_map = {1: [1, 2]}
        worker.sampling_params_map = {1: SamplingParams(temperature=0)}
        worker.adapter.convert(
            SchedulerOutput(
                [ScheduledRequest(1, [1, 2], 0, [1], [2])], num_batched_tokens=2
            ),
            worker.prompt_token_ids_map,
            worker.sampling_params_map,
        )
        worker.adapter.preempted.add(1)
        with patch("vllm.config.set_current_vllm_config", return_value=nullcontext()):
            result = worker.execute_model(SchedulerOutput(finished_request_ids=[1]))
        self.assertEqual(result.outputs, [])
        notification = worker._worker.execute_model.call_args.args[0]
        self.assertEqual(notification.finished_req_ids, {"1"})
        worker._worker.sample_tokens.assert_not_called()
        self.assertFalse(worker.adapter.blocks)
        self.assertFalse(worker.adapter.output_counts)
        self.assertFalse(worker.adapter.preempted)
        self.assertFalse(worker.prompt_token_ids_map)
        self.assertFalse(worker.sampling_params_map)

    def test_prefix_hit_is_new_and_running_blocks_are_deltas(self):
        adapter = SchedulerAdapter(["fa", "mamba", "mamba"])
        prompts = {1: list(range(1570))}
        sampling = {1: SamplingParams(temperature=0)}
        request = ScheduledRequest(1, [1568, 1569], 1568, [1, 2, 3], [0, 4, 5])
        first = adapter.convert(
            SchedulerOutput([request], num_batched_tokens=2), prompts, sampling
        )
        self.assertEqual(first.scheduled_new_reqs[0].num_computed_tokens, 1568)
        self.assertEqual(len(first.scheduled_new_reqs[0].block_ids), 3)
        physical = first.scheduled_new_reqs[0].block_ids
        self.assertEqual(physical, ([2, 4, 6], [0, 8, 10], [0, 9, 11]))
        self.assertTrue(set(physical[1][1:]).isdisjoint(physical[2][1:]))
        adapter.output_counts[1] = 1
        request = ScheduledRequest(1, [42], 1570, [1, 2, 3], [0, 0, 5])
        second = adapter.convert(
            SchedulerOutput([request], num_batched_tokens=1), prompts, sampling
        )
        self.assertEqual(second.scheduled_cached_reqs.new_block_ids, [([], [], [])])
        self.assertEqual(second.scheduled_cached_reqs.num_output_tokens, [1])
        adapter.convert(SchedulerOutput(finished_request_ids=[1]), prompts, sampling)
        self.assertFalse(adapter.blocks)
        self.assertFalse(adapter.output_counts)

    def test_capacity_rounding_and_null_mapping(self):
        adapter = SchedulerAdapter(["mamba", "mamba", "mamba", "fa"])
        physical_capacity = 128
        logical_capacity = physical_capacity // adapter.stride
        request = ScheduledRequest(
            1, [7], 0, [logical_capacity - 2], [0, logical_capacity - 1]
        )
        out = adapter.convert(
            SchedulerOutput([request], num_batched_tokens=1),
            {1: [7]},
            {1: SamplingParams(temperature=0)},
        )
        blocks = out.scheduled_new_reqs[0].block_ids
        self.assertEqual([group[0] for group in blocks[:3]], [0, 0, 0])
        used = [block for group in blocks for block in group if block]
        self.assertEqual(len(used), len(set(used)))
        self.assertLess(max(used), physical_capacity)

    def test_preempted_request_replaces_tables(self):
        adapter = SchedulerAdapter(["fa", "mamba"])
        prompts = {1: [1, 2]}
        sampling = {1: SamplingParams(temperature=0)}
        first = ScheduledRequest(1, [1, 2], 0, [1], [2])
        adapter.convert(
            SchedulerOutput([first], num_batched_tokens=2), prompts, sampling
        )
        resumed = ScheduledRequest(1, [1, 2], 0, [3], [4])
        out = adapter.convert(
            SchedulerOutput([resumed], preempted_request_ids=[1], num_batched_tokens=2),
            prompts,
            sampling,
        )
        self.assertEqual(out.scheduled_cached_reqs.resumed_req_ids, {"1"})
        self.assertEqual(out.scheduled_cached_reqs.new_block_ids, [([3], [4])])


if __name__ == "__main__":
    unittest.main()
