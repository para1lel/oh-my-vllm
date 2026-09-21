"""Exercise current installed vLLM dataclasses without a model or CUDA kernels."""

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

from oh_my_vllm.worker.model_runner import (
    OhMyVllmWorker,
    ScheduledRequest,
    SchedulerAdapter,
    SchedulerOutput,
)
from vllm import SamplingParams


class AdapterTest(unittest.TestCase):
    def test_new_allocations_zero_full_physical_stride_only(self):
        adapter = SchedulerAdapter(["mamba", "mamba", "mamba", "fa"])
        history = list(range(1570))
        request = ScheduledRequest(
            1,
            history[1568:],
            1568,
            [1, 2, 3],
            [0, 4, 5],
            history,
            new_block_ids_to_zero=[3, 5],
        )
        result = adapter.convert(
            SchedulerOutput([request], num_batched_tokens=2),
            {1: history},
            {1: SamplingParams(temperature=0)},
        )
        self.assertEqual(result.new_block_ids_to_zero, [9, 10, 11, 15, 16, 17])
        self.assertTrue(
            set(result.new_block_ids_to_zero).isdisjoint([3, 6, 12, 13, 14])
        )

    def test_warmup_cache_clear_preserves_captured_storage(self):
        import torch
        from oh_my_vllm.worker.v2_runner import clear_warmup_cache

        attention = torch.full((2, 3, 4), float("nan"))
        state = torch.full((3, 4), float("nan"))
        convolution = torch.ones(2, 3)
        tensors = [attention, state, convolution]
        addresses = [tensor.data_ptr() for tensor in tensors]
        clear_warmup_cache(SimpleNamespace(kv_caches=[attention, [state, convolution]]))
        for tensor, address in zip(tensors, addresses, strict=True):
            self.assertEqual(tensor.data_ptr(), address)
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)

    def test_v2_transfers_real_drafts_without_changing_sampling_flags(self):
        from oh_my_vllm.worker.v2_runner import RustDraftTokensHandler
        from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

        @dataclass
        class Batch:
            req_ids: list[str]
            has_structured_output_reqs: bool

        handler = object.__new__(RustDraftTokensHandler)
        drafts = object()
        for constrained in (False, True):
            batch = Batch(["1", "2"], constrained)
            with patch.object(DraftTokensHandler, "set_draft_tokens") as transfer:
                handler.set_draft_tokens(batch, drafts)
            copied_batch, copied_drafts = transfer.call_args.args
            self.assertTrue(copied_batch.has_structured_output_reqs)
            self.assertEqual(copied_batch.req_ids, ["1", "2"])
            self.assertIs(copied_drafts, drafts)
            self.assertEqual(batch.has_structured_output_reqs, constrained)

    def test_finished_notification_clears_all_worker_request_state(self):
        worker = object.__new__(OhMyVllmWorker)
        worker.config = object()
        worker.serving = None
        worker._num_speculative_tokens = 0
        worker._worker = Mock()
        worker._worker.execute_model.return_value = None
        worker.adapter = SchedulerAdapter(["fa", "mamba"])
        worker.prompt_token_ids_map = {1: [1, 2]}
        worker.sampling_params_map = {1: SamplingParams(temperature=0)}
        worker.adapter.convert(
            SchedulerOutput(
                [ScheduledRequest(1, [1, 2], 0, [1], [2], [1, 2])], num_batched_tokens=2
            ),
            worker.prompt_token_ids_map,
            worker.sampling_params_map,
        )
        with patch("vllm.config.set_current_vllm_config", return_value=nullcontext()):
            result = worker.execute_model(SchedulerOutput(finished_request_ids=[1]))
        self.assertEqual(result.outputs, [])
        notification = worker._worker.execute_model.call_args.args[0]
        self.assertEqual(notification.finished_req_ids, {"1"})
        worker._worker.sample_tokens.assert_not_called()
        self.assertFalse(worker.adapter.blocks)
        self.assertFalse(worker.adapter.output_counts)
        self.assertFalse(worker.prompt_token_ids_map)
        self.assertFalse(worker.sampling_params_map)

    def test_prefix_hit_is_new_and_running_blocks_are_deltas(self):
        adapter = SchedulerAdapter(["fa", "mamba", "mamba"])
        prompts = {1: list(range(1570))}
        sampling = {1: SamplingParams(temperature=0)}
        request = ScheduledRequest(
            1, [1568, 1569], 1568, [1, 2, 3], [0, 4, 5], prompts[1]
        )
        first = adapter.convert(
            SchedulerOutput([request], num_batched_tokens=2), prompts, sampling
        )
        self.assertEqual(first.scheduled_new_reqs[0].num_computed_tokens, 1568)
        self.assertEqual(first.scheduled_new_reqs[0].prefill_token_ids, prompts[1])
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
            1, [7], 0, [logical_capacity - 2], [0, logical_capacity - 1], [7]
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
        first = ScheduledRequest(1, [1, 2], 0, [1], [2], [1, 2])
        adapter.convert(
            SchedulerOutput([first], num_batched_tokens=2), prompts, sampling
        )
        adapter.output_counts[1] = 2
        resumed = ScheduledRequest(1, [1, 2], 0, [3], [4], [1, 2, 9, 10])
        out = adapter.convert(
            SchedulerOutput([resumed], preempted_request_ids=[1], num_batched_tokens=2),
            prompts,
            sampling,
        )
        self.assertEqual(out.scheduled_cached_reqs.req_ids, [])
        self.assertEqual(out.scheduled_new_reqs[0].block_ids, ([3], [4]))
        self.assertEqual(out.scheduled_new_reqs[0].prefill_token_ids, [1, 2, 9, 10])
        self.assertEqual(adapter.output_counts[1], 2)
        self.assertFalse(out.scheduled_spec_decode_tokens)
        # A later decode includes the last accepted token and two unverified drafts.
        adapter.output_counts[1] += 1
        decoded = adapter.convert(
            SchedulerOutput(
                [ScheduledRequest(1, [11, 12, 13], 4, [3], [4])], num_batched_tokens=3
            ),
            prompts,
            sampling,
        )
        self.assertEqual(decoded.scheduled_spec_decode_tokens, {"1": [12, 13]})

    def test_delayed_resume_then_preemption_and_cancellation(self):
        adapter = SchedulerAdapter(["fa", "mamba"])
        prompts = {1: [1, 2]}
        sampling = {1: SamplingParams(temperature=0)}
        adapter.convert(
            SchedulerOutput(
                [ScheduledRequest(1, [1, 2], 0, [1], [2], [1, 2])],
                num_batched_tokens=2,
            ),
            prompts,
            sampling,
        )
        adapter.output_counts[1] = 2
        adapter.convert(SchedulerOutput(preempted_request_ids=[1]), prompts, sampling)
        self.assertNotIn(1, adapter.blocks)
        adapter.convert(SchedulerOutput(), prompts, sampling)
        resumed = adapter.convert(
            SchedulerOutput(
                [ScheduledRequest(1, [9, 10], 2, [3], [4], [1, 2, 9, 10])],
                num_batched_tokens=2,
            ),
            prompts,
            sampling,
        )
        self.assertEqual(resumed.scheduled_new_reqs[0].num_computed_tokens, 2)
        self.assertEqual(resumed.scheduled_new_reqs[0].prefill_token_ids, [1, 2, 9, 10])
        self.assertFalse(resumed.scheduled_spec_decode_tokens)
        self.assertEqual(adapter.output_counts[1], 2)
        adapter.convert(SchedulerOutput(preempted_request_ids=[1]), prompts, sampling)
        canceled = adapter.convert(
            SchedulerOutput(finished_request_ids=[1]), prompts, sampling
        )
        self.assertEqual(canceled.finished_req_ids, {"1"})
        self.assertFalse(adapter.blocks)
        self.assertFalse(adapter.output_counts)

    def test_missing_or_wrong_recovery_history_fails(self):
        for history in [None, [99, 2], [1]]:
            adapter = SchedulerAdapter(["fa", "mamba"])
            with self.assertRaisesRegex(ValueError, "complete accepted history"):
                adapter.convert(
                    SchedulerOutput(
                        [ScheduledRequest(1, [1], 0, [1], [2], history)],
                        num_batched_tokens=1,
                    ),
                    {1: [1, 2]},
                    {1: SamplingParams(temperature=0)},
                )


if __name__ == "__main__":
    unittest.main()
