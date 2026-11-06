import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from oh_my_vllm.models.qwen38.cache import page_valid_ends
from oh_my_vllm.models.qwen38.cache_budget import capacity_plan
from oh_my_vllm.models.qwen38.cache_wire import Endpoint, sequence_plan, write_ranges

from tests.qwen_radix_physical_fixtures import (
    cache,
    config,
    context_batch,
    execute_message,
    reference,
    request,
    row,
    tiny_cache,
    worker,
)


class PhysicalCacheTests(unittest.TestCase):
    def test_budget_counts_null_private_scales_and_workspace(self):
        layout, plan = capacity_plan(config(), byte_limit=124200480768)
        self.assertEqual((layout.checkpoints, layout.pages), (279, 17856))
        self.assertEqual(layout.slots, 312)
        self.assertEqual(layout.table_width, 2048)
        self.assertEqual(layout.page_bytes, 4259840)
        self.assertEqual(plan["planned_bytes"], 124101705728)
        reduced, _ = capacity_plan(
            config(), byte_limit=124200480768, workspace_bytes=100000000
        )
        self.assertEqual((reduced.checkpoints, reduced.pages), (278, 17792))
        self.assertFalse(plan["capacity_verified"])
        self.assertEqual(layout.pool_bytes["private_checkpoints"], 24 * 153954304)

    def test_ratio_rejects_conflicting_and_insufficient_budget(self):
        with self.assertRaises(ValueError):
            capacity_plan(config(num_gpu_blocks=65), byte_limit=1 << 40)
        with self.assertRaises(ValueError):
            capacity_plan(
                config(mamba_blocks=4, num_gpu_blocks=192), byte_limit=1 << 40
            )
        with self.assertRaises(ValueError):
            capacity_plan(config(), byte_limit=100)

    def test_dense_pool_has_two_independent_kv_arrays_per_layer(self):
        value = cache()
        codes, scales = value.fa[0]
        self.assertEqual((codes.shape, codes.dtype), ((257, 2, 128, 4, 256), "uint8"))
        self.assertEqual((scales.shape, scales.dtype), ((257, 2, 128, 4), "float32"))
        self.assertEqual(value.hidden.dtype, "bfloat16")
        self.assertEqual(value.state[0].dtype, "float32")
        self.assertEqual(value.convolution[0].dtype, "bfloat16")

    def test_partial_cow_passes_both_slabs_to_one_native_copy(self):
        value = tiny_cache()
        with patch("oh_my_vllm.models.qwen38.cache.copy_fp8_prefix") as copy_page:
            value.copy_prefix(
                [
                    {
                        "source_slot": 1,
                        "source_version": 1,
                        "destination_slot": 2,
                        "destination_version": 2,
                        "valid_tokens": 127,
                    }
                ]
            )
        self.assertEqual(copy_page.call_count, 1)
        codes, scales, source, destination, valid = copy_page.call_args.args
        self.assertIs(codes, value.fa[0][0])
        self.assertIs(scales, value.fa[0][1])
        self.assertEqual((source, destination, valid), (1, 2, 127))

    def test_invalid_cow_list_does_not_partially_copy(self):
        value = tiny_cache()
        valid = {
            "source_slot": 1,
            "source_version": 1,
            "destination_slot": 2,
            "destination_version": 2,
            "valid_tokens": 17,
        }
        with self.assertRaises(ValueError):
            value.copy_prefix([valid, dict(valid, valid_tokens=128)])
        self.assertEqual(value.device.copies, [])

    def test_interior_hole_keeps_later_reuse_and_independent_hits(self):
        pages = [
            {
                "page_index": index,
                "slot": index + 1,
                "version": 2,
                "valid_tokens": 128,
                "access": "write" if index == 1 else "read",
            }
            for index in range(3)
        ]
        item = row(
            end=384,
            deepest=300,
            contiguous=128,
            pages=pages,
            spans=[
                {"start": 0, "end": 128, "mode": "reuse"},
                {"start": 128, "end": 256, "mode": "write"},
                {"start": 256, "end": 384, "mode": "reuse"},
            ],
        )
        plan = sequence_plan(item, cache().layout)
        self.assertEqual(write_ranges(plan.write_mask), ((128, 256),))
        self.assertEqual(plan.read.end, 0)
        resumed = row(
            128,
            384,
            deepest=300,
            contiguous=128,
            resume=reference(128, 3),
            pages=pages,
            spans=item["model_state"]["kv_spans"][1:],
        )
        plan = sequence_plan(resumed, cache().layout)
        self.assertEqual(plan.read.end, 128)
        self.assertEqual(
            page_valid_ends(plan, 200),
            [
                {"page_index": 0, "slot": 1, "version": 2, "valid_end": 128},
                {"page_index": 1, "slot": 2, "version": 2, "valid_end": 256},
                {"page_index": 2, "slot": 3, "version": 2, "valid_end": 384},
            ],
        )

    def test_verify_retains_every_endpoint_for_accepted_restore(self):
        value, owner = cache(), request()
        item = row(
            end=3,
            sampling="verify",
            exports=[
                {**reference(end, slot), "role": "private_endpoint"}
                for end, slot in ((1, 5), (2, 6), (3, 7))
            ],
        )
        item["sampling"]["drafts"] = 2
        batch = value.batch([item], [owner], True)
        value.complete_exports(batch)
        value.commit(owner, batch, 0, 1)
        self.assertEqual(
            (owner.evaluated_end, owner.computed_end, owner.state_position), (3, 2, 2)
        )
        final = row(
            2,
            2,
            purpose="finalize",
            action="export_only",
            exports=[{**reference(2, 3), "role": "candidate_tail"}],
        )
        value.export_endpoint(final, owner)
        self.assertEqual(
            (owner.state_position, owner.state_slot, owner.kv_valid_end), (2, 3, 2)
        )
        self.assertEqual(len(value.device.copies), 97)

    def test_endpoint_copy_rejects_overwritten_slot_version(self):
        value = cache()
        value.versions[5] = 2
        with self.assertRaises(ValueError):
            value.copy_endpoint(Endpoint(3, 5, 1), Endpoint(3, 2, 1))
        self.assertEqual(value.device.copies, [])

    def test_zero_row_work_validates_without_fake_forward(self):
        item = row(3, 3, purpose="finalize", action="export_only")
        plan = sequence_plan(item, cache().layout)
        self.assertEqual(plan.write_mask, ())
        bad = copy.deepcopy(item)
        bad["sampling"] = {"kind": "target"}
        with self.assertRaises(ValueError):
            sequence_plan(bad, cache().layout)
        bad = copy.deepcopy(item)
        bad["action"] = "head_only"
        with self.assertRaises(ValueError):
            sequence_plan(bad, cache().layout)

    def test_finalize_forward_is_compute_only_and_joins_before_ack(self):
        owner = request()
        value = worker(owner)
        item = row(purpose="finalize")
        result = value.execute(execute_message(item))["outputs"][0]
        value.sampler.assert_not_called()
        owner.append.assert_not_called()
        self.assertEqual(
            (owner.random_offset, owner.proposal_offset, owner.accept_offset),
            (11, 13, 17),
        )
        self.assertEqual(
            (
                result["token_ids"],
                result["text"],
                result["finish_reason"],
                result["reasoning_tokens"],
            ),
            ([], "", None, 0),
        )
        self.assertEqual((result["computed_end"], result["accepted_end"]), (3, 3))
        self.assertEqual(value.device.events[-1], "joined")
        with self.assertRaises(ValueError):
            value.execute(execute_message(item))
        self.assertEqual(value.model.forward.call_count, 1)

    def test_zero_row_finalize_restores_exact_saved_endpoint(self):
        owner = request()
        value = worker(owner)
        value.execute(execute_message(row()))
        value.model.forward.reset_mock()
        owner.state_position = owner.computed_end = 5
        item = row(3, 3, purpose="finalize", action="export_only")
        message = execute_message(item)
        message["step_id"] = 2
        result = value.execute(message)["outputs"][0]
        value.model.forward.assert_not_called()
        value.sampler.assert_not_called()
        self.assertEqual((owner.state_position, owner.state_slot), (3, 2))
        self.assertEqual(result["computed_end"], 3)
        self.assertEqual(len(value.device.copies), 97)
        self.assertEqual(value.device.events[-1], "joined")

    def test_exact_head_uses_saved_hidden_without_forward(self):
        owner = request(history=[0, 1, 2], finish=None)
        owner.append = Mock(
            side_effect=lambda tokens: (owner.history.extend(tokens) or tokens, "x")
        )
        value = worker(owner)
        value.execute(execute_message(row()))
        value.model.forward.reset_mock()
        value.sampler = Mock(
            return_value=SimpleNamespace(select_hidden=Mock(return_value=([42], None)))
        )
        item = row(
            3,
            3,
            action="head_only",
            sampling="target",
            deepest=3,
            contiguous=3,
            resume=reference(3, 5),
            exports=[{**reference(3, 6), "role": "private_endpoint"}],
        )
        item["model_state"]["endpoint_source"] = {
            "kind": "saved_hidden",
            **reference(3, 5),
        }
        message = execute_message(item)
        message["step_id"] = 2
        result = value.execute(message)["outputs"][0]
        value.model.forward.assert_not_called()
        self.assertEqual(result["token_ids"], [42])
        self.assertEqual(result["computed_end"], 3)
        self.assertEqual(result["accepted_end"], 4)
        self.assertEqual(value.sampler.call_count, 1)

    def test_draft_rollback_restores_codes_and_scales(self):
        from oh_my_vllm.models.qwen38.draft_pipeline import DraftPipeline

        base = tiny_cache()
        value = DraftPipeline.__new__(DraftPipeline)
        value.device = base.device
        value.cache_rows = (
            np.zeros((10, 1024), np.uint8),
            np.zeros((10, 4), np.float32),
        )
        value.saved = [
            (np.full((2, 1024), 29, np.uint8), np.full((2, 4), 0.125, np.float32))
        ]
        value.indices, value.snapshots = [[2, 7]], {1}
        value.restore(1)
        np.testing.assert_array_equal(value.cache_rows[0][[2, 7]], 29)
        np.testing.assert_array_equal(value.cache_rows[1][[2, 7]], 0.125)
        np.testing.assert_array_equal(value.cache_rows[0][3], 0)
        np.testing.assert_array_equal(value.cache_rows[1][3], 0)
        self.assertEqual(value.snapshots, set())

    def test_unknown_and_relabelled_checkpoint_sources_fail_before_copy(self):
        value = worker(request())
        item = row(3, 3, purpose="finalize", action="export_only")
        item["model_state"]["endpoint_source"] = {
            "kind": "saved_hidden",
            **reference(3, 1),
        }
        with self.assertRaisesRegex(ValueError, "completed matching endpoint"):
            value.execute(execute_message(item))
        self.assertEqual(value.device.copies, [])
        value.execute(execute_message(row()))
        forged = row(99, 99, purpose="finalize", action="export_only")
        forged["model_state"]["endpoint_source"] = {
            "kind": "saved_hidden",
            **reference(99, 5),
        }
        message = execute_message(forged)
        message["step_id"] = 2
        with self.assertRaisesRegex(ValueError, "completed matching endpoint"):
            value.execute(message)
        self.assertEqual(value.device.copies, [])

    def test_slot_reuse_rejects_old_end_and_failed_copy_stays_unpublished(self):
        value, owner = cache(), request()
        batch = value.batch([row()], [owner], False)
        value.complete_exports(batch)
        with (
            patch.object(value.device, "copy", side_effect=RuntimeError("copy failed")),
            self.assertRaises(RuntimeError),
        ):
            value.copy_endpoint(Endpoint(3, 5, 1), Endpoint(3, 2, 1))
        self.assertNotIn(2, value.checkpoints)
        item = row(end=4, exports=[{**reference(4, 5, 2), "role": "private_endpoint"}])
        item["model_state"]["private_checkpoints"][0]["version"] = 2
        batch = value.batch([item], [owner], False)
        value.complete_exports(batch)
        with self.assertRaises(ValueError):
            value.copy_endpoint(Endpoint(3, 5, 1), Endpoint(3, 2, 1))
        with self.assertRaises(ValueError):
            value.copy_endpoint(Endpoint(3, 5, 2), Endpoint(3, 2, 1))

    def test_stale_or_unwritten_page_copy_is_rejected(self):
        value = tiny_cache()
        item = {
            "source_slot": 1,
            "source_version": 2,
            "destination_slot": 2,
            "destination_version": 1,
            "valid_tokens": 127,
        }
        with patch("oh_my_vllm.models.qwen38.cache.copy_fp8_prefix") as native:
            with self.assertRaises(ValueError):
                value.copy_prefix([item])
            with self.assertRaises(ValueError):
                value.copy_prefix([dict(item, source_slot=3, source_version=1)])
        native.assert_not_called()

    def test_mtp_full_config_budget_reserves_shifted_context_rows(self):
        from oh_my_vllm.models.qwen38.cache_budget import target_token_limit
        from oh_my_vllm.models.qwen38.mtp_context import context_rows
        from oh_my_vllm.operators.mtp_fp8.plan import context_plan

        options = config(num_speculative_tokens=4)
        self.assertEqual(options["max_num_batched_tokens"], 32768)
        self.assertEqual(target_token_limit(options), 32764)
        for lengths, expected in (((32764,), 32765), ((8191,) * 4, 32768)):
            batch = context_batch(lengths)
            prepared = context_rows(batch, dict.fromkeys(range(len(lengths)), (1,)))
            arrays = prepared[3]
            self.assertEqual(len(arrays[0]), expected)
            self.assertEqual(arrays[1][0], -1)
            self.assertEqual(arrays[8][-1], expected - 1)
            plan = context_plan(
                rows=expected,
                batch=len(lengths),
                max_queries=int(max(np.diff(arrays[6]))),
            )
            self.assertLessEqual(plan.rows, 32768)
        value = worker(request())
        value.config, value.speculation = options, object()
        pieces = list(value.batches([{"token_ids": [0] * 16384}] * 2))
        self.assertEqual(
            [sum(len(item["token_ids"]) for item in piece) for piece in pieces],
            [16384, 16384],
        )


if __name__ == "__main__":
    unittest.main()
