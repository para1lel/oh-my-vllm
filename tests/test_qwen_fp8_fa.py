import unittest
from unittest.mock import patch

from oh_my_vllm.operators.fa_fp8.api import CacheView
from oh_my_vllm.operators.fa_fp8.copy import copy_prefix
from oh_my_vllm.operators.fa_fp8.plan import make_plan
from oh_my_vllm.operators.fa_fp8.workspace import prepare

from tests.qwen_fp8_fa_fixtures import Device, Tensor, bindings, native_module


class QwenFp8Fa(unittest.TestCase):
    def plan(self, **overrides):
        arguments = dict(
            rows=4, batch=2, table_width=2048, max_queries=2, max_context=262144
        )
        return make_plan(**(arguments | overrides))

    def test_128_page_plan_retains_full_context_bound(self):
        self.assertEqual(self.plan().max_context, 262144)
        self.assertEqual(self.plan().table_width, 2048)
        for changes in (
            {"table_width": 2049},
            {"max_context": 262145},
            {"table_width": 1, "max_context": 129},
            {"rows": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)
        self.assertNotEqual(
            self.plan().reuse_key, self.plan(max_context=131072).reuse_key
        )

    def test_replay_rows_have_no_kv_projection_storage(self):
        full = self.plan()
        mixed = self.plan(write_ranges=((1, 2), (3, 4)))
        replay = self.plan(variant="query_only")
        self.assertEqual(full.storage[5], ((4, 2048), "bfloat16"))
        self.assertEqual(mixed.storage[5], ((2, 2048), "bfloat16"))
        self.assertEqual(replay.storage[5], ((0, 2048), "bfloat16"))
        self.assertNotEqual(full.reuse_key, mixed.reuse_key)
        self.assertEqual(full.scratch_bytes - mixed.scratch_bytes, 8192)
        self.assertEqual(full.scratch_bytes - replay.scratch_bytes, 16128)

    def test_write_spans_reject_overlap_order_and_out_of_bounds(self):
        for ranges in (
            ((0, 3), (2, 4)),
            ((3, 4), (0, 1)),
            ((0, 5),),
            ((-1, 1),),
            ((2, 2),),
            ((False, 1),),
            ((0, 1, 2),),
        ):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                self.plan(write_ranges=ranges)
        with self.assertRaises(ValueError):
            self.plan(variant="query_only", write_ranges=((0, 1),))

    def test_kv_only_has_no_query_attention_or_output_storage(self):
        plan = self.plan(variant="kv_only", write_ranges=((1, 4),))
        for field in (0, 4, 6, 7, 8, 9, 10, 11):
            self.assertEqual(plan.storage[field][0][0], 0)
        self.assertEqual(plan.storage[5], ((3, 2048), "bfloat16"))
        self.assertEqual(plan.scratch_bytes, 76800)

    def test_kv_only_prefill_warmup_executes_before_graph_recording(self):
        plan = self.plan(rows=33, batch=1, max_queries=33, variant="kv_only")
        device = Device()
        w, inputs, batch, state = bindings(plan)
        module, calls = native_module()
        with patch(
            "oh_my_vllm.operators.fa_fp8.workspace.load",
            return_value=(module, "source", "library"),
        ):
            workspace = prepare(plan, device, (w,))
        allocations = len(device.allocations)
        self.assertFalse(calls)
        workspace.warmup(w, inputs, batch, state)
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][-6]), (0, 33))
        with workspace.graph_lease():
            self.assertIs(workspace.run(w, inputs, batch, state), workspace.outputs)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(device.allocations), allocations)

    def test_warmup_follows_projection_spans_and_attention_paths(self):
        for variant, rows, spans, expected in (
            ("kv_only", 32, None, 0),
            ("kv_only", 64, ((0, 32), (32, 64)), 0),
            ("kv_only", 64, ((0, 31), (31, 64)), 1),
            ("kv_only", 64, ((1, 33),), 0),
            ("kv_only", 64, (), 0),
            ("full", 1, None, 1),
            ("full", 33, (), 1),
            ("query_only", 1, None, 1),
        ):
            with self.subTest(variant=variant, rows=rows, spans=spans):
                plan = self.plan(
                    rows=rows,
                    batch=1,
                    max_queries=rows,
                    variant=variant,
                    write_ranges=spans,
                )
                device = Device()
                w, inputs, batch, state = bindings(plan)
                module, calls = native_module()
                with patch(
                    "oh_my_vllm.operators.fa_fp8.workspace.load",
                    return_value=(module, "source", "library"),
                ):
                    workspace = prepare(plan, device, (w,))
                workspace.warmup(w, inputs, batch, state)
                self.assertEqual(len(calls), expected)

    def test_prepared_run_keeps_addresses_and_forwards_both_cache_buffers(self):
        plan, device = self.plan(write_ranges=((1, 4),)), Device()
        w, inputs, batch, state = bindings(plan)
        module, calls = native_module()
        with patch(
            "oh_my_vllm.operators.fa_fp8.workspace.load",
            return_value=(module, "source", "library"),
        ):
            workspace = prepare(plan, device, (w,))
        allocations = len(device.allocations)
        outputs = workspace.outputs
        for _ in range(2):
            self.assertIs(workspace.run(w, inputs, batch, state), outputs)
        self.assertEqual(len(device.allocations), allocations)
        self.assertIs(calls[0][8], state.codes)
        self.assertIs(calls[0][9], state.scales)
        self.assertIs(calls[0][14], state.write_mask)
        self.assertEqual(tuple(calls[0][-6]), (1, 4))
        self.assertEqual(len(workspace.bindings), 1)
        workspace.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            workspace.run(w, inputs, batch, state)
        with self.assertRaisesRegex(RuntimeError, "closed"), workspace.graph_lease():
            pass

    def test_changed_table_layout_fails_before_native_launch(self):
        plan, device = self.plan(), Device()
        w, inputs, batch, state = bindings(plan)
        module, calls = native_module()
        with patch(
            "oh_my_vllm.operators.fa_fp8.workspace.load",
            return_value=(module, "source", "library"),
        ):
            workspace = prepare(plan, device, (w,))
        bad = CacheView(
            state.codes, state.scales, state.slots, Tensor((2, 335), "int32")
        )
        with self.assertRaisesRegex(ValueError, "table"):
            workspace.run(w, inputs, batch, bad)
        self.assertFalse(calls)

    def test_cache_layout_rejects_wrong_page_format_before_native_load(self):
        codes = Tensor((4, 2, 128, 4, 256), "uint8")
        scales = Tensor((4, 2, 128, 4), "float32")
        for payload, divisor in (
            (Tensor((4, 2, 784, 4, 256), "uint8"), scales),
            (Tensor((4, 2, 128, 4, 256), "float8_e4m3fn"), scales),
            (codes, Tensor((4, 2, 128, 4), "bfloat16")),
            (codes, Tensor((4, 2, 128, 2), "float32")),
            (Tensor((1, 2, 128, 4, 256), "uint8"), scales),
        ):
            with patch("oh_my_vllm.operators.fa_fp8.copy.load") as load:
                with self.assertRaises(ValueError):
                    copy_prefix(payload, divisor, 1, 2, 127)
                load.assert_not_called()

    def test_copy_prefix_validates_tail_and_forwards_scale_unchanged(self):
        codes = Tensor((4, 2, 128, 4, 256), "uint8")
        scales = Tensor((4, 2, 128, 4), "float32")
        with patch("oh_my_vllm.operators.fa_fp8.copy.load") as load:
            copy_prefix(codes, scales, 1, 2, 127)
            load.return_value[0].fa_fp8_copy_pages.assert_called_once_with(
                codes, scales, 1, 2, 127
            )
            load.reset_mock()
            for source, target, valid in (
                (1, 1, 2),
                (0, 2, 1),
                (1, 4, 1),
                (1, 2, 0),
                (1, 2, 128),
                (True, 2, 1),
            ):
                with (
                    self.subTest(case=(source, target, valid)),
                    self.assertRaises(ValueError),
                ):
                    copy_prefix(codes, scales, source, target, valid)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
