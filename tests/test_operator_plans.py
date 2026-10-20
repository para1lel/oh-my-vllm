import gc
import unittest
import weakref
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.operators.fa import api as fa
from oh_my_vllm.operators.fa.plan import make_plan
from oh_my_vllm.operators.fa.workspace import prepare

from tests.test_execution import graph_device


class OperatorPlans(unittest.TestCase):
    def test_fa_plan_separates_recorded_launches(self):
        def plan(rows=4, queries=1, context=32767, variant="full"):
            return make_plan(
                rows=rows,
                batch=(rows + queries - 1) // queries,
                table_width=335,
                max_queries=queries,
                max_context=context,
                min_context=context,
                variant=variant,
            )

        self.assertEqual(plan(context=784).reuse_key, plan(context=32767).reuse_key)
        self.assertNotEqual(plan().reuse_key, plan(context=32768).reuse_key)
        for left, right in ((32, 33), (128, 129), (1023, 1024)):
            self.assertNotEqual(
                plan(left, left).reuse_key, plan(right, right).reuse_key
            )
        self.assertNotEqual(plan(1024, 512).reuse_key, plan(1024, 1024).reuse_key)
        for variant in ("kv_only", "query_only"):
            self.assertNotEqual(plan().reuse_key, plan(variant=variant).reuse_key)
        self.assertEqual(plan(context=12543).reuse_key, plan(context=12544).reuse_key)
        wide = make_plan(rows=784, batch=1, table_width=64, max_queries=784)
        narrow = make_plan(rows=784, batch=1, table_width=1, max_queries=784)
        batched = make_plan(rows=784, batch=2, table_width=64, max_queries=784)
        self.assertEqual(wide.storage, narrow.storage)
        self.assertNotEqual(wide.reuse_key, narrow.reuse_key)
        self.assertNotEqual(wide.reuse_key, batched.reuse_key)
        with self.assertRaises(ValueError):
            plan(rows=0)

    def test_vision_plan_rejects_invalid_grids_without_loading_native(self):
        from oh_my_vllm.operators.vision.plan import make_plan as vision_plan

        with patch(
            "oh_my_vllm.operators.runtime.native.load",
            side_effect=AssertionError("native load"),
        ):
            self.assertEqual(vision_plan(rows=65536, grid=(256, 256)).rows, 65536)
            self.assertEqual(vision_plan(rows=392, grid=(14, 14), frames=3).frames, 3)
            for rows, grid, frames in (
                (65540, (256, 256), 1),
                (196, (14, 14), 3),
                (256, (15, 16), 1),
                (0, (16, 16), 1),
            ):
                with self.subTest(rows=rows, grid=grid), self.assertRaises(ValueError):
                    vision_plan(rows=rows, grid=grid, frames=frames)

    def test_operator_workspace_keeps_outputs_and_graph_owners(self):
        from oh_my_vllm.operators.ffn import api as ffn
        from oh_my_vllm.operators.ffn.plan import make_plan as ffn_plan
        from oh_my_vllm.operators.ffn.workspace import prepare as prepare_ffn
        from oh_my_vllm.operators.gdn import api as gdn
        from oh_my_vllm.operators.gdn.plan import make_plan as gdn_plan
        from oh_my_vllm.operators.gdn.workspace import prepare as prepare_gdn

        class Tensor:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype

        def start_capture(family, events, fail):
            allocated = 0

            def allocate(shape, dtype):
                nonlocal allocated
                allocated += 1
                return Tensor(shape, dtype)

            device = SimpleNamespace(empty=allocate, scratch=allocate)
            arguments = ()
            if family == "fa":
                api, factory = fa, prepare
                plan = make_plan(
                    rows=4,
                    batch=4,
                    table_width=335,
                    max_queries=1,
                    max_context=784,
                    min_context=784,
                )
                bindings = tuple(
                    fa.Weights(*(object() for _ in range(7))) for _ in range(3)
                )
                arguments = (
                    fa.BatchView(object(), object(), object()),
                    fa.CacheView((object(),), object(), Tensor((4, 335), "int32")),
                )
            elif family == "gdn":
                api, factory = gdn, prepare_gdn
                schedule = gdn.Schedule(
                    (0, 4), (0,), (-1,) * 4, (3,), (1,), (0,), 2, False
                )
                plan = gdn_plan(schedule)
                bindings = tuple(
                    gdn.Weights(*(object() for _ in range(10))) for _ in range(3)
                )
                arguments = (
                    gdn.BatchView(object(), object(), object()),
                    gdn.StateView(object(), object()),
                )
            else:
                api, factory = ffn, prepare_ffn
                plan = ffn_plan(rows=4)
                bindings = tuple(
                    ffn.Weights(*(object() for _ in range(5))) for _ in range(3)
                )

            launches = []

            def launch(*args):
                if fail:
                    raise ValueError("record failed")
                launches.append(args[2])

            module = SimpleNamespace(**{family: launch})
            with patch(
                f"oh_my_vllm.operators.{family}.workspace.load",
                return_value=(module, "source", "library"),
            ):
                workspace = factory(plan, device, bindings)
            inputs = api.Inputs(object(), object())
            outputs = workspace.outputs
            count = allocated
            if not fail:
                for weights in bindings:
                    self.assertIs(
                        api.run(weights, workspace, inputs, *arguments), outputs
                    )
                self.assertEqual(len(launches), 3)
                self.assertEqual(allocated, count)
            references = [weakref.ref(workspace), weakref.ref(outputs.value)]
            references.extend(
                weakref.ref(weight, lambda _: events.append(("weight-release",)))
                for weight in bindings
            )
            operation = partial(api.run, bindings[0], workspace, inputs, *arguments)
            operation.graph_lease = workspace.graph_lease
            recording = graph_device(events).graph(operation)
            if fail:
                with self.assertRaisesRegex(ValueError, "record failed"):
                    recording.__enter__()
                return None, None, references
            replay = recording.__enter__()
            workspace.close()
            with self.assertRaises(RuntimeError):
                api.run(bindings[0], workspace, inputs, *arguments)
            return recording, replay, references

        with (
            patch("tvm_ffi.use_raw_stream", new=lambda *_: nullcontext()),
            patch("tvm_ffi.get_raw_stream", return_value=77),
        ):
            for family in ("fa", "gdn", "ffn"):
                for fail in (False, True):
                    events = []
                    with self.subTest(family=family, fail=fail):
                        recording, replay, references = start_capture(
                            family, events, fail
                        )
                        gc.collect()
                        if fail:
                            self.assertTrue(
                                all(reference() is None for reference in references)
                            )
                            continue
                        self.assertTrue(
                            all(reference() is not None for reference in references)
                        )
                        replay()
                        recording.__exit__(None, None, None)
                        del recording
                        gc.collect()
                        self.assertTrue(
                            all(reference() is None for reference in references)
                        )
                        names = [event[0] for event in events]
                        self.assertLess(
                            names.index("cudaGraphExecDestroy"),
                            names.index("weight-release"),
                        )
                        with self.assertRaisesRegex(RuntimeError, "closed"):
                            replay()
