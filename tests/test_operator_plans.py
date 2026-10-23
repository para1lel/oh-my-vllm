import gc
import os
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

    def test_quantization_plan_rejects_invalid_layouts_without_loading_native(self):
        from oh_my_vllm.operators.quantization.plan import make_plan as quant_plan

        with patch(
            "oh_my_vllm.operators.runtime.native.load",
            side_effect=AssertionError("native load"),
        ):
            for mode, encoded, groups in (
                ("fp8_ue8m0", 64, 2),
                ("fp4_ue8m0", 32, 2),
                ("fp4_e4m3", 32, 4),
            ):
                plan = quant_plan(rows=2, width=64, mode=mode)
                self.assertEqual(
                    plan.storage,
                    (
                        ((2, encoded), "uint8"),
                        ((2, groups), "uint8"),
                        ((2, 64), "bfloat16"),
                        ((2, groups), "uint8"),
                    ),
                )
                self.assertNotEqual(
                    plan.reuse_key,
                    quant_plan(rows=0, width=64, mode=mode).reuse_key,
                )
                self.assertEqual(
                    quant_plan(rows=0, width=64, mode=mode).storage[0][0][0], 0
                )
            for rows, width, mode in (
                (-1, 32, "fp8_ue8m0"),
                (1, 0, "fp8_ue8m0"),
                (1, 16, "fp8_ue8m0"),
                (1, 48, "fp4_ue8m0"),
                (1, 24, "fp4_e4m3"),
                (1, 32, "fp8_e5m2"),
                (True, 32, "fp8_ue8m0"),
                (1, 32.0, "fp8_ue8m0"),
                (1 << 31, 32, "fp8_ue8m0"),
                (1 << 20, 1 << 20, "fp8_ue8m0"),
            ):
                with (
                    self.subTest(rows=rows, width=width, mode=mode),
                    self.assertRaises(ValueError),
                ):
                    quant_plan(rows=rows, width=width, mode=mode)

    def test_operator_workspace_keeps_outputs_and_graph_owners(self):
        from oh_my_vllm.operators.ffn import api as ffn
        from oh_my_vllm.operators.ffn.plan import make_plan as ffn_plan
        from oh_my_vllm.operators.ffn.workspace import prepare as prepare_ffn
        from oh_my_vllm.operators.gdn import api as gdn
        from oh_my_vllm.operators.gdn.plan import make_plan as gdn_plan
        from oh_my_vllm.operators.gdn.workspace import prepare as prepare_gdn
        from oh_my_vllm.operators.quantization import api as quantization
        from oh_my_vllm.operators.quantization.plan import make_plan as quant_plan
        from oh_my_vllm.operators.quantization.workspace import prepare as prepare_quant

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
            elif family == "quantization":
                api, factory = quantization, prepare_quant
                plan = quant_plan(rows=2, width=64, mode="fp4_e4m3")
                bindings = tuple(api.Inputs(object()) for _ in range(3))
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

            function = "quantize" if family == "quantization" else family
            module = SimpleNamespace(**{function: launch})
            with patch(
                f"oh_my_vllm.operators.{family}.workspace.load",
                return_value=(module, "source", "library"),
            ):
                workspace = (
                    factory(plan, device)
                    if family == "quantization"
                    else factory(plan, device, bindings)
                )
            inputs = (
                None if family == "quantization" else api.Inputs(object(), object())
            )

            def operation_for(binding):
                if family == "quantization":
                    return partial(api.run, workspace, binding)
                return partial(api.run, binding, workspace, inputs, *arguments)

            outputs = workspace.outputs
            count = allocated
            if not fail:
                for weights in bindings:
                    self.assertIs(operation_for(weights)(), outputs)
                self.assertEqual(len(launches), 3)
                self.assertEqual(allocated, count)
            result = outputs.encoded if family == "quantization" else outputs.value
            references = [weakref.ref(workspace), weakref.ref(result)]
            owners = bindings[:1] if family == "quantization" else bindings
            references.extend(
                weakref.ref(weight, lambda _: events.append(("weight-release",)))
                for weight in owners
            )
            operation = operation_for(bindings[0])
            operation.graph_lease = workspace.graph_lease
            recording = graph_device(events).graph(operation)
            if fail:
                with self.assertRaisesRegex(ValueError, "record failed"):
                    recording.__enter__()
                return None, None, references
            replay = recording.__enter__()
            workspace.close()
            with self.assertRaises(RuntimeError):
                operation_for(bindings[0])()
            return recording, replay, references

        with (
            patch("tvm_ffi.use_raw_stream", new=lambda *_: nullcontext()),
            patch("tvm_ffi.get_raw_stream", return_value=77),
        ):
            for family in ("fa", "gdn", "ffn", "quantization"):
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

    def test_engram_graph_inputs_have_separate_lifetimes(self):
        from oh_my_vllm.operators.engram import lookup
        from oh_my_vllm.operators.engram.api import Inputs
        from oh_my_vllm.operators.engram.plan import make_plan
        from oh_my_vllm.operators.engram.shared import SharedWeightFile
        from oh_my_vllm.operators.engram.workspace import prepare

        page = os.sysconf("SC_PAGE_SIZE")
        with SharedWeightFile(4, limit=page) as shared:
            self.assertEqual(shared.append(b"\x01\x02"), 0)
            with self.assertRaises(ValueError):
                shared.seal()
            with self.assertRaises(RuntimeError):
                lookup.register(shared, limit=page)
            self.assertEqual(shared.append(b"\x03\x04"), 2)
            shared.seal()
            self.assertEqual(os.pread(shared.fileno(), 4, 0), b"\x01\x02\x03\x04")
            with self.assertRaises(OSError):
                os.pwrite(shared.fileno(), b"\x05", 0)
            with self.assertRaises(RuntimeError):
                shared.append(b"")
        with self.assertRaises(RuntimeError):
            shared.fileno()
        with self.assertRaises(ValueError):
            SharedWeightFile(page + 1, limit=page + 1)

        class Tensor:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype

        plan = make_plan(
            rows=1,
            pad_id=2,
            vocabulary_size=10,
            layers=1,
            heads=1,
            max_ngram_size=3,
        )
        status = [1]
        ready = [True]
        event_ids = []

        def capture(stream, result):
            self.assertIn(stream, (77, 88))
            result._obj.value = status[0]
            return 0

        def create_event(result, flags):
            event_ids.append(len(event_ids) + 1)
            result._obj.value = event_ids[-1]
            return 0

        device = SimpleNamespace(
            empty=Tensor,
            checked=lambda result: self.assertEqual(result, 0),
            synchronize=lambda: None,
            runtime=SimpleNamespace(
                cudaStreamIsCapturing=capture,
                cudaEventCreateWithFlags=create_event,
                cudaEventRecord=lambda *args: 0,
                cudaEventQuery=lambda event: 0 if ready[0] else 600,
                cudaEventSynchronize=lambda event: 0,
                cudaEventDestroy=lambda event: 0,
            ),
        )
        module = SimpleNamespace(
            engram_hash=lambda *args: None, engram_lookup=lambda *args: None
        )

        for family in ("hash", "lookup"):
            with self.subTest(family=family):
                table_ref = None
                if family == "hash":
                    target = "oh_my_vllm.operators.engram.workspace.load"
                    factory = partial(prepare, plan, device)
                else:
                    target = "oh_my_vllm.operators.engram.lookup.load"
                    table = Tensor((528,), "uint8")
                    table_ref = weakref.ref(table)
                    lookup_plan = lookup.make_plan(
                        rows=1, heads=1, table_rows=2, table_offset=0, scale_offset=512
                    )
                    factory = partial(lookup.prepare, lookup_plan, device, table)
                    for changes in (
                        dict(rows=385),
                        dict(heads=25),
                        dict(table_rows=0),
                        dict(scale_offset=511),
                    ):
                        arguments = dict(
                            rows=1,
                            heads=1,
                            table_rows=2,
                            table_offset=0,
                            scale_offset=512,
                        )
                        arguments.update(changes)
                        with self.assertRaises(ValueError):
                            lookup.make_plan(**arguments)
                    self.assertEqual(
                        lookup_plan.storage,
                        (((1, 1, 256), "bfloat16"), ((1, 1), "uint8")),
                    )
                with patch(target, return_value=(module, "source", "library")):
                    workspace = factory()
                del factory
                if family == "lookup":
                    del table

                def invoke(family=family, workspace=workspace):
                    inputs = (
                        Inputs(*(SimpleNamespace(shape=s) for s in plan.input_shapes))
                        if family == "hash"
                        else Tensor((1, 1), "int64")
                    )
                    workspace.run(inputs)
                    return weakref.ref(inputs)

                with patch("tvm_ffi.get_raw_stream", return_value=77):
                    if family == "lookup":
                        status[0] = 1
                        with self.assertRaises(RuntimeError):
                            invoke()
                        status[0], ready[0] = 0, False
                        with patch.object(
                            device.runtime,
                            "cudaEventSynchronize",
                            wraps=device.runtime.cudaEventSynchronize,
                        ) as waited:
                            pending = [
                                invoke() for _ in range(workspace.max_pending + 2)
                            ]
                            self.assertTrue(waited.called)
                        gc.collect()
                        self.assertLessEqual(
                            sum(ref() is not None for ref in pending),
                            workspace.max_pending,
                        )
                        with (
                            patch("tvm_ffi.get_raw_stream", return_value=88),
                            self.assertRaisesRegex(RuntimeError, "stream"),
                        ):
                            invoke()
                        ready[0] = True
                        invoke()
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in pending))
                    status[0] = 0
                    with workspace.graph_lease():
                        status[0] = 1
                        with patch(
                            "tvm_ffi.get_raw_stream",
                            return_value=88 if family == "lookup" else 77,
                        ):
                            first = [invoke(), invoke()]
                            if family == "lookup":
                                with self.assertRaisesRegex(RuntimeError, "capture"):
                                    workspace.close()
                        status[0] = 0
                        eager = [invoke() for _ in range(100)]
                        gc.collect()
                        self.assertTrue(all(ref() is not None for ref in first))
                        self.assertEqual(sum(ref() is not None for ref in eager), 1)
                        with workspace.graph_lease():
                            status[0] = 1
                            second = [invoke(), invoke()]
                            status[0] = 0
                            invoke()
                            gc.collect()
                            self.assertTrue(all(ref() is not None for ref in second))
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in second))
                        self.assertTrue(all(ref() is not None for ref in first))
                        workspace.close()
                        self.assertTrue(all(ref() is not None for ref in first))
                        if table_ref:
                            self.assertIsNotNone(table_ref())
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in first))
                    if table_ref:
                        self.assertIsNone(table_ref())
                    with self.assertRaises(RuntimeError):
                        invoke()
                    if family == "lookup":
                        with (
                            patch(target, return_value=(module, "source", "library")),
                            patch.object(lookup, "_failed_workspaces", []) as retained,
                        ):
                            broken = lookup.prepare(
                                lookup_plan, device, Tensor((528,), "uint8")
                            )
                            broken.run(Tensor((1, 1), "int64"))
                            with (
                                patch.object(
                                    device.runtime,
                                    "cudaEventSynchronize",
                                    side_effect=RuntimeError("completion failed"),
                                ),
                                self.assertRaisesRegex(RuntimeError, "completion"),
                            ):
                                broken.close()
                            self.assertEqual(retained, [broken])
                            broken.close()
                            self.assertIsNotNone(broken.table)
                            with self.assertRaisesRegex(RuntimeError, "closed"):
                                broken.run(Tensor((1, 1), "int64"))
                status[0] = 0
