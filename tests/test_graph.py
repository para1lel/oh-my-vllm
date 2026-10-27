import ctypes
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from oh_my_vllm.operators.runtime.device import Device


class GraphOwner:
    def __init__(self, events):
        self.events = events

    def run(self):
        self.events.append(("record",))


class LeasedOwner(GraphOwner):
    def __init__(self, events, failure=None):
        super().__init__(events)
        self.failure = failure

    def run(self):
        if self.failure is not None:
            raise self.failure
        super().run()

    @contextmanager
    def graph_lease(self):
        self.events.append(("lease",))
        try:
            yield
        finally:
            self.events.append(("release",))


def graph_device(events):
    device = Device.__new__(Device)
    handles = {
        "cudaStreamCreateWithFlags": (0, 11),
        "cudaStreamEndCapture": (1, 12),
        "cudaGraphInstantiateWithFlags": (0, 13),
    }
    event_handle = [21]

    def call(name, *args):
        snapshot = tuple(
            argument.value if isinstance(argument, ctypes.c_void_p) else argument
            for argument in args
        )
        events.append((name, *snapshot))
        if name == "cudaEventCreateWithFlags":
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = event_handle[0]
            event_handle[0] += 1
        elif name in handles:
            argument, value = handles[name]
            ctypes.cast(args[argument], ctypes.POINTER(ctypes.c_void_p))[0] = value
        return 0

    names = (
        "cudaEventCreateWithFlags",
        *handles,
        "cudaEventRecord",
        "cudaEventDestroy",
        "cudaStreamIsCapturing",
        "cudaStreamWaitEvent",
        "cudaStreamBeginCapture",
        "cudaStreamDestroy",
        "cudaGraphDestroy",
        "cudaGraphExecDestroy",
        "cudaGraphLaunch",
    )
    device.runtime = SimpleNamespace(
        **{name: lambda *args, name=name: call(name, *args) for name in names},
        cudaGetErrorString=lambda _: b"injected CUDA failure",
    )
    device.synchronize = lambda: events.append(("synchronize",))
    device.transfer_engines = set()
    return device


class GraphCapture(unittest.TestCase):
    def test_capture_handoffs_caller_and_transfer_streams(self):
        events = []
        device = graph_device(events)
        device.transfer_engines = (
            SimpleNamespace(
                streams={"h2d": ctypes.c_void_p(88), "d2h": ctypes.c_void_p(99)}
            ),
        )
        with (
            patch("tvm_ffi.get_raw_stream", return_value=77),
            device.graph(GraphOwner(events).run),
        ):
            pass
        records = [event for event in events if event[0] == "cudaEventRecord"]
        waits = [event for event in events if event[0] == "cudaStreamWaitEvent"]
        self.assertEqual({event[2] for event in records}, {77, 88, 99})
        self.assertEqual({event[1] for event in waits}, {11})
        self.assertEqual(len(records), len(waits))

    def test_capture_restores_the_caller_stream_after_recording(self):
        events = []
        device = graph_device(events)
        current = [77]
        observed = []

        @contextmanager
        def use_raw_stream(_, stream):
            previous = current[0]
            current[0] = stream
            try:
                yield
            finally:
                current[0] = previous

        with (
            patch("tvm_ffi.get_raw_stream", side_effect=lambda _: current[0]),
            patch("tvm_ffi.use_raw_stream", use_raw_stream),
        ):
            device.capture(lambda: observed.append(current[0]))
        self.assertEqual(observed, [11])
        self.assertEqual(current, [77])

    def test_capture_rejects_an_active_caller_stream(self):
        events = []
        device = graph_device(events)

        def active(stream, status):
            events.append(("cudaStreamIsCapturing", stream))
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0

        device.runtime.cudaStreamIsCapturing = active
        with (
            patch("tvm_ffi.get_raw_stream", return_value=77),
            self.assertRaisesRegex(RuntimeError, "already capturing"),
        ):
            device.capture(lambda: None)
        self.assertNotIn("cudaStreamBeginCapture", [event[0] for event in events])

    def test_capture_cleanup_does_not_retry_destroyed_events(self):
        events = []
        device = graph_device(events)
        device.transfer_engines = (
            SimpleNamespace(streams={"h2d": ctypes.c_void_p(88)}),
        )
        destroy_calls = {}

        def destroy_event(handle):
            value = handle.value
            destroy_calls[value] = destroy_calls.get(value, 0) + 1
            events.append(("cudaEventDestroy", value))
            return 1 if value == 22 and destroy_calls[value] == 1 else 0

        device.runtime.cudaEventDestroy = destroy_event
        with self.assertRaisesRegex(RuntimeError, "injected CUDA failure"):
            device.capture(lambda: None)
        self.assertEqual(destroy_calls, {21: 1, 22: 2})

    def test_capture_cleanup_retries_a_failed_graph_destroy(self):
        events = []
        device = graph_device(events)
        destroy_calls = [0]

        def destroy_graph(handle):
            destroy_calls[0] += 1
            events.append(("cudaGraphDestroy", handle))
            return 1 if destroy_calls[0] == 1 else 0

        device.runtime.cudaGraphDestroy = destroy_graph
        with self.assertRaisesRegex(RuntimeError, "injected CUDA failure"):
            device.capture(lambda: None)
        self.assertEqual(destroy_calls[0], 2)

    def test_capture_handoff_failures_release_partial_handles(self):
        for failure in (
            "cudaEventCreateWithFlags",
            "cudaEventRecord",
            "cudaStreamWaitEvent",
            "cudaStreamDestroy",
        ):
            with self.subTest(failure=failure):
                events = []
                device = graph_device(events)

                def fail(*args, name=failure, event_log=events):
                    event_log.append((name, *args))
                    return 1

                setattr(device.runtime, failure, fail)
                with self.assertRaisesRegex(RuntimeError, "injected CUDA failure"):
                    device.capture(lambda: None)
                names = [event[0] for event in events]
                stream_destroys = names.count("cudaStreamDestroy")
                event_destroys = names.count("cudaEventDestroy")
                self.assertEqual(
                    stream_destroys, 3 if failure == "cudaStreamDestroy" else 1
                )
                self.assertEqual(
                    event_destroys, 0 if failure == "cudaEventCreateWithFlags" else 1
                )

    def test_release_graph_retries_a_failed_executable_destroy(self):
        events = []
        device = graph_device(events)
        destroy_calls = [0]

        def destroy_executable(handle):
            destroy_calls[0] += 1
            events.append(("cudaGraphExecDestroy", handle))
            return 1 if destroy_calls[0] == 1 else 0

        device.runtime.cudaGraphExecDestroy = destroy_executable
        with (
            self.assertRaisesRegex(RuntimeError, "injected CUDA failure"),
            device.graph(lambda: None),
        ):
            pass
        self.assertEqual(destroy_calls[0], 2)

    def test_graph_body_failure_preserves_cleanup_error_and_poison_device(self):
        events = []
        device = graph_device(events)
        device.synchronize = lambda: (_ for _ in ()).throw(
            RuntimeError("synchronize failed")
        )
        with (
            self.assertRaisesRegex(ValueError, "body failed") as failure,
            device.graph(lambda: None),
        ):
            raise ValueError("body failed")
        self.assertIsInstance(failure.exception.__cause__, RuntimeError)
        self.assertTrue(device.graph_failed)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            device.capture(lambda: None)

    def test_graph_body_failure_keeps_sync_and_executable_cleanup_causes(self):
        events = []
        device = graph_device(events)
        device.synchronize = lambda: (_ for _ in ()).throw(
            RuntimeError("synchronize failure")
        )
        destroy_calls = [0]

        def destroy_executable(handle):
            destroy_calls[0] += 1
            return 2 if destroy_calls[0] == 1 else 3

        device.runtime.cudaGraphExecDestroy = destroy_executable
        device.runtime.cudaGetErrorString = lambda status: {
            2: b"executable destroy failure",
            3: b"executable retry failure",
        }[status]
        with (
            self.assertRaisesRegex(ValueError, "body failed") as failure,
            device.graph(lambda: None),
        ):
            raise ValueError("body failed")
        cause = failure.exception.__cause__
        self.assertIsInstance(cause, ExceptionGroup)
        messages = {str(error) for error in cause.exceptions}
        self.assertTrue(any("synchronize failure" in message for message in messages))
        self.assertTrue(
            any("executable destroy failure" in message for message in messages)
        )
        self.assertTrue(
            any("executable retry failure" in message for message in messages)
        )
        self.assertEqual(destroy_calls[0], 2)

    def test_body_failure_checks_stream_end_capture(self):
        events = []
        device = graph_device(events)
        device.runtime.cudaStreamEndCapture = lambda *_: 1
        owner = LeasedOwner(events, ValueError("body failed"))

        with (
            self.assertRaisesRegex(ValueError, "body failed") as failure,
            device.graph(owner.run),
        ):
            pass
        self.assertIsInstance(failure.exception.__cause__, RuntimeError)
        self.assertTrue(device.graph_failed)
        self.assertNotIn(("release",), events)

    def test_capture_failure_keeps_multiple_cleanup_causes(self):
        events = []
        device = graph_device(events)
        device.runtime.cudaStreamEndCapture = lambda *_: 2
        device.runtime.cudaStreamDestroy = lambda *_: 3
        device.runtime.cudaGetErrorString = lambda status: {
            2: b"capture end failure",
            3: b"stream destroy failure",
        }[status]
        owner = LeasedOwner(events, ValueError("body failed"))

        with (
            self.assertRaisesRegex(ValueError, "body failed") as failure,
            device.graph(owner.run),
        ):
            pass
        cause = failure.exception.__cause__
        self.assertIsInstance(cause, ExceptionGroup)
        messages = [str(error) for error in cause.exceptions]
        self.assertEqual(
            messages,
            [
                "CUDA runtime error 2: capture end failure",
                "CUDA runtime error 3: stream destroy failure",
                "CUDA runtime error 3: stream destroy failure",
            ],
        )
        self.assertTrue(device.graph_failed)
        self.assertNotIn(("release",), events)

    def test_poisoned_device_rejects_an_existing_graph_replay(self):
        events = []
        device = graph_device(events)
        with device.graph(lambda: None) as replay:
            device.graph_failed = True
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                replay()
        self.assertNotIn("cudaGraphLaunch", [event[0] for event in events])

    def test_failed_graph_cleanup_retains_owner_lease(self):
        events = []
        device = graph_device(events)
        device.runtime.cudaGraphExecDestroy = lambda *_: 1
        owner = LeasedOwner(events)
        with (
            self.assertRaises(ExceptionGroup),
            device.graph(owner.run),
        ):
            pass
        self.assertIn(("lease",), events)
        self.assertNotIn(("release",), events)

    def test_decode_workspace_cache_closes_oldest_graph_owner(self):
        from oh_my_vllm.models.qwen38 import model

        closed = []
        live = []
        peak = [0]

        def initialize(workspace, _, __, ___, batch, ____):
            workspace.output = object()
            workspace.hidden = None
            workspace.runs = 0
            workspace.key = batch.rows
            live.append(workspace.key)
            peak[0] = max(peak[0], len(live))

        def close(workspace):
            closed.append(workspace)
            live.remove(workspace.key)

        def batch(rows):
            return SimpleNamespace(
                rows=rows,
                schedule=SimpleNamespace(verify=True),
                scheduled=[dict(request_id=str(rows), token_ids=[1])],
                requests=[SimpleNamespace(media=None)],
            )

        owner = model.Model(
            SimpleNamespace(),
            None,
            None,
            False,
            SimpleNamespace(rows=lambda _: (None, None)),
        )
        with (
            patch.object(model, "MAX_GRAPH_WORKSPACES", 2),
            patch.object(
                model,
                "make_plans",
                side_effect=lambda batch, _: (SimpleNamespace(reuse_key=batch.rows),),
            ),
            patch.object(model.Workspace, "__init__", initialize),
            patch.object(model.Workspace, "close", close),
            patch.object(model.Workspace, "update", lambda *_: None),
            patch.object(model.Workspace, "bind_media", lambda *_: None),
            patch.object(
                model.Workspace,
                "run",
                lambda workspace, graph: workspace.output,
            ),
        ):
            for rows in (0, 1, 0, 2):
                owner.forward(batch(rows), False)
                self.assertLessEqual(len(live), 2)
            self.assertEqual(len(owner.decode), 2)
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0].key, 1)
            self.assertEqual(peak[0], 2)
            owner.close()
            self.assertEqual(sorted(workspace.key for workspace in closed), [0, 1, 2])

    def test_speculation_context_cache_closes_oldest_graph_owner(self):
        fake_guidance = SimpleNamespace(
            LLExecutor=lambda *_: None,
            LLMatcher=SimpleNamespace(grammar_from_json_schema=lambda *_: None),
        )
        with patch.dict(sys.modules, {"llguidance": fake_guidance}):
            from oh_my_vllm.models.qwen38 import speculation

            request = SimpleNamespace(
                state_position=2,
                mtp_position=0,
                history=[0, 1],
                finish=False,
                generated=0,
                limit=10,
                position=lambda value: (value,) * 3,
            )
            batch = SimpleNamespace(
                requests=[request],
                scheduled=[
                    dict(
                        request_id="request",
                        num_computed_tokens=0,
                        fa_block_table=[0],
                    )
                ],
                schedule=SimpleNamespace(
                    verify=True, sequence_rows=(0, 2), read_slots=(0,)
                ),
                metadata=(None, None, np.zeros((1, 1), np.int32)),
            )
            closed = []
            live = []
            peak = [0]
            plan_keys = iter((0, 1, 0, 2))

            def make_context(*args):
                key = args[3].reuse_key
                live.append(key)
                peak[0] = max(peak[0], len(live))
                return SimpleNamespace(
                    copy=lambda *_: None,
                    hidden=args[4],
                    endpoint_rows=(object(),),
                    execute=lambda *_: None,
                    metadata_inputs=SimpleNamespace(update=lambda *_: None),
                    close=lambda: (closed.append(key), live.remove(key)),
                )

            owner = speculation.Speculation(
                SimpleNamespace(view=lambda *_: object()),
                None,
                SimpleNamespace(),
                SimpleNamespace(rows=lambda _: (None, None)),
                SimpleNamespace(record=lambda *_, **__: None),
            )
            with (
                patch.object(speculation, "MAX_GRAPH_WORKSPACES", 2),
                patch.object(
                    speculation,
                    "context_plan",
                    side_effect=lambda **_: SimpleNamespace(reuse_key=next(plan_keys)),
                ),
                patch.object(speculation, "Context", side_effect=make_context),
            ):
                for _ in range(4):
                    request.mtp_position = 0
                    owner.context(batch, np.zeros((1, 1), np.float32), True, set())
                    self.assertLessEqual(len(live), 2)
                self.assertEqual(len(owner.context_decode), 2)
                self.assertEqual(closed, [1])
                self.assertEqual(peak[0], 2)
                owner.close()
                self.assertEqual(sorted(closed), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
