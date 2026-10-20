import ctypes
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.operators.runtime.device import Device
from oh_my_vllm.worker.inference import Worker


class GraphOwner:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail

    @contextmanager
    def graph_lease(self):
        self.events.append(("lease",))
        try:
            yield
        finally:
            self.events.append(("release",))

    def run(self):
        self.events.append(("record",))
        if self.fail:
            raise ValueError("recording failed")


def graph_device(events, failure=None):
    device = Device.__new__(Device)
    handles = {
        "cudaStreamCreateWithFlags": (0, 11),
        "cudaStreamEndCapture": (1, 12),
        "cudaGraphInstantiateWithFlags": (0, 13),
    }

    def call(name, *args):
        events.append((name, *args))
        if name == failure:
            return 1
        if name in handles:
            argument, value = handles[name]
            ctypes.cast(args[argument], ctypes.POINTER(ctypes.c_void_p))[0] = value
        return 0

    names = (
        *handles,
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
    return device


class Execution(unittest.TestCase):
    def test_prefill_controls_the_entire_execute(self):
        worker = Worker.__new__(Worker)
        worker.speculation = None
        worker.requests = {
            "decode": SimpleNamespace(prompt_len=16, history=list(range(17))),
            "prefill": SimpleNamespace(prompt_len=40000, history=list(range(40000))),
        }
        decode = dict(request_id="decode", num_computed_tokens=16, token_ids=[13])
        prefill = dict(
            request_id="prefill", num_computed_tokens=0, token_ids=[1] * 40000
        )
        for rows, expected in (
            ([decode], [(True, True)]),
            ([prefill], [(False, False), (False, False)]),
            ([decode, prefill], [(True, False), (False, False), (False, False)]),
        ):
            calls = []

            def batch(items, verify, graph, calls=calls):
                calls.append((verify, graph))
                return [dict(request_id=item["request_id"]) for item in items]

            worker.execute_batch = batch
            result = worker.execute(
                dict(
                    rpc_id="execution",
                    scheduled=rows,
                    finished_request_ids=[],
                    preempted_request_ids=[],
                )
            )
            self.assertEqual(calls, expected)
            self.assertEqual(result["rpc_id"], "execution")
            self.assertEqual(
                {row["request_id"] for row in result["outputs"]},
                {row["request_id"] for row in rows},
            )

    def test_stop_truncation_controls_accepted_slot(self):
        from oh_my_vllm.worker.cache import Cache

        for emitted, expected_slot, expected_position in ((1, 10, 101), (0, 99, 100)):
            worker = Worker.__new__(Worker)
            request = SimpleNamespace(
                prompt_len=16,
                state_slot=99,
                state_position=100,
                finish="stop",
                thinking_count=0,
                drafts=[],
                append=lambda tokens, n=emitted: (tokens[:n], ""),
            )
            rows = [
                dict(
                    request_id="stop",
                    num_computed_tokens=100,
                    token_ids=[1, 2, 3, 4, 5],
                )
            ]
            batch = SimpleNamespace(
                rows=5,
                scheduled=rows,
                writes=[10, 11, 12, 13, 14],
                schedule=SimpleNamespace(sequence_rows=(0, 5)),
            )
            worker.requests = {"stop": request}
            cache = Cache.__new__(Cache)
            cache.batch = lambda *_, batch=batch: batch
            worker.cache = cache
            worker.model = SimpleNamespace(forward=lambda *_: ((None, None), None))
            worker.sampler = lambda *_: None
            worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
            worker.speculation = SimpleNamespace(
                accept=lambda *_: [[2, 3, 4, 5]],
                context=lambda *_: {},
                propose=lambda *_: None,
            )
            result = worker.execute_batch(rows, True, True)
            self.assertEqual(result[0]["token_ids"], [2][:emitted])
            self.assertEqual(request.state_slot, expected_slot)
            self.assertEqual(request.state_position, expected_position)

    def test_graph_owns_capture_replay_and_cleanup(self):
        with (
            patch("tvm_ffi.use_raw_stream") as use_stream,
            patch("tvm_ffi.get_raw_stream", return_value=77),
        ):
            for failure in (
                None,
                "cudaStreamBeginCapture",
                "cudaStreamEndCapture",
                "cudaGraphInstantiateWithFlags",
                "record",
            ):
                events = []
                device = graph_device(events, failure)
                owner = GraphOwner(events, failure == "record")
                if failure:
                    with (
                        self.assertRaises(
                            ValueError if failure == "record" else RuntimeError
                        ),
                        device.graph(owner.run),
                    ):
                        self.fail("failed recording must not produce a replay")
                    self.assertEqual(events[-1][0], "release")
                    self.assertIn("cudaStreamDestroy", [event[0] for event in events])
                    continue
                with device.graph(owner.run) as replay:
                    self.assertNotIn("cudaGraphLaunch", [event[0] for event in events])
                    replay()
                    self.assertEqual(events[-1][-1], 77)
                names = [event[0] for event in events]
                self.assertLess(
                    names.index("synchronize"), names.index("cudaGraphExecDestroy")
                )
                self.assertLess(
                    names.index("cudaGraphExecDestroy"), names.index("release")
                )
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    replay()
            events = []
            device = graph_device(events)
            with (
                self.assertRaises(KeyboardInterrupt),
                device.graph(GraphOwner(events).run) as replay,
                patch.object(
                    device.runtime, "cudaGraphLaunch", side_effect=KeyboardInterrupt
                ),
            ):
                replay()
            self.assertEqual(events[-1][0], "release")
            self.assertEqual(sum(event[0] == "synchronize" for event in events), 2)
            self.assertEqual(use_stream.call_count, 6)


if __name__ == "__main__":
    unittest.main()
