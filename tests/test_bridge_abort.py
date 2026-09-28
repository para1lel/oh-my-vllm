"""Real DEALER bridge tests for asynchronous preparation and abort cleanup."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgpack
import zmq
from oh_my_vllm.worker.protocol import RequestOutput, WorkerOutput
from oh_my_vllm.worker.serving import RequestValidationError
from oh_my_vllm.worker.zmq_bridge import serve


class FakeAdapter:
    def __init__(self):
        self.generations = {}
        self.release = threading.Event()
        self.entered = threading.Event()
        self.prepare_thread = None

    def prepare_inputs(self, request):
        self.prepare_thread = threading.get_ident()
        self.entered.set()
        if request.get("slow") and not self.release.wait(5):
            raise RuntimeError("test prepare release timed out")
        if request.get("error") == "validation":
            raise RequestValidationError("bad schema")
        if request.get("error") == "internal":
            raise ValueError("tokenizer failed")
        return [1, 2], object(), object()


class FakeWorker:
    def __init__(self):
        self.serving = FakeAdapter()
        self.config = SimpleNamespace(model="unused", max_model_len=262144)
        self.histories = {}
        self.samplers = {}
        self.registration_thread = None
        self.executions = 0
        self.shutdown_calls = 0

    def cache_capacities(self):
        return 85, 16

    def register_request(self, request_id, prompt_token_ids, sampling_params=None):
        self.registration_thread = threading.get_ident()
        if request_id in self.histories:
            raise ValueError("duplicate request_id")
        if request_id == 99:
            raise ValueError("invalid prompt tokens")
        self.histories[request_id] = list(prompt_token_ids)
        self.samplers[request_id] = sampling_params

    def unregister_request(self, request_id):
        self.histories.pop(request_id, None)
        self.samplers.pop(request_id, None)
        self.serving.generations.pop(request_id, None)

    def execute_model(self, scheduled):
        self.executions += 1
        return WorkerOutput([RequestOutput(1, [self.executions])])

    def shutdown(self):
        self.shutdown_calls += 1


class BridgeAbortTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oh-my-vllm-bridge-")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.RCVTIMEO = 1500
        self.address = "ipc://" + str(Path(self.temp.name) / "worker.ipc")
        self.socket.bind(self.address)
        self.worker = FakeWorker()
        self.patches = [
            patch(
                "oh_my_vllm.worker.zmq_bridge._handle_init",
                return_value=self.worker,
            ),
            patch("oh_my_vllm.worker.zmq_bridge.configure_logging"),
        ]
        for active in self.patches:
            active.start()
        self.thread = threading.Thread(target=serve, args=(self.address,), daemon=True)
        self.thread.start()
        self.send({"type": "init"})
        self.assertEqual(self.recv()["type"], "ready")

    def tearDown(self):
        if self.thread.is_alive():
            self.send({"type": "shutdown"})
            self.thread.join(1)
        self.worker.serving.release.set()
        self.assertFalse(self.thread.is_alive(), "bridge shutdown must be bounded")
        self.socket.close(linger=0)
        self.context.term()
        for active in reversed(self.patches):
            active.stop()
        self.temp.cleanup()

    def send(self, message):
        self.socket.send(msgpack.packb(message, use_bin_type=True))

    def recv(self):
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def prepare(self, rpc_id, request_id, **request):
        self.send(
            {
                "type": "prepare",
                "rpc_id": rpc_id,
                "request_id": request_id,
                "request": request,
            }
        )

    def execute(self, rpc_id):
        self.send(
            {
                "type": "execute",
                "rpc_id": rpc_id,
                "step_id": rpc_id,
                "scheduled": [],
                "finished_request_ids": [],
                "preempted_request_ids": [],
                "num_batched_tokens": 0,
            }
        )

    def test_prepare_error_kind_is_explicit_and_cleans_up(self):
        for rpc_id, kind in ((11, "validation"), (12, "internal")):
            with self.subTest(kind=kind):
                self.prepare(rpc_id, rpc_id, error=kind)
                reply = self.recv()
                self.assertEqual((reply["type"], reply["kind"]), ("error", kind))
                self.assertEqual(reply["rpc_id"], rpc_id)
                self.assertNotIn(rpc_id, self.worker.histories)

    def test_abort_notifies_worker_without_unsolicited_reply(self):
        self.prepare(11, 7)
        self.assertEqual(self.recv()["type"], "prepared")
        self.send({"type": "abort", "request_id": 7})
        self.execute(12)
        self.assertEqual(self.recv()["type"], "execute_result")
        self.assertNotIn(7, self.worker.histories)
        self.assertNotIn(7, self.worker.serving.generations)

    def test_duplicate_prepare_preserves_live_request(self):
        self.prepare(11, 7)
        self.assertEqual(self.recv()["type"], "prepared")
        self.prepare(12, 7)
        self.assertEqual(self.recv()["message"], "duplicate request_id")
        self.assertIn(7, self.worker.histories)

    def test_register_during_prepare_cannot_remove_prepared_state(self):
        self.prepare(11, 7, slow=True)
        self.assertTrue(self.worker.serving.entered.wait(1))
        self.send({"type": "register", "request_id": 7, "prompt_token_ids": [9]})
        self.execute(12)
        self.assertEqual(self.recv()["type"], "execute_result")
        self.assertNotIn(7, self.worker.histories)
        self.worker.serving.release.set()
        self.assertEqual(self.recv()["type"], "prepared")
        self.assertEqual(self.worker.histories[7], [1, 2])

    def test_existing_register_survives_duplicate_prepare(self):
        self.send({"type": "register", "request_id": 7, "prompt_token_ids": [9]})
        self.execute(11)
        self.assertEqual(self.recv()["type"], "execute_result")
        self.prepare(12, 7)
        self.assertEqual(self.recv()["message"], "duplicate request_id")
        self.assertEqual(self.worker.histories[7], [9])

    def test_bad_registration_does_not_kill_bridge(self):
        self.prepare(11, 99)
        self.assertEqual(self.recv()["type"], "error")
        self.execute(12)
        self.assertEqual(self.recv()["type"], "execute_result")

    def test_completion_wakes_bridge_without_new_rpc(self):
        start = time.monotonic()
        self.prepare(11, 7)
        self.assertEqual(self.recv()["type"], "prepared")
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertNotEqual(
            self.worker.serving.prepare_thread, self.worker.registration_thread
        )
        self.assertEqual(self.worker.histories[7], [1, 2])

    def test_both_prepare_execute_reply_orders(self):
        self.prepare(11, 7, slow=True)
        self.execute(12)
        self.assertEqual(self.recv()["type"], "execute_result")
        self.worker.serving.release.set()
        self.assertEqual(self.recv()["type"], "prepared")
        self.prepare(13, 8)
        self.assertEqual(self.recv()["type"], "prepared")
        self.execute(14)
        self.assertEqual(self.recv()["type"], "execute_result")

    def test_cancelled_late_result_cannot_register_or_shift_next_rpc(self):
        self.prepare(11, 7, slow=True)
        self.send({"type": "abort", "request_id": 7})
        self.prepare(12, 8)
        self.worker.serving.release.set()
        reply = self.recv()
        self.assertEqual((reply["type"], reply["rpc_id"]), ("prepared", 12))
        self.assertNotIn(7, self.worker.histories)
        self.assertIn(8, self.worker.histories)
        self.execute(13)
        self.assertEqual(self.recv()["rpc_id"], 13)

    def test_shutdown_while_prepare_runs_is_bounded(self):
        self.prepare(11, 7, slow=True)
        start = time.monotonic()
        self.send({"type": "shutdown"})
        self.thread.join(0.5)
        self.assertFalse(self.thread.is_alive())
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(self.worker.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
