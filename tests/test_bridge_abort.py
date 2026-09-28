"""An abort releases request state without an unsolicited RPC reply."""

import unittest
from unittest.mock import Mock, patch

import msgpack
from oh_my_vllm.worker.protocol import RequestOutput, WorkerOutput
from oh_my_vllm.worker.zmq_bridge import serve


class BridgeAbortTest(unittest.TestCase):
    def test_abort_notifies_worker_without_unsolicited_reply(self):
        worker = Mock(logical_num_blocks=85)
        context = Mock()
        socket = context.socket.return_value
        socket.recv.side_effect = [
            msgpack.packb(message)
            for message in (
                {"type": "init"},
                {"type": "abort", "request_id": 7},
                {"type": "shutdown"},
            )
        ]
        with (
            patch("oh_my_vllm.worker.zmq_bridge.zmq.Context", return_value=context),
            patch("oh_my_vllm.worker.zmq_bridge._handle_init", return_value=worker),
            patch("oh_my_vllm.worker.zmq_bridge.configure_logging"),
        ):
            serve("ipc:///unused-test-socket")
        worker.execute_model.assert_not_called()
        worker.unregister_request.assert_called_once_with(7)
        self.assertEqual(socket.send.call_count, 1)  # Only the init ready reply.
        worker.shutdown.assert_called_once()
        socket.close.assert_called_once()
        context.term.assert_called_once()

    def test_duplicate_prepare_preserves_live_request(self):
        worker = Mock(logical_num_blocks=85, histories={7: [1]})
        context = Mock()
        socket = context.socket.return_value
        socket.recv.side_effect = [
            msgpack.packb(message)
            for message in (
                {"type": "init"},
                {"type": "prepare", "rpc_id": 11, "request_id": 7, "request": {}},
                {"type": "shutdown"},
            )
        ]
        with (
            patch("oh_my_vllm.worker.zmq_bridge.zmq.Context", return_value=context),
            patch("oh_my_vllm.worker.zmq_bridge._handle_init", return_value=worker),
            patch("oh_my_vllm.worker.zmq_bridge.configure_logging"),
        ):
            serve("ipc:///unused-test-socket")
        replies = [
            msgpack.unpackb(call.args[0], raw=False)
            for call in socket.send.call_args_list
        ]
        self.assertEqual(replies[1]["rpc_id"], 11)
        self.assertEqual(replies[1]["message"], "duplicate request_id")
        worker.unregister_request.assert_not_called()
        worker.prepare_request.assert_not_called()

    def test_bad_registration_does_not_kill_bridge(self):
        worker = Mock(logical_num_blocks=85)
        worker.register_request.side_effect = ValueError("invalid prompt tokens")
        worker.execute_model.return_value = WorkerOutput(
            [RequestOutput(7, [], error="request registration failed")]
        )
        context = Mock()
        socket = context.socket.return_value
        socket.recv.side_effect = [
            msgpack.packb(message)
            for message in (
                {"type": "init"},
                {"type": "register", "request_id": 7, "prompt_token_ids": [248320]},
                {
                    "type": "execute",
                    "rpc_id": 12,
                    "step_id": 1,
                    "scheduled": [],
                    "finished_request_ids": [],
                    "preempted_request_ids": [],
                    "num_batched_tokens": 0,
                },
                {"type": "shutdown"},
            )
        ]
        with (
            patch("oh_my_vllm.worker.zmq_bridge.zmq.Context", return_value=context),
            patch("oh_my_vllm.worker.zmq_bridge._handle_init", return_value=worker),
            patch("oh_my_vllm.worker.zmq_bridge.configure_logging"),
        ):
            serve("ipc:///unused-test-socket")
        replies = [
            msgpack.unpackb(call.args[0], raw=False)
            for call in socket.send.call_args_list
        ]
        self.assertEqual(replies[1]["type"], "execute_result")
        self.assertEqual(replies[1]["rpc_id"], 12)
        self.assertEqual(
            replies[1]["outputs"][0]["error"], "request registration failed"
        )
        worker.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
