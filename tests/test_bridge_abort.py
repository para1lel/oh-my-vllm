"""The legacy abort message must reach the model runner's finished lifecycle."""

import unittest
from unittest.mock import Mock, patch

import msgpack
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
        worker.execute_model.assert_called_once()
        notification = worker.execute_model.call_args.args[0]
        self.assertEqual(notification.finished_request_ids, [7])
        self.assertEqual(notification.scheduled, [])
        worker.unregister_request.assert_not_called()
        self.assertEqual(socket.send.call_count, 1)  # Only the init ready reply.
        worker.shutdown.assert_called_once()
        socket.close.assert_called_once()
        context.term.assert_called_once()


if __name__ == "__main__":
    unittest.main()
