import tempfile
import unittest
from pathlib import Path
from queue import Empty
from unittest.mock import patch

from oh_my_vllm.models.deepseek_v41.launcher import GroupWorker


class FakeProcess:
    def __init__(self):
        self.alive = True
        self.calls = []
        self.join_calls = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.join_calls += 1
        self.calls.append(("join", timeout))

    def terminate(self):
        self.calls.append(("terminate",))

    def kill(self):
        self.calls.append(("kill",))
        self.alive = False


class DeadProcess:
    exitcode = 1

    def is_alive(self):
        return False


class EmptyResults:
    def get(self, timeout):
        raise Empty


class ErrorResults:
    def get(self, timeout):
        return 2, {
            "type": "error",
            "message": "CUDA initialization failed",
        }


class Reader:
    def poll(self, timeout):
        return True

    def recv(self):
        return 2, {
            "type": "error",
            "message": "rank setup failed",
        }


class ReaderResults:
    _reader = Reader()


class EmptyThenErrorResults:
    def __init__(self):
        self.calls = 0

    def get(self, timeout):
        self.calls += 1
        if self.calls == 1:
            raise Empty
        return 2, {
            "type": "error",
            "message": "late rank setup failed",
        }


class ReadyReader:
    def recv(self):
        return 0, {
            "type": "error",
            "message": "device-side assert",
        }


class ExitedProcess:
    exitcode = 9


class DeepSeekLauncher(unittest.TestCase):
    def test_collect_fails_when_a_rank_exits_before_reply(self):
        worker = GroupWorker.__new__(GroupWorker)
        worker.results = EmptyResults()
        worker.processes = [DeadProcess()]

        with self.assertRaisesRegex(RuntimeError, "rank exited before reply"):
            worker._collect()

    def test_collect_reports_rank_error_before_waiting_for_other_ranks(self):
        worker = GroupWorker.__new__(GroupWorker)
        worker.results = ErrorResults()
        worker.processes = []

        with self.assertRaisesRegex(
            RuntimeError, "rank 2 returned an error: CUDA initialization failed"
        ):
            worker._collect()

    def test_collect_polls_the_result_pipe_with_a_timeout(self):
        worker = GroupWorker.__new__(GroupWorker)
        worker.results = ReaderResults()
        worker.processes = []

        with self.assertRaisesRegex(
            RuntimeError, "rank 2 returned an error: rank setup failed"
        ):
            worker._collect()

    def test_collect_keeps_polling_until_a_rank_replies(self):
        worker = GroupWorker.__new__(GroupWorker)
        worker.results = EmptyThenErrorResults()
        worker.processes = []

        with self.assertRaisesRegex(
            RuntimeError, "rank 2 returned an error: late rank setup failed"
        ):
            worker._collect()

    def test_collect_keeps_a_ready_rank_error_when_the_process_has_exited(self):
        worker = GroupWorker.__new__(GroupWorker)
        reader = ReadyReader()
        worker.result_readers = [reader]
        worker.processes = [ExitedProcess()]

        with (
            patch(
                "oh_my_vllm.models.deepseek_v41.launcher.wait_connections",
                return_value=[reader],
            ),
            self.assertRaisesRegex(
                RuntimeError, "rank 0 returned an error: device-side assert"
            ),
        ):
            worker._collect()

    def test_close_uses_bounded_process_cleanup_and_removes_store_directory(self):
        worker = GroupWorker.__new__(GroupWorker)
        process = FakeProcess()
        store_directory = tempfile.TemporaryDirectory()
        worker.commands = []
        worker.processes = [process]
        worker.store_directory = store_directory
        worker.store_path = str(Path(store_directory.name) / "store")
        worker._closed = False

        worker.close()

        self.assertEqual(
            process.calls,
            [("join", 30), ("terminate",), ("join", 5), ("kill",), ("join", 5)],
        )
        self.assertFalse(Path(store_directory.name).exists())

        worker.close()
        self.assertEqual(
            process.calls,
            [("join", 30), ("terminate",), ("join", 5), ("kill",), ("join", 5)],
        )


if __name__ == "__main__":
    unittest.main()
