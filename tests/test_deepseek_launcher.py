import tempfile
import unittest
from pathlib import Path
from queue import Empty

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


class DeepSeekLauncher(unittest.TestCase):
    def test_collect_fails_when_a_rank_exits_before_reply(self):
        worker = GroupWorker.__new__(GroupWorker)
        worker.results = EmptyResults()
        worker.processes = [DeadProcess()]

        with self.assertRaisesRegex(RuntimeError, "rank exited before reply"):
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
