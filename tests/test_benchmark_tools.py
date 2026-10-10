"""CPU regression checks for benchmark process ownership and configuration."""

import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "benchmarks/common.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class BenchmarkTests(unittest.TestCase):
    def test_timeout_terminates_worker_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            code = (
                "import subprocess,sys,time; "
                "from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                compare.run_engine(
                    [sys.executable, "-c", code, str(pid_file)], timeout=1
                )
            pid = int(pid_file.read_text())
            # Orphan zombies may await PID 1; they consume no GPU/CPU resources.
            state = Path(f"/proc/{pid}/stat")
            if state.exists():
                self.assertIn(state.read_text().split(") ")[1][0], ("Z", "X"))

    def test_isolated_worker_group_is_owned_by_exact_ancestry(self):
        code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
            "start_new_session=True); print(p.pid,flush=True); time.sleep(60)"
        )
        with compare._child_subreaper():
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            ownership = compare.ProcessOwnership(process)
            try:
                child = int(process.stdout.readline())
                self.assertNotEqual(os.getpgid(child), process.pid)
                with patch.object(
                    compare.subprocess,
                    "check_output",
                    return_value=f"GPU-own, {child}\nGPU-other, 12\n",
                ):
                    compare.assert_gpu_exclusive(
                        "GPU-own", process.pid, ownership=ownership
                    )
                self.assertTrue(ownership.owns(child))
            finally:
                compare._stop_engine(process, ownership)
                ownership.close()
                process.stdout.close()
            self.assertFalse(Path(f"/proc/{child}").exists())

    def test_same_uid_and_group_does_not_prove_ownership(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        ownership = compare.ProcessOwnership(os.getpid())
        try:
            # Bind an unrelated root, rather than this test's parent process.
            root = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"]
            )
            ownership.close()
            ownership = compare.ProcessOwnership(root)
            try:
                self.assertEqual(os.getpgid(foreign.pid), os.getpgid(root.pid))
                with (
                    patch.object(
                        compare.subprocess,
                        "check_output",
                        return_value=f"GPU-own, {foreign.pid}\n",
                    ),
                    self.assertRaisesRegex(RuntimeError, "external process"),
                ):
                    compare.assert_gpu_exclusive(
                        "GPU-own", root.pid, ownership=ownership
                    )
            finally:
                root.terminate()
                root.wait(timeout=5)
        finally:
            ownership.close()
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_birth_reuse_cannot_signal_or_accept_gpu_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"],
            start_new_session=True,
        )
        ownership = compare.ProcessOwnership(process)
        original = compare._process_record

        def reused(pid):
            record = original(pid)
            if pid == process.pid and record is not None:
                record = dict(record, start_ticks=record["start_ticks"] + 1)
            return record

        try:
            with (
                patch.object(compare, "_process_record", side_effect=reused),
                patch.object(compare.signal, "pidfd_send_signal") as send,
            ):
                self.assertFalse(ownership.owns(process.pid))
                self.assertFalse(ownership.signal(ownership.root, signal.SIGKILL))
                self.assertFalse(ownership._session_owned(process.pid))
                send.assert_not_called()
                with (
                    patch.object(
                        compare.subprocess,
                        "check_output",
                        return_value=f"GPU-own, {process.pid}\n",
                    ),
                    self.assertRaisesRegex(RuntimeError, "external process"),
                ):
                    compare.assert_gpu_exclusive(
                        "GPU-own", process.pid, ownership=ownership
                    )
            self.assertIsNone(process.poll())
        finally:
            ownership.close()
            process.terminate()
            process.wait(timeout=5)

    def test_libc_pidfd_fallback_binds_real_identity(self):
        with patch.object(compare.os, "pidfd_open", None, create=True):
            fd = compare._open_pidfd(os.getpid())
            os.close(fd)

    def test_binding_race_closes_unpublished_pidfd(self):
        ownership = compare.ProcessOwnership(os.getpid())
        before = len(list(Path("/proc/self/fd").iterdir()))
        record = compare._process_record(os.getpid())
        ownership.records.clear()
        try:
            with patch.object(compare, "_process_record", side_effect=[record, None]):
                self.assertIsNone(ownership._bind(os.getpid()))
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
        finally:
            ownership.close()


if __name__ == "__main__":
    unittest.main()
