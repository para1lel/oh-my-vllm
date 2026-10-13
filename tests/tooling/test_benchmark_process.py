"""Failed and interrupted measurements retain their child output on disk."""

import errno
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[2] / "benchmarks/common.py"
spec = importlib.util.spec_from_file_location("benchmark_process", path)
process_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_module)


class BenchmarkProcessTest(unittest.TestCase):
    def test_pidfd_einval_after_child_reap_does_not_publish_ownership(self):
        ownership = process_module.ProcessOwnership(os.getpid())
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:

            def reap_before_open(pid):
                self.assertEqual(pid, child.pid)
                child.terminate()
                child.wait(timeout=5)
                raise OSError(errno.EINVAL, "pidfd_open failed")

            with patch.object(
                process_module, "_open_pidfd", side_effect=reap_before_open
            ):
                self.assertIsNone(ownership._bind(child.pid, parent=ownership.root))
            self.assertNotIn(
                child.pid, {row["pid"] for row in ownership.records.values()}
            )
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
            ownership.close()

    def test_pidfd_einval_for_existing_birth_still_fails(self):
        ownership = process_module.ProcessOwnership(os.getpid())
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with (
                patch.object(
                    process_module,
                    "_open_pidfd",
                    side_effect=OSError(errno.EINVAL, "pidfd_open failed"),
                ),
                self.assertRaises(OSError) as failure,
            ):
                ownership._bind(child.pid, parent=ownership.root)
            self.assertEqual(failure.exception.errno, errno.EINVAL)
            self.assertIsNone(child.poll())
            self.assertNotIn(
                child.pid, {row["pid"] for row in ownership.records.values()}
            )
            self.assertIn(str(child.pid), " ".join(failure.exception.__notes__))
        finally:
            child.terminate()
            child.wait(timeout=5)
            ownership.close()

    def test_pidfd_einval_with_reused_pid_does_not_bind_new_birth(self):
        ownership = process_module.ProcessOwnership(os.getpid())
        old = dict(ownership.root, pid=99999999, start_ticks=1)
        reused = dict(old, start_ticks=2)
        try:
            with (
                patch.object(
                    process_module, "_process_record", side_effect=[old, reused]
                ),
                patch.object(
                    process_module,
                    "_open_pidfd",
                    side_effect=OSError(errno.EINVAL, "pidfd_open failed"),
                ),
            ):
                self.assertIsNone(ownership._bind(old["pid"], parent=ownership.root))
            self.assertEqual(len(ownership.records), 1)
            self.assertEqual(len(ownership.pidfds), 1)
        finally:
            ownership.close()

    def test_other_pidfd_errors_remain_fatal_after_child_reap(self):
        ownership = process_module.ProcessOwnership(os.getpid())
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:

            def reap_before_open(pid):
                child.terminate()
                child.wait(timeout=5)
                raise OSError(errno.EMFILE, "pidfd_open failed")

            with (
                patch.object(
                    process_module, "_open_pidfd", side_effect=reap_before_open
                ),
                self.assertRaises(OSError) as failure,
            ):
                ownership._bind(child.pid, parent=ownership.root)
            self.assertEqual(failure.exception.errno, errno.EMFILE)
            self.assertNotIn(
                child.pid, {row["pid"] for row in ownership.records.values()}
            )
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
            ownership.close()

    def assert_audit_released(self, log, expected):
        audit = json.loads(Path(str(log) + ".processes.json").read_text())
        self.assertEqual(audit["live"], [])
        self.assertEqual(audit["unverified_adopted"], [])
        owned = {item["pid"] for item in audit["processes"]}
        self.assertTrue(set(expected) <= owned, audit)
        for pid in expected:
            self.assertFalse(Path(f"/proc/{pid}").exists())
        return audit

    def test_timeout_reaps_isolated_worker_and_compiler(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "isolated.log"
            child = (
                "import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import time;time.sleep(60)']);"
                "print(p.pid,flush=True);time.sleep(60)"
            )
            root = (
                "import subprocess,sys,time;"
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],"
                "start_new_session=True);print(p.pid,flush=True);time.sleep(60)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                process_module.run_engine(
                    [sys.executable, "-c", root, child],
                    timeout=0.6,
                    log_path=log,
                )
            worker, compiler = map(int, log.read_text().splitlines())
            audit = self.assert_audit_released(log, [worker, compiler])
            records = {item["pid"]: item for item in audit["processes"]}
            self.assertEqual(records[worker]["pgid"], worker)
            self.assertEqual(records[compiler]["session"], worker)

    def test_failed_root_reaps_already_proved_setsid_child(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "abrupt.log"
            code = (
                "import subprocess,sys,time,os;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
                "start_new_session=True);print(p.pid,flush=True);"
                "time.sleep(.2);os._exit(1)"
            )
            with self.assertRaises(subprocess.CalledProcessError):
                process_module.run_engine([sys.executable, "-c", code], log_path=log)
            self.assert_audit_released(log, [int(log.read_text())])

    def test_cancel_reaps_worker_with_independent_process_group(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "cancel.log"
            code = (
                "import subprocess,sys,time;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
                "start_new_session=True);print(p.pid,flush=True);time.sleep(60)"
            )
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                process_module.run_engine(
                    [sys.executable, "-c", code],
                    log_path=log,
                    cancelled=lambda: log.exists() and bool(log.read_text()),
                )
            self.assert_audit_released(log, [int(log.read_text())])

    def test_root_exit_with_inherited_stdout_reaps_without_timeout(self):
        code = (
            "import subprocess,sys,time,os;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
            "start_new_session=True);print(p.pid,flush=True);"
            "time.sleep(.2);os._exit(1)"
        )
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            self.assertRaises(subprocess.CalledProcessError) as failure,
        ):
            process_module.run_engine([sys.executable, "-c", code], timeout=3)
        child = int(failure.exception.output)
        self.assertFalse(Path(f"/proc/{child}").exists())

    def test_fast_fork_orphan_inherits_bound_root_session(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "fork-orphan.log"
            code = """import os,time
if os.fork() == 0:
    compiler = os.fork()
    if compiler == 0:
        time.sleep(60)
        os._exit(0)
    print(compiler,flush=True)
    os._exit(0)
time.sleep(.15)
os._exit(1)
"""
            with self.assertRaises(subprocess.CalledProcessError):
                process_module.run_engine([sys.executable, "-c", code], log_path=log)
            compiler = int(log.read_text())
            audit = self.assert_audit_released(log, [compiler])
            self.assertTrue(
                any(item.get("adopted_session") for item in audit["signals"])
            )

    def test_bound_child_delayed_setsid_then_orphan_is_reaped(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "delayed-setsid.log"
            child = """import os,time
time.sleep(.25)
os.setsid()
compiler = os.fork()
if compiler == 0:
    time.sleep(60)
    os._exit(0)
print(compiler,flush=True)
os._exit(0)
"""
            root = (
                "import subprocess,sys,time,os;"
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]);"
                "print(p.pid,flush=True);time.sleep(.6);os._exit(1)"
            )
            with self.assertRaises(subprocess.CalledProcessError):
                process_module.run_engine(
                    [sys.executable, "-c", root, child], log_path=log
                )
            worker, compiler = map(int, log.read_text().splitlines())
            audit = self.assert_audit_released(log, [worker, compiler])
            records = {item["pid"]: item for item in audit["processes"]}
            self.assertEqual(records[worker]["session"], worker)
            self.assertEqual(records[compiler]["session"], worker)
            self.assertTrue(
                any(item.get("adopted_session") == worker for item in audit["signals"])
            )

    def test_existing_foreign_child_is_preserved_as_baseline(self):
        foreign = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"],
            start_new_session=True,
        )
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            ):
                log = Path(directory) / "foreign.log"
                self.assertEqual(
                    process_module.run_engine(
                        [sys.executable, "-c", "print('ok')"], log_path=log
                    ),
                    "ok\n",
                )
                audit = self.assert_audit_released(log, [])
                self.assertNotIn(
                    foreign.pid, {item["pid"] for item in audit["processes"]}
                )
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_unproved_new_foreign_session_fails_cleanup_without_signal(self):
        with process_module._child_subreaper():
            root = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                start_new_session=True,
            )
            ownership = process_module.ProcessOwnership(
                root, adopt_orphans=True, baseline=set()
            )
            foreign = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                start_new_session=True,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "unproved adopted"):
                    process_module._stop_engine(root, ownership)
                self.assertIsNone(foreign.poll())
                self.assertFalse(ownership.owns(foreign.pid))
                self.assertFalse(
                    ownership.signal(
                        process_module._process_record(foreign.pid), signal.SIGKILL
                    )
                )
                self.assertEqual(
                    [item["pid"] for item in ownership.unverified_adopted],
                    [foreign.pid],
                )
            finally:
                ownership.close()
                foreign.terminate()
                foreign.wait(timeout=5)
                if root.poll() is None:
                    root.kill()
                root.wait(timeout=5)

    def test_external_gpu_interference_keeps_log(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}),
            patch.object(
                process_module,
                "assert_gpu_exclusive",
                side_effect=[None, RuntimeError("external GPU process")],
            ),
        ):
            log = Path(directory) / "interference.log"
            with self.assertRaisesRegex(RuntimeError, "external GPU"):
                process_module.run_engine(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('partial run', flush=True); time.sleep(30)",
                    ],
                    log_path=log,
                )
            self.assertEqual(log.read_text(), "partial run\n")

    def test_failed_process_keeps_log(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "failed.log"
            with self.assertRaises(subprocess.CalledProcessError):
                process_module.run_engine(
                    [
                        sys.executable,
                        "-c",
                        "print('failure evidence'); raise SystemExit(1)",
                    ],
                    log_path=log,
                )
            self.assertEqual(log.read_text(), "failure evidence\n")

    def test_timeout_keeps_log_and_stops_process(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            log = Path(directory) / "timeout.log"
            with self.assertRaises(subprocess.TimeoutExpired):
                process_module.run_engine(
                    [
                        sys.executable,
                        "-c",
                        "import os,time; print(os.getpid(), flush=True); "
                        "time.sleep(30)",
                    ],
                    timeout=0.5,
                    log_path=log,
                )
            pid = int(log.read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
