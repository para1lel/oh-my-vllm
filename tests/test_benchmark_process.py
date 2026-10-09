"""Failed and interrupted measurements retain their child output on disk."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "benchmarks/common.py"
spec = importlib.util.spec_from_file_location("benchmark_process", path)
process_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_module)


class BenchmarkProcessTest(unittest.TestCase):
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
