"""CPU regression checks for benchmark process ownership and configuration."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare", ROOT / "benchmarks/compare_vllm.py"
)
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

    def test_contention_detection_ignores_other_gpus_and_own_group(self):
        with (
            patch.object(
                compare.subprocess,
                "check_output",
                return_value="GPU-own, 11\nGPU-other, 12\n",
            ),
            patch.object(compare.os, "getpgid", return_value=77),
        ):
            compare.assert_gpu_exclusive("GPU-own", 77)
        with (
            patch.object(
                compare.subprocess, "check_output", return_value="GPU-own, 11\n"
            ),
            patch.object(compare.os, "getpgid", return_value=88),
            self.assertRaisesRegex(RuntimeError, "external process 11"),
        ):
            compare.assert_gpu_exclusive("GPU-own", 77)

    def test_numeric_gpu_id_cannot_bypass_contention_monitor(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks/compare_vllm.py")],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("idle GPU UUID", result.stderr)

    def test_zero_drafts_rejected_before_loading_model(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/compare_vllm.py"),
                "--mode",
                "mtp",
                "--speculative-tokens",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("MTP requires positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
