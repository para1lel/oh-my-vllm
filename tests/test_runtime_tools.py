"""CPU-only checks for diagnostic logs and cooperative GPU selection."""

import json
import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from oh_my_vllm.worker.logging_utils import JsonFormatter

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_log_correlation_and_timestamp(self):
        record = logging.LogRecord(
            "oh_my_vllm.worker", logging.DEBUG, "", 0, "step", (), None
        )
        record.fields = {"step_id": 42, "execute_host_us": 1.5}
        result = json.loads(JsonFormatter().format(record))
        self.assertEqual(result["step_id"], 42)
        self.assertTrue(result["timestamp"].endswith("+00:00"))
        self.assertEqual(result["execute_host_us"], 1.5)

    def test_gpu_selector_skips_busy_and_preserves_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            mock = Path(directory) / "nvidia-smi"
            mock.write_text(
                '#!/bin/bash\ncase "$1" in\n'
                '--query-gpu=*) echo "GPU-test-busy, NVIDIA B200, 0, 0"; '
                'echo "GPU-test-idle, NVIDIA B200, 0, 0";;\n'
                "*) echo GPU-test-busy;;\nesac\n"
            )
            mock.chmod(0o755)
            # The outer full suite pins a real GPU. This subprocess tests an
            # independent mock inventory and must select from that inventory.
            environment = dict(os.environ)
            environment.pop("OH_MY_VLLM_GPU_UUID", None)
            environment["PATH"] = directory + ":" + os.environ["PATH"]
            result = subprocess.run(
                [
                    str(ROOT / "scripts/with-gpu.sh"),
                    "bash",
                    "-c",
                    'read -r input; echo "$CUDA_VISIBLE_DEVICES:$input"; exit 7',
                ],
                env=environment,
                input="stdin-preserved\n",
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 7)
            self.assertEqual(result.stdout.strip(), "GPU-test-idle:stdin-preserved")


if __name__ == "__main__":
    unittest.main()
