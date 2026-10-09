"""The GPU wrapper pins the requested idle device and rejects invalid selection."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory(tmp_path):
    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        " *query-compute-apps*) exit 0 ;;\n"
        " *) printf 'GPU-test-one, NVIDIA B200, 0, 0\\n"
        "GPU-test-two, NVIDIA B200, 0, 0\\n' ;;\nesac\n"
    )
    executable.chmod(0o755)
    return dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_requested_idle_uuid_is_pinned(tmp_path):
    env = inventory(tmp_path)
    env["OH_MY_VLLM_GPU_UUID"] = "GPU-test-two"
    lock = Path("/tmp/oh-my-vllm-GPU-test-two.lock")
    try:
        result = subprocess.run(
            [str(ROOT / "scripts/with-gpu.sh"), "printenv", "CUDA_VISIBLE_DEVICES"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        assert result.stdout.strip() == "GPU-test-two"
    finally:
        lock.unlink(missing_ok=True)


def test_missing_uuid_fails_before_waiting(tmp_path):
    env = inventory(tmp_path)
    env["OH_MY_VLLM_GPU_UUID"] = "GPU-test-missing"
    result = subprocess.run(
        [str(ROOT / "scripts/with-gpu.sh"), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "not in the device inventory" in result.stderr
