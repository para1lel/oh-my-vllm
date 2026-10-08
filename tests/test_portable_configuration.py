"""Deployment configuration must work outside the current checkout location."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment_priority_and_relative_repository_paths(tmp_path):
    explicit = tmp_path / "explicit"
    active = tmp_path / "active"
    for prefix in (explicit, active):
        (prefix / "bin").mkdir(parents=True)
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "VIRTUAL_ENV": str(active),
        "CONDA_PREFIX": str(tmp_path / "other"),
        "OH_MY_VLLM_CONDA_PREFIX": str(explicit),
        "UV_PYTHON": str(tmp_path / "unrelated/bin/python"),
    }
    command = [
        str(ROOT / "scripts/with-env.sh"),
        "bash",
        "-c",
        'printf "%s\\n" "$PATH" "$PYTHONPATH" "$CARGO_TARGET_DIR" '
        '"$OH_MY_VLLM_WORKER_PYTHON" "$TVM_FFI_CACHE_DIR" "$UV_PYTHON"',
    ]
    rows = subprocess.check_output(
        command, env=environment, cwd=tmp_path, text=True
    ).splitlines()
    assert rows[0].startswith(str(explicit / "bin") + ":")
    assert rows[1] == str(ROOT / "python")
    assert rows[2] == str(ROOT / "target")
    assert rows[3] == str(explicit / "bin/python")
    assert rows[4].startswith(str(tmp_path / ".cache"))
    assert rows[5] == str(explicit / "bin/python")
    del environment["OH_MY_VLLM_CONDA_PREFIX"]
    active_rows = subprocess.check_output(
        command, env=environment, text=True
    ).splitlines()
    assert active_rows[0].startswith(str(active / "bin") + ":")
    assert active_rows[5] == str(active / "bin/python")


def test_invalid_explicit_environment_fails_before_command(tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "OH_MY_VLLM_CONDA_PREFIX": str(tmp_path / "missing"),
    }
    result = subprocess.run(
        [str(ROOT / "scripts/with-env.sh"), "true"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "environment does not exist" in result.stderr
