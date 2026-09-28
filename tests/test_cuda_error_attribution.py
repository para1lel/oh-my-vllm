"""CUDA debug-mode fault attribution and non-consuming launch checks."""

import os
import subprocess
import sys

import pytest


def test_debug_sync_requires_eager_before_loading(monkeypatch):
    from oh_my_vllm.kernels.cuda_backend import compiled

    monkeypatch.setenv("OH_MY_VLLM_CUDA_DEBUG_SYNC", "1")
    monkeypatch.delenv("OH_MY_VLLM_ENFORCE_EAGER", raising=False)
    with pytest.raises(RuntimeError, match="requires OH_MY_VLLM_ENFORCE_EAGER=1"):
        compiled.__wrapped__()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "scenario",
    (
        "prior_launch",
        "normal_preserves_error",
        "prior_execution",
        "current_execution",
        "eager",
        "capture_debug",
        "capture_normal",
    ),
)
def test_cuda_error_attribution_in_isolated_process(scenario):
    debug = scenario not in ("capture_normal", "normal_preserves_error")
    env = dict(os.environ)
    env["OH_MY_VLLM_CUDA_DEBUG_SYNC"] = "1" if debug else "0"
    env["OH_MY_VLLM_ENFORCE_EAGER"] = "1" if debug else "0"
    result = subprocess.run(
        [sys.executable, "-m", "tests.gpu_cuda_error_case", scenario],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"passed {scenario}" in result.stdout
