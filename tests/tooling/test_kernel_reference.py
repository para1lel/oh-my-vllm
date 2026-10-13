"""Numerical references and runtime identity stay independent of operator providers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from development.kernels import oracles
from development.kernels.reference import (
    source_hashes,
    validate_native_inputs,
    verify_reference,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("selected", [None, "cuda"])
def test_backend_identity_is_fixed_at_process_start(selected):
    environment = dict(os.environ)
    environment.pop("OH_MY_VLLM_KERNEL_BACKEND", None)
    if selected is not None:
        environment["OH_MY_VLLM_KERNEL_BACKEND"] = selected
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from oh_my_vllm.kernels.backend import NAME; print(NAME)",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "cuda"


def test_unknown_provider_fails_before_loading_model():
    result = subprocess.run(
        [sys.executable, "-c", "import oh_my_vllm.kernels.backend"],
        env=os.environ | {"OH_MY_VLLM_KERNEL_BACKEND": "unsupported"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown OH_MY_VLLM_KERNEL_BACKEND" in result.stderr


def test_reference_identity_includes_independent_mathematics_and_lock():
    record = verify_reference()
    assert record["kind"] == "independent_torch_mathematics"
    assert "development/kernels/oracles.py" in record["files"]
    assert (
        record["files"]["development/kernels/oracles.py"]
        == source_hashes()["development/kernels/oracles.py"]
    )
    assert len(record["runtime_requirements_sha256"]) == 64


def test_chunk_reference_matches_closed_form_delta_update():
    q = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0], k[..., 0] = 1, 1
    values = torch.ones_like(q)
    output, state = oracles.prefill(
        q,
        k,
        values,
        torch.zeros(1, 1),
        torch.full((1, 1), 0.5),
        torch.zeros(1, 1, 128, 128),
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(state[0, 0, :, 0], torch.full((128,), 0.5))
    torch.testing.assert_close(state[..., 1:], torch.zeros_like(state[..., 1:]))
    torch.testing.assert_close(output, torch.full_like(values, 0.5 / 128**0.5))


@pytest.mark.parametrize("eps", [None, 0.125])
def test_dspark_oracles_keep_public_epsilon_default_and_bf16_boundaries(eps):
    x = torch.ones(2, 1, 128, dtype=torch.bfloat16)
    weight = torch.ones(128)
    kwargs = {} if eps is None else {"eps": eps}
    expected = torch.full_like(x, (1 + (1e-6 if eps is None else eps)) ** -0.5)
    rms = oracles.operator("dspark_attention", "rms_norm")
    norm_rope = oracles.operator("dspark_attention", "normalize_rope")
    hidden = torch.ones(2, 5120, dtype=torch.bfloat16)
    hidden_weight = torch.ones(5120)
    hidden_expected = torch.full_like(
        hidden, (1 + (1e-6 if eps is None else eps)) ** -0.5
    )
    torch.testing.assert_close(
        rms(hidden, hidden_weight, **kwargs), hidden_expected, rtol=0, atol=0
    )
    actual = norm_rope(
        x, weight, torch.zeros(2, dtype=torch.int32), torch.ones(64), 1.0, **kwargs
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["headers_sha256", "flags", "project_headers"])
def test_native_collection_rejects_template_or_build_input_drift(changed):
    expected = {
        "headers_sha256": "header-closure",
        "flags": ["-O3"],
        "project_headers": {"owned.cuh": "owned-header"},
    }
    loaded = dict(expected)
    validate_native_inputs(loaded, expected)
    loaded[changed] = [] if changed == "flags" else "changed"
    with pytest.raises(RuntimeError, match="native GEMM inputs differ"):
        validate_native_inputs(loaded, expected)
