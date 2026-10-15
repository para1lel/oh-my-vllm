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
from tests.support.fp64 import specials

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_special_storage_comparison_rejects_independent_class_and_sign_faults(index):
    expected = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
    specials(expected.clone(), expected)
    faulty = expected.clone()
    faulty[index] = 0.0
    with pytest.raises(AssertionError):
        specials(faulty, expected)


@pytest.mark.parametrize(
    ("value", "gate", "expected"),
    [
        (0.390625, -8, 0.0001316070556640625),
        (-0.390625, -8, -0.0001316070556640625),
    ],
)
def test_attention_gate_reference_keeps_intermediate_bf16_storage(
    value, gate, expected
):
    actual = oracles.operator("elementwise", "attention_gate")(
        torch.tensor([value], dtype=torch.bfloat16),
        torch.tensor([gate], dtype=torch.bfloat16),
    )
    assert actual.item() == expected


def test_prequantized_reference_uses_stored_scales_without_requantization():
    encoded = torch.full((2, 128), 2, dtype=torch.float8_e4m3fn)
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    activation_scale = torch.full((1, 2), 0.25).T
    weight_scale = torch.full((1, 1), 0.125)
    result = oracles.groupwise(encoded, weight, activation_scale, weight_scale)
    assert torch.equal(result, torch.full((2, 128), 8, dtype=torch.bfloat16))


def test_gdn_prepare_reference_combines_projections_gates_and_history():
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.full((16384, 128), 0.0625).to(torch.float8_e4m3fn)
    scale = torch.full((128, 1), 0.125)
    ba_weight = torch.zeros(96, 128, dtype=torch.bfloat16)
    conv = torch.zeros(10240, 4, dtype=torch.bfloat16)
    conv[:, -1] = 1
    pool = torch.zeros(2, 10240, 3, dtype=torch.bfloat16)
    mixed, z, decay, beta = oracles.gdn_prepare(
        x,
        weight,
        scale,
        ba_weight,
        conv,
        pool,
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([0, 2]),
        torch.tensor([0]),
        torch.tensor([-1, 1]),
        torch.zeros(48),
        torch.zeros(48),
    )
    assert torch.equal(mixed, torch.full_like(mixed, 0.73046875))
    assert torch.equal(z, torch.ones_like(z))
    torch.testing.assert_close(
        decay.double(),
        torch.full((2, 48), -0.6931471805599453, dtype=torch.float64),
        atol=2e-8,
        rtol=0,
    )
    assert torch.equal(beta, torch.full_like(beta, 0.5))
    assert not pool[0].count_nonzero() and not pool[1, :, 0].count_nonzero()
    assert torch.equal(pool[1, :, 1:], torch.ones_like(pool[1, :, 1:]))


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


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("column", [False, True])
def test_quantization_oracle_keeps_exact_codes_and_single_row_scale_layout(
    rows, column
):
    import torch

    from development.kernels.oracles import quantize

    x = torch.zeros(rows, 256, dtype=torch.bfloat16)
    x[:, 0] = 448
    x[:, 1] = 1
    encoded, scale = quantize(x, column_major=column)
    assert encoded.float()[:, 0].tolist() == [448] * rows
    assert encoded.float()[:, 1].tolist() == [1] * rows
    assert scale.stride() == ((1, rows) if column else (2, 1))
    assert scale[:, 0].tolist() == [1] * rows


def test_prequantized_groupwise_raw_reference_retains_fp64_low_bits():
    encoded = torch.full((1, 128), 1.5, dtype=torch.float8_e4m3fn)
    weight = torch.full((128, 128), 1.75, dtype=encoded.dtype)
    activation_scale = torch.full((1, 1), 0.1)
    weight_scale = torch.full((1, 1), 0.3)
    raw = oracles.groupwise(
        encoded, weight, activation_scale, weight_scale, dtype=torch.float64
    )
    expected = 336 * float(activation_scale[0, 0]) * float(weight_scale[0, 0])
    assert raw.dtype == torch.float64
    torch.testing.assert_close(raw, torch.full_like(raw, expected), rtol=1e-15, atol=0)
    assert not torch.equal(raw, raw.float().double())


def test_quantization_reference_keeps_prescribed_fp32_division_before_fp8():
    x = torch.zeros(1, 128, dtype=torch.float32)
    x[0, :2] = torch.tensor([1, 0.9642857313156128])
    encoded, scale = oracles.quantize(x)
    assert scale[0, 0].item() == 0.0022321429569274187
    assert encoded.view(torch.uint8)[0, 1].item() == 126
    raw_quotient = x[0, 1].double() / scale[0, 0].double()
    assert raw_quotient < 432
    assert raw_quotient.float().item() == 432
