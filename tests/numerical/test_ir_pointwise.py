"""CPU semantic and schema checks for the first model call sites routed via IR."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward, compiled_graph_counts, operations, pointwise

from development.kernels import oracles
from tests.support.fp64 import close, cpu


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="GPU required"
                ),
            ],
        ),
    ],
)
def test_pointwise_keeps_observable_bf16_intermediate_stores(device):
    packed = torch.empty(1, 256, device=device, dtype=torch.bfloat16)
    packed[:, :128] = -8
    packed[:, 128:] = 0.390625
    expected = oracles.silu_mul(cpu(packed))
    assert expected.unique().tolist() == [-0.0010528564453125]
    silu = pointwise._native_silu_mul if device == "cpu" else pointwise.silu_mul
    results = [silu(packed)]
    if device == "cuda":
        results.append(compile_forward(silu)(packed))
    for actual in results:
        close(actual, expected, rtol=0, atol=0)

    x = torch.ones(1, 128, device=device, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    residual[:, ::2] = 3 / 256
    weight = torch.ones(128, device=device)
    expected_sum, expected_norm = oracles.add_rms(*cpu((x, residual, weight)))
    assert expected_sum[0, :2].tolist() == [1.015625, 1]
    assert expected_norm[0, :2].tolist() == [1.0078125, 0.9921875]
    add_rms = (
        pointwise._native_add_rms_norm if device == "cpu" else pointwise.add_rms_norm
    )
    results = [add_rms(x, residual, weight)]
    if device == "cuda":
        results.append(compile_forward(add_rms)(x, residual, weight))
    for summed, normalized in results:
        close(summed, expected_sum, rtol=0, atol=0)
        close(normalized, expected_norm, rtol=0, atol=0)


def test_silu_semantics_preserve_gate_rounding():
    packed = torch.randn(3, 256, dtype=torch.bfloat16)
    expected = oracles.silu_mul(packed)
    actual = pointwise._native_silu_mul(packed)
    close(actual, expected, rtol=0.008, atol=0.0001)


def test_gate_and_norm_semantics():
    ba = torch.randn(3, 96, dtype=torch.bfloat16)
    a_log = torch.randn(48)
    bias = torch.randn(48)
    decay, beta = pointwise._native_delta_gates(ba, a_log, bias)
    expected_decay, expected_beta = oracles.gates(ba, a_log, bias)
    close(decay, expected_decay, rtol=2e-6, atol=2e-6)
    close(beta, expected_beta, rtol=1e-6, atol=1e-7)

    x = torch.randn(3, 128, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128)
    gate = torch.randn_like(x)
    normalized = pointwise._native_rms_norm(x, weight, 1e-6, gate)
    expected = oracles.rms(x, weight, gate=gate)
    close(normalized, expected, rtol=0.03, atol=0.03)
    summed, normalized = pointwise._native_add_rms_norm(x, residual, weight)
    expected_sum, expected_norm = oracles.add_rms(x, residual, weight)
    close(summed, expected_sum, rtol=0, atol=0)
    close(normalized, expected_norm, rtol=0.03, atol=0.03)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("silu_mul", (torch.randn(2, 256, dtype=torch.bfloat16),)),
        (
            "delta_gates",
            (
                torch.randn(2, 96, dtype=torch.bfloat16),
                torch.randn(48),
                torch.randn(48),
            ),
        ),
        (
            "rms_norm",
            (torch.randn(2, 128, dtype=torch.bfloat16), torch.randn(128), 1e-6, None),
        ),
        (
            "add_rms_norm",
            (
                torch.randn(2, 128, dtype=torch.bfloat16),
                torch.randn(2, 128, dtype=torch.bfloat16),
                torch.randn(128),
            ),
        ),
    ],
)
def test_native_custom_op_fake_and_schema(name, args):
    result = torch.library.opcheck(
        operations()[name].reference,
        args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert set(result.values()) == {"SUCCESS"}


def test_default_provider_fails_closed_without_cuda():
    with pytest.raises(RuntimeError, match="no supported IR implementation"):
        pointwise.silu_mul(torch.randn(2, 256, dtype=torch.bfloat16))


@pytest.mark.parametrize("gated", [False, True])
def test_native_rms_noncontiguous_input_contract_and_compile(monkeypatch, gated):
    x = torch.randn(128, 3, dtype=torch.bfloat16).T
    weight = torch.randn(128)
    gate = torch.randn_like(x) if gated else None
    args = (x, weight, 1e-6, gate)
    result = torch.library.opcheck(
        pointwise._native_rms_norm,
        args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert set(result.values()) == {"SUCCESS"}
    monkeypatch.setattr(pointwise._RMS, "priority", ("native",))

    def forward(x, weight, gate):
        return pointwise.rms_norm(x, weight, gate=gate).reshape(-1)

    expected = forward(x, weight, gate)
    actual = compile_forward(forward)(x, weight, gate)
    assert pointwise._native_rms_norm(*args).is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    independent = oracles.rms(x, weight, gate=gate)
    close(expected.reshape(x.shape), independent, rtol=0.03, atol=0.03)
    close(actual.reshape(x.shape), independent, rtol=0.03, atol=0.03)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_pointwise_model_path_compiles_with_cuda_provider():
    from oh_my_vllm.kernels import elementwise, normalization

    a_log = torch.randn(48, device="cuda")
    bias = torch.randn(48, device="cuda")
    weight = torch.randn(128, device="cuda")

    def forward(packed, ba, a_log, bias, x, weight):
        gate = pointwise.silu_mul(packed)
        decay, beta = pointwise.delta_gates(ba, a_log, bias)
        value = pointwise.rms_norm(x, weight, gate=gate)
        summed, normalized = pointwise.add_rms_norm(value, x, weight)
        return summed, normalized, decay, beta

    label = "test_dynamic_cuda_pointwise"
    compiled_forward = compile_forward(forward, unit=label)
    warmed = None
    for rows in (2, 3, 5, 7):
        packed = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        ba = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        expected_gate = elementwise.silu_mul(packed)
        expected_decay, expected_beta = elementwise.delta_gates(ba, a_log, bias)
        expected_value = normalization.rms_norm(x, weight, gate=expected_gate)
        expected_sum, expected_norm = normalization.add_rms_norm(
            expected_value, x, weight
        )
        # The unfused path checks execution equivalence, never numerical quality.
        unfused = expected_sum, expected_norm, expected_decay, expected_beta
        raw_packed, raw_ba, raw_a_log, raw_bias, raw_x, raw_weight = cpu(
            (packed, ba, a_log, bias, x, weight)
        )
        reference_gate = oracles.silu_mul(raw_packed)
        reference_decay, reference_beta = oracles.gates(raw_ba, raw_a_log, raw_bias)
        reference_value = oracles.rms(raw_x, raw_weight, gate=reference_gate)
        reference_sum, reference_norm = oracles.add_rms(
            reference_value, raw_x, raw_weight
        )
        expected = reference_sum, reference_norm, reference_decay, reference_beta
        eager = forward(packed, ba, a_log, bias, x, weight)
        compiled = compiled_forward(packed, ba, a_log, bias, x, weight)
        for actual, equivalent in zip(
            (*eager, *compiled), (*unfused, *unfused), strict=True
        ):
            torch.testing.assert_close(actual, equivalent, rtol=0, atol=0)
        for actual in (eager, compiled):
            for value, reference in zip(actual[:2], expected[:2], strict=True):
                close(value, reference, rtol=0.03, atol=0.03)
            close(actual[2], expected[2], rtol=2e-6, atol=2e-6)
            close(actual[3], expected[3], rtol=1e-6, atol=1e-7)
        if rows == 3:
            warmed = compiled_graph_counts()[label]
        elif rows > 3:
            assert compiled_graph_counts()[label] == warmed
