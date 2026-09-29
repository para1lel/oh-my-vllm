"""CPU semantic and schema checks for the first model call sites routed via IR."""

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.ir import compile_forward, compiled_graph_counts, operations, pointwise


def test_model_pointwise_ops_are_registered():
    assert {
        "silu_mul",
        "delta_gates",
        "rms_norm",
        "add_rms_norm",
    } <= operations().keys()


def test_silu_semantics_preserve_gate_rounding():
    packed = torch.randn(3, 256, dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)
    expected = F.silu(gate.float()).bfloat16() * up
    actual = pointwise._native_silu_mul(packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_and_norm_semantics():
    ba = torch.randn(3, 96, dtype=torch.bfloat16)
    a_log = torch.randn(48)
    bias = torch.randn(48)
    decay, beta = pointwise._native_delta_gates(ba, a_log, bias)
    b, a = ba.float().chunk(2, dim=-1)
    torch.testing.assert_close(decay, -a_log.exp() * F.softplus(a + bias))
    torch.testing.assert_close(beta, b.sigmoid())

    x = torch.randn(3, 128, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128)
    gate = torch.randn_like(x)
    normalized = pointwise._native_rms_norm(x, weight, 1e-6, gate)
    values = x.float()
    expected = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        * weight
        * F.silu(gate.float())
    ).bfloat16()
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    summed, normalized = pointwise._native_add_rms_norm(x, residual, weight)
    expected_sum = (x.float() + residual.float()).bfloat16()
    torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
    torch.testing.assert_close(
        normalized,
        pointwise._native_rms_norm(summed, weight, 1e-6, None),
        rtol=0,
        atol=0,
    )


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
        expected = expected_sum, expected_norm, expected_decay, expected_beta
        eager = forward(packed, ba, a_log, bias, x, weight)
        compiled = compiled_forward(packed, ba, a_log, bias, x, weight)
        for actual, reference in zip(
            (*eager, *compiled), (*expected, *expected), strict=True
        ):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        if rows == 3:
            warmed = compiled_graph_counts()[label]
        elif rows > 3:
            assert compiled_graph_counts()[label] == warmed
