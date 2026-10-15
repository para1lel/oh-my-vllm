"""FP8 base and fused IR operations preserve the block-scale contract."""

import pytest
import torch
from oh_my_vllm.ir import applied_rewrites, compile_forward, fp8, operations
from oh_my_vllm.ir.pointwise import silu_mul

from development.kernels import oracles
from tests.support.fp64 import close, cpu


def test_fp8_reference_uses_128_element_blocks():
    x = torch.arange(1, 257, dtype=torch.float32).reshape(1, 256)
    data, scales = fp8._native_quantize(x, False, False)
    assert data.shape == x.shape
    assert scales.shape == (1, 2)
    torch.testing.assert_close(scales, torch.tensor([[128 / 448, 256 / 448]]))
    assert data.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    for name in ("fp8_quantize", "fp8_linear"):
        assert name in operations()


def test_fp8_reference_keeps_nonfinite_cuda_semantics():
    x = torch.zeros(3, 128, dtype=torch.bfloat16)
    x[0, :2] = torch.tensor([float("inf"), float("-inf")])
    x[1] = float("nan")
    x[2, :2] = torch.tensor([float("nan"), 1.0])
    actual, scales = fp8._native_quantize(x, False, False)
    expected, expected_scales = oracles.quantize(x)
    close(scales, expected_scales, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("silu_gate", [False, True])
def test_fp8_linear_compiled_matches_independent_fp64(silu_gate):
    width = 5120
    x = torch.randn(
        1, width * (2 if silu_gate else 1), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(256, width, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.rand(2, width // 128, device="cuda") * 0.01
    expected = oracles.linear(*cpu((x, weight, weight_scale)), silu_gate=silu_gate)
    eager = fp8.linear(x, weight, weight_scale, silu_gate=silu_gate)
    compiled = compile_forward(fp8.linear)(x, weight, weight_scale, silu_gate=silu_gate)
    for actual in (eager, compiled):
        close(actual, expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fp8_quantize_compiled_keeps_column_major_scale_layout():
    x = torch.randn(2, 5120, device="cuda", dtype=torch.bfloat16)
    expected_data, expected_scales = oracles.quantize(cpu(x), column_major=True)
    data, scales = compile_forward(fp8.quantize)(x, column_major=True)
    assert torch.equal(data.cpu().view(torch.uint8), expected_data.view(torch.uint8))
    close(scales, expected_scales, rtol=0, atol=0)
    assert scales.stride() == expected_scales.stride()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_single_use_silu_fp8_linear_pattern_rewrites_to_fused_semantics():
    x = torch.randn(1, 10240, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 5120, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(2, 40, device="cuda") * 0.01
    expected = oracles.linear(*cpu((x, weight, scale)), silu_gate=True)
    before = len(applied_rewrites())

    def forward(packed, projection, projection_scale):
        return fp8.linear(silu_mul(packed), projection, projection_scale)

    actual = compile_forward(forward)(x, weight, scale)
    close(actual, expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(actual, forward(x, weight, scale), rtol=0, atol=0)
    assert len(applied_rewrites()) == before + 1
    assert applied_rewrites()[-1].pattern == "silu_mul+fp8_linear"
