"""Fused projection keeps residual and normalized BF16 rounding boundaries."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir import normalized_linear as ir
from oh_my_vllm.ir.fp8 import _native_linear
from oh_my_vllm.ir.pointwise import _native_add_rms_norm


def inputs(rows, device):
    generator = torch.Generator(device=device).manual_seed(791 + rows)
    x = torch.randn(rows, 5120, device=device, generator=generator).bfloat16()
    residual = torch.randn(rows, 5120, device=device, generator=generator).bfloat16()
    gamma = torch.rand(5120, device=device, generator=generator) + 0.5
    weight = torch.randn(256, 5120, device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(2, 40, device=device, generator=generator) * 0.01
    return x, residual, gamma, weight, scale


def test_native_projection_keeps_rounded_intermediate_values():
    args = inputs(2, "cpu")
    summed, normalized = _native_add_rms_norm(*args[:3])
    expected = _native_linear(normalized, *args[3:], False)
    result = ir._native(*args)
    torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected, rtol=0, atol=0)


def test_native_schema_fake_and_compilation(monkeypatch):
    args = inputs(2, "cpu")
    # SchemaCheckMode uses allclose on inputs; its CPU FP8 comparison is absent.
    # The native reference accepts float weights and rounds FP8 activations itself.
    args = (*args[:3], args[3].float(), args[4])
    torch.library.opcheck(
        ir._native,
        args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    monkeypatch.setattr(ir._OP, "priority", ("native",))
    expected = ir._native(*args)
    result = compile_forward(ir.add_norm_fp8_linear)(*args)
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("rows", [1, 8, 32, 33, 624, 2048, 4095, 4096, 8192, 32144])
def test_cuda_projection_and_residual_match_unfused_path(rows):
    from oh_my_vllm.kernels import fp8, normalization

    args = inputs(rows, "cuda")
    summed, normalized = normalization.add_rms_norm(*args[:3])
    expected = fp8.linear(normalized, *args[3:])
    result = ir.add_norm_fp8_linear(*args)
    torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected, rtol=0, atol=0)
    if rows <= 33:
        result = compile_forward(ir.add_norm_fp8_linear)(*args)
        torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
        torch.testing.assert_close(result[1], expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("column", [False, True])
def test_captured_quantization_reads_changed_inputs_and_keeps_exact_scales(column):
    from oh_my_vllm.kernels import fp8, normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, residual, gamma, _, _ = inputs(8, "cuda")
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((40, 8) if column else (8, 40), device="cuda")
    if column:
        scales = scales.T
    summed = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled().rms_quantize(x, residual, gamma, data, scales, summed, column)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled().rms_quantize(x, residual, gamma, data, scales, summed, column)
    torch.cuda.current_stream().wait_stream(stream)
    for factor in (0.0, -2.0, 0.5):
        x.fill_(factor)
        residual.add_(0.125)
        gamma.mul_(0.75)
        expected_sum, normalized = normalization.add_rms_norm(x, residual, gamma)
        expected_data, expected_scales = fp8.quantize(normalized, column_major=column)
        graph.replay()
        torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
        assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "scenario",
    ["cpu", "dtype", "short", "stride", "scale_layout", "alignment", "alias"],
)
def test_direct_quantization_rejects_bad_storage_without_writes(scenario):
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, residual, gamma, _, _ = inputs(8, "cuda")
    data = torch.full(x.shape, -1, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.full((8, 40), -1, device="cuda", dtype=torch.float32)
    summed = torch.full_like(x, -1)
    if scenario == "cpu":
        gamma = gamma.cpu()
    elif scenario == "dtype":
        gamma = gamma.bfloat16()
    elif scenario == "short":
        scales = scales[:, :-1]
    elif scenario == "stride":
        residual = residual.T.contiguous().T
    elif scenario == "scale_layout":
        scales = scales.T.contiguous().T
    elif scenario == "alignment":
        gamma = torch.empty(5121, device="cuda")[1:]
    elif scenario == "alias":
        summed = x
    before = [tensor.clone() for tensor in (data, scales, summed)]
    messages = {
        "cpu": "device mismatch",
        "dtype": "BF16 inputs, FP8 data, and FP32 weights/scales",
        "short": "model-width matrices and scales",
        "stride": "contiguous matrices",
        "scale_layout": "scale layout mismatch",
        "alignment": "offset or alignment is invalid",
        "alias": "outputs overlap inputs or each other",
    }
    with pytest.raises(Exception, match=messages[scenario]):
        compiled().rms_quantize(x, residual, gamma, data, scales, summed, False)
    torch.cuda.synchronize()
    for actual, expected in zip((data, scales, summed), before, strict=True):
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_quantization_keeps_nonfinite_values_and_residual_alias_input():
    from oh_my_vllm.kernels import fp8, normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, _, gamma, _, _ = inputs(8, "cuda")
    x[0, :2] = float("inf")
    x[1, :2] = float("nan")
    summed, normalized = normalization.add_rms_norm(x, x, gamma)
    expected, expected_scales = fp8.quantize(normalized)
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 40), device="cuda")
    result_sum = torch.empty_like(x)
    compiled().rms_quantize(x, x, gamma, data, scales, result_sum, False)
    torch.testing.assert_close(result_sum, summed, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(data.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0, equal_nan=True)
