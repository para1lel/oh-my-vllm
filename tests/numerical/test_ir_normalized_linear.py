"""Fused projection keeps residual and normalized BF16 rounding boundaries."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir import normalized_linear as ir
from oh_my_vllm.ir.fp8 import _native_linear
from oh_my_vllm.ir.pointwise import _native_add_rms_norm

from development.kernels import oracles
from tests.support.fp64 import close, cpu, specials


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


def assert_fp64_projection(args, actual):
    # Large comparisons remain individual fused operators. FP64 Torch math
    # is independent of the production norm/quantization/CUTLASS chain.
    reference_args = cpu(args) if len(args[0]) <= 33 else args
    summed, projection = oracles.add_norm_linear(*reference_args)
    close(actual[0], summed, rtol=0, atol=0)
    close(actual[1], projection, rtol=0.03, atol=0.03)


def test_native_projection_keeps_rounded_intermediate_values():
    args = inputs(2, "cpu")
    summed, normalized = _native_add_rms_norm(*args[:3])
    # Exact equality below checks native composition, not numerical accuracy.
    expected = _native_linear(normalized, *args[3:], False)
    result = ir._native(*args)
    torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected, rtol=0, atol=0)
    assert_fp64_projection(args, result)


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
    assert_fp64_projection(args, expected)
    assert_fp64_projection(args, result)


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
    assert_fp64_projection(args, result)
    if rows <= 33:
        result = compile_forward(ir.add_norm_fp8_linear)(*args)
        torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
        torch.testing.assert_close(result[1], expected, rtol=0, atol=0)
        assert_fp64_projection(args, result)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("rows", [1, 2, 4, 6, 12, 24, 32])
def test_small_projection_replays_changed_inputs_without_padding_or_mutation(rows):
    from oh_my_vllm.kernels import fp8, normalization

    args = inputs(rows, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8.add_norm_linear(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fp8.add_norm_linear(*args)
    torch.cuda.current_stream().wait_stream(stream)
    for factor in (0.0, -2.0, 0.5):
        args[0].fill_(factor)
        args[1].add_(0.125)
        args[2].mul_(0.75)
        args[4].mul_(1.25)
        before = [tensor.view(torch.uint8).clone() for tensor in args]
        summed, normalized = normalization.add_rms_norm(*args[:3])
        expected = fp8.linear(normalized, *args[3:])
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
        torch.testing.assert_close(result[1], expected, rtol=0.03, atol=0.03)
        assert_fp64_projection(args, result)
        assert result[1].shape == (rows, args[3].shape[0])
        assert result[1].is_contiguous()
        for tensor, protected in zip(args, before, strict=True):
            assert torch.equal(tensor.view(torch.uint8), protected)
    torch.cuda.synchronize()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("rows", [2, 17])
@pytest.mark.parametrize("offset_input", [0, 1, 2])
def test_small_projection_keeps_contiguous_storage_offset_inputs(rows, offset_input):
    from oh_my_vllm.kernels import fp8, normalization

    args = list(inputs(rows, "cuda"))
    original = args[offset_input]
    storage = torch.full(
        (original.numel() + 2,), -7, dtype=original.dtype, device="cuda"
    )
    args[offset_input] = storage[1:-1].view(original.shape)
    args[offset_input].copy_(original)
    before = [tensor.view(torch.uint8).clone() for tensor in (*args, storage)]
    summed, normalized = normalization.add_rms_norm(*args[:3])
    expected = fp8.linear(normalized, *args[3:])
    result = fp8.add_norm_linear(*args)
    torch.testing.assert_close(result[0], summed, rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected, rtol=0, atol=0)
    assert_fp64_projection(args, result)
    for tensor, protected in zip((*args, storage), before, strict=True):
        assert torch.equal(tensor.view(torch.uint8), protected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("column", [False, True])
def test_captured_quantization_reads_changed_inputs_and_keeps_exact_scales(column):
    from oh_my_vllm.kernels import normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, residual, gamma, weight, weight_scale = inputs(8, "cuda")
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
        reference_args = cpu((x, residual, gamma, weight, weight_scale))
        expected_sum, independent_norm = oracles.add_rms(*reference_args[:3])
        _, producer_norm = normalization.add_rms_norm(x, residual, gamma)
        close(producer_norm, independent_norm, rtol=0.03, atol=0.03)
        # Given this BF16 producer boundary, encoding is independently exact.
        # It is an execution property, not the full-fused numerical oracle.
        expected_data, expected_scales = oracles.quantize(
            cpu(producer_norm), column_major=column
        )
        expected_projection = oracles.add_norm_linear(*reference_args)[1]
        graph.replay()
        close(summed, expected_sum, rtol=0, atol=0)
        close(data.view(torch.uint8), expected_data.view(torch.uint8), rtol=0, atol=0)
        close(scales, expected_scales, rtol=0, atol=0)
        actual_projection = oracles.groupwise(
            *cpu((data, weight, scales, weight_scale))
        )
        close(actual_projection, expected_projection, rtol=0.03, atol=0.03)


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
    from oh_my_vllm.kernels import normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, _, gamma, _, _ = inputs(8, "cuda")
    x[0, :2] = float("inf")
    x[1, :2] = float("nan")
    summed, independent_norm = oracles.add_rms(*cpu((x, x, gamma)))
    _, producer_norm = normalization.add_rms_norm(x, x, gamma)
    close(producer_norm, independent_norm, rtol=0.03, atol=0.03, equal_nan=True)
    specials(producer_norm, independent_norm)
    expected, expected_scales = oracles.quantize(cpu(producer_norm))
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 40), device="cuda")
    result_sum = torch.empty_like(x)
    compiled().rms_quantize(x, x, gamma, data, scales, result_sum, False)
    close(result_sum, summed, rtol=0, atol=0, equal_nan=True)
    specials(result_sum, summed)
    close(data.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    close(scales, expected_scales, rtol=0, atol=0, equal_nan=True)
