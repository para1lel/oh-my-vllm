"""Gated RMS fusion preserves head rounding, packed reads, and replay storage."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir import normalized_linear as ir
from oh_my_vllm.ir.fp8 import _native_linear
from oh_my_vllm.ir.pointwise import _native_rms_norm

from development.kernels import oracles
from tests.support.fp64 import close, cpu, specials


def inputs(rows, device, columns=256):
    generator = torch.Generator(device=device).manual_seed(821 + rows)
    packed_x = torch.randn(rows, 8192, device=device, generator=generator).bfloat16()
    x = packed_x[:, :6144].view(rows, 48, 128)
    packed = torch.randn(rows, 16384, device=device, generator=generator).bfloat16()
    gate = packed[:, 10240:].view(rows, 48, 128)
    gamma = torch.rand(128, device=device, generator=generator) + 0.5
    weight = torch.randn(columns, 6144, device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(columns // 128, 48, device=device, generator=generator) * 0.01
    return x, gamma, gate, weight, scale


def assert_fp64_projection(args, actual):
    reference_args = cpu(args) if len(args[0]) <= 33 else args
    expected = oracles.gated_norm_linear(*reference_args)
    close(actual, expected, rtol=0.03, atol=0.03)


def test_native_keeps_gated_head_rounding_before_quantization():
    args = inputs(2, "cpu")
    normalized = _native_rms_norm(*args[:2], 1e-6, args[2])
    expected = _native_linear(normalized.flatten(1), *args[3:], False)
    # Exact native composition stays separate from FP64 numerical accuracy.
    actual = ir._gated_native(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_fp64_projection(args, actual)


def test_native_schema_fake_and_compilation(monkeypatch):
    args = inputs(2, "cpu")
    # CPU opcheck cannot compare FP8 input tensors. Float reference weights
    # preserve the explicit FP8 activation rounding and block-scale semantics.
    args = (*args[:3], args[3].float(), args[4])
    torch.library.opcheck(
        ir._gated_native,
        args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    monkeypatch.setattr(ir._GATED_OP, "priority", ("native",))
    expected = ir._gated_native(*args)
    actual = compile_forward(ir.gated_norm_fp8_linear)(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_fp64_projection(args, expected)
    assert_fp64_projection(args, actual)


GPU = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def replay_stream():
    from oh_my_vllm.kernels.cuda_backend import groupwise

    with pytest.MonkeyPatch.context() as context:
        context.setattr(groupwise, "_WORKSPACES", {})
        yield torch.cuda.Stream()
        torch.cuda.synchronize()


@GPU[0]
@GPU[1]
@pytest.mark.parametrize("rows", [1, 8, 32, 33, 624, 2496, 8192, 32143, 32144, 32290])
def test_projection_matches_previous_cuda_chain(rows):
    from oh_my_vllm.kernels import fp8, normalization

    args = inputs(rows, "cuda", columns=5120)
    normalized = normalization.rms_norm(args[0], args[1], gate=args[2])
    expected = fp8.linear(normalized.flatten(1), *args[3:])
    actual = ir.gated_norm_fp8_linear(*args)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert_fp64_projection(args, actual)
    assert actual.shape == (rows, 5120) and actual.is_contiguous()
    if rows <= 33:
        compiled = compile_forward(ir.gated_norm_fp8_linear)(*args)
        assert torch.equal(compiled.view(torch.int16), expected.view(torch.int16))
        assert_fp64_projection(args, compiled)


@GPU[0]
@GPU[1]
@pytest.mark.parametrize("column", [False, True])
def test_side_stream_replay_reads_changed_packed_inputs(column, replay_stream):
    from oh_my_vllm.kernels import normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, gamma, gate, weight, weight_scale = inputs(8, "cuda")
    data = torch.empty((8, 6144), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((48, 8) if column else (8, 48), device="cuda")
    if column:
        scales = scales.T

    def quantize():
        compiled().gated_quantize(x, gamma, gate, data, scales, column)

    replay_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(replay_stream):
        quantize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize()
    torch.cuda.current_stream().wait_stream(replay_stream)
    for value in (0.0, -2.0, 0.5):
        x.fill_(value)
        gamma.mul_(0.75)
        gate.add_(0.125)
        before = [t.contiguous().view(torch.uint8).clone() for t in (x, gamma, gate)]
        reference_args = cpu((x, gamma, gate, weight, weight_scale))
        independent_norm = oracles.rms(*reference_args[:2], gate=reference_args[2])
        producer_norm = normalization.rms_norm(x, gamma, gate=gate)
        close(producer_norm, independent_norm, rtol=0.03, atol=0.03)
        # Encoding truth uses only the given BF16 boundary and FP64 equations.
        expected_data, expected_scales = oracles.quantize(
            cpu(producer_norm).flatten(1), column_major=column
        )
        expected_projection = oracles.gated_norm_linear(*reference_args)
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        close(data.view(torch.uint8), expected_data.view(torch.uint8), rtol=0, atol=0)
        close(
            scales.view(torch.int32), expected_scales.view(torch.int32), rtol=0, atol=0
        )
        actual_projection = oracles.groupwise(
            *cpu((data, weight, scales, weight_scale))
        )
        close(actual_projection, expected_projection, rtol=0.03, atol=0.03)
        for actual, expected in zip((x, gamma, gate), before, strict=True):
            assert torch.equal(actual.contiguous().view(torch.uint8), expected)


@GPU[0]
@GPU[1]
@pytest.mark.parametrize(
    "scenario",
    ["cpu", "dtype", "short", "stride", "scale_layout", "input_alias", "output_alias"],
)
def test_direct_fusion_rejects_bad_storage_before_writes(scenario):
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, gamma, gate, _, _ = inputs(8, "cuda")
    data = torch.full((8, 6144), -1, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.full((8, 48), -1, device="cuda")
    if scenario == "cpu":
        gamma = gamma.cpu()
    elif scenario == "dtype":
        gamma = gamma.bfloat16()
    elif scenario == "short":
        scales = scales[:, :-1]
    elif scenario == "stride":
        gate = gate.transpose(1, 2).contiguous().transpose(1, 2)
    elif scenario == "scale_layout":
        scales = scales.T.contiguous().T
    elif scenario == "input_alias":
        data = x.view(torch.float8_e4m3fn).as_strided((8, 6144), (6144, 1))
    elif scenario == "output_alias":
        scales = data.view(torch.float32).as_strided((8, 48), (48, 1))
    before = [
        t.contiguous().view(torch.uint8).clone() for t in (x, gamma, gate, data, scales)
    ]
    with pytest.raises(Exception, match="gated quantization"):
        compiled().gated_quantize(x, gamma, gate, data, scales, False)
    torch.cuda.synchronize()
    for actual, expected in zip((x, gamma, gate, data, scales), before, strict=True):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected)


@GPU[0]
@GPU[1]
def test_fusion_keeps_extreme_and_nonfinite_gates():
    from oh_my_vllm.kernels import normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, gamma, gate, _, _ = inputs(8, "cuda")
    x[0, 0, :2] = float("inf")
    x[1, 0, :2] = float("nan")
    gate[2].fill_(torch.finfo(torch.bfloat16).max)
    gate[3].fill_(-torch.finfo(torch.bfloat16).max)
    gate[4, 0, :2] = float("inf")
    gate[5, 0, :2] = float("nan")
    raw_x, raw_gamma, raw_gate = cpu((x, gamma, gate))
    independent_norm = oracles.rms(raw_x, raw_gamma, gate=raw_gate)
    producer_norm = normalization.rms_norm(x, gamma, gate=gate)
    close(producer_norm, independent_norm, rtol=0.03, atol=0.03, equal_nan=True)
    specials(producer_norm, independent_norm)
    expected_data, expected_scales = oracles.quantize(cpu(producer_norm).flatten(1))
    data = torch.empty((8, 6144), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 48), device="cuda")
    compiled().gated_quantize(x, gamma, gate, data, scales, False)
    close(data.view(torch.uint8), expected_data.view(torch.uint8), rtol=0, atol=0)
    close(scales.view(torch.int32), expected_scales.view(torch.int32), rtol=0, atol=0)


@GPU[0]
@GPU[1]
def test_fused_quantization_keeps_independent_negative_zero_codes():
    from oh_my_vllm.kernels import normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x = torch.ones((1, 48, 128), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(128, device="cuda")
    gate = torch.full_like(x, -100)
    expected_norm = oracles.rms(*cpu((x, gamma)), gate=cpu(gate))
    assert torch.equal(expected_norm, torch.zeros_like(expected_norm))
    assert torch.signbit(expected_norm).all()
    specials(normalization.rms_norm(x, gamma, gate=gate), expected_norm)
    data = torch.empty((1, 6144), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((1, 48), device="cuda")
    compiled().gated_quantize(x, gamma, gate, data, scales, False)
    # Literal storage witnesses come from the input, never a production result.
    assert (data.view(torch.uint8) == 0x80).all()
    assert (scales == 2.2321428288629014e-13).all()
