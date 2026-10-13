"""Gated RMS fusion preserves head rounding, packed reads, and replay storage."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir import normalized_linear as ir
from oh_my_vllm.ir.fp8 import _native_linear
from oh_my_vllm.ir.pointwise import _native_rms_norm


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


def test_native_keeps_gated_head_rounding_before_quantization():
    args = inputs(2, "cpu")
    normalized = _native_rms_norm(*args[:2], 1e-6, args[2])
    expected = _native_linear(normalized.flatten(1), *args[3:], False)
    torch.testing.assert_close(ir._gated_native(*args), expected, rtol=0, atol=0)


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
    assert actual.shape == (rows, 5120) and actual.is_contiguous()
    if rows <= 33:
        compiled = compile_forward(ir.gated_norm_fp8_linear)(*args)
        assert torch.equal(compiled.view(torch.int16), expected.view(torch.int16))


@GPU[0]
@GPU[1]
@pytest.mark.parametrize("column", [False, True])
def test_side_stream_replay_reads_changed_packed_inputs(column, replay_stream):
    from oh_my_vllm.kernels import fp8, normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, gamma, gate, _, _ = inputs(8, "cuda")
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
        normalized = normalization.rms_norm(x, gamma, gate=gate)
        expected_data, expected_scales = fp8.quantize(
            normalized.flatten(1), column_major=column
        )
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))
        assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
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
    from oh_my_vllm.kernels import fp8, normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    x, gamma, gate, _, _ = inputs(8, "cuda")
    x[0, 0, :2] = float("inf")
    x[1, 0, :2] = float("nan")
    gate[2].fill_(torch.finfo(torch.bfloat16).max)
    gate[3].fill_(-torch.finfo(torch.bfloat16).max)
    gate[4, 0, :2] = float("inf")
    gate[5, 0, :2] = float("nan")
    normalized = normalization.rms_norm(x, gamma, gate=gate)
    expected_data, expected_scales = fp8.quantize(normalized.flatten(1))
    data = torch.empty((8, 6144), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 48), device="cuda")
    compiled().gated_quantize(x, gamma, gate, data, scales, False)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
