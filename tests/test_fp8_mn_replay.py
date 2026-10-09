"""Large scale-layout changes must keep graph inputs live and padding private."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def replay_stream():
    return torch.cuda.Stream()


@pytest.mark.parametrize("rows", [32145, 32290])
@pytest.mark.parametrize("pdl", [False, True])
def test_graph_reads_changed_original_scales_and_inputs(
    rows, pdl, replay_stream, monkeypatch
):
    from oh_my_vllm.kernels import fp8
    from oh_my_vllm.kernels.cuda_backend import groupwise

    monkeypatch.setattr(groupwise, "execution_policy", lambda: SimpleNamespace(pdl=pdl))
    generator = torch.Generator(device="cuda").manual_seed(931 + rows)
    x = torch.randn(rows, 5120, device="cuda", generator=generator).bfloat16()
    residual = torch.randn_like(x)
    gamma = torch.rand(5120, device="cuda") + 0.5
    weight = torch.randn(34816, 5120, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(272, 40, device="cuda") * 0.01
    weight_before = weight.view(torch.uint8).clone()

    def projection():
        return fp8.add_norm_linear(x, residual, gamma, weight, scale)

    replay_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(replay_stream):
        projection()
        projection()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = projection()
    torch.cuda.current_stream().wait_stream(replay_stream)
    for step in range(2):
        x.mul_(-0.75)
        residual.add_(0.125)
        gamma.mul_(0.9)
        scale.mul_(1.25 + step)
        scale_before = scale.clone()
        # The previous row-major path is an independent layout reference.
        with monkeypatch.context() as context:
            context.setattr(fp8, "_large_mn_projection", lambda *_: False)
            expected = projection()
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert result[1].shape == (rows, 34816)
        assert result[1].is_contiguous()
        assert torch.equal(scale, scale_before)
        assert torch.equal(weight.view(torch.uint8), weight_before)


def test_pitched_scale_output_preserves_tail_canary():
    from oh_my_vllm.kernels import fp8, normalization
    from oh_my_vllm.kernels.cuda_backend import compiled

    rows, physical = 33, 36
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gamma = torch.rand(5120, device="cuda") + 0.5
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_storage = torch.full((40, physical), -7.0, device="cuda")
    scales = scale_storage.T[:rows]
    summed = torch.empty_like(x)
    compiled().rms_quantize(x, residual, gamma, data, scales, summed, True)
    expected_sum, normalized = normalization.add_rms_norm(x, residual, gamma)
    expected_data, expected_scales = fp8.quantize(normalized, column_major=True)
    torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
    assert torch.equal(data.view(torch.uint8), expected_data.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    assert torch.all(scale_storage[:, rows:] == -7.0)


@pytest.mark.parametrize("fault", ["row_layout", "short_shape", "unaligned_leading"])
def test_direct_mn_rejects_invalid_layout_before_output_writes(fault):
    from oh_my_vllm.kernels.cuda_backend.groupwise import compiled

    rows = 32145 if fault == "unaligned_leading" else 624
    data = torch.empty(rows, 5120, device="cuda", dtype=torch.float8_e4m3fn)
    weight = torch.empty(34816, 5120, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.empty(40, rows, device="cuda").T
    sb = torch.empty(40, 272, device="cuda").T
    output = torch.full((rows, 34816), 1.0, device="cuda", dtype=torch.bfloat16)
    if fault == "row_layout":
        sa = sa.contiguous()
    workspace = torch.empty(16, device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="MN scale"):
        compiled().groupwise_fp8(
            data, weight, sa, sb, output, workspace, 2, True, False
        )
    assert torch.all(output == 1.0)


@pytest.mark.parametrize("pitch", [32, 37])
def test_rms_rejects_overlapping_or_excessive_pitch_without_writes(pitch):
    from oh_my_vllm.kernels.cuda_backend import compiled

    rows = 33
    x = torch.ones(rows, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    gamma = torch.ones(5120, device="cuda")
    data = torch.full_like(x, -1, dtype=torch.float8_e4m3fn)
    summed = torch.full_like(x, -2)
    storage = torch.full((40 * pitch + 1,), -7.0, device="cuda")
    scales = storage.as_strided((rows, 40), (1, pitch))
    with pytest.raises(RuntimeError, match="scale layout mismatch"):
        compiled().rms_quantize(x, residual, gamma, data, scales, summed, True)
    assert torch.all(data.float() == -1)
    assert torch.all(summed == -2)
    assert torch.all(storage == -7)
