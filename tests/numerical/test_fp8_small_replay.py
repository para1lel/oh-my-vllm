"""Small projection padding must clear stale rows and preserve live graph inputs."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("pdl", [False, True])
def test_small_projection_replays_data_scales_and_zero_padding(pdl, monkeypatch):
    from oh_my_vllm.kernels import fp8
    from oh_my_vllm.kernels.cuda_backend import compiled, groupwise

    monkeypatch.setattr(groupwise, "execution_policy", lambda: SimpleNamespace(pdl=pdl))
    data = torch.randn(4, 5120, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(4, 40, device="cuda") * 0.01
    weight = torch.randn(256, 5120, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.rand(2, 40, device="cuda") * 0.01
    padded = torch.empty(64, 5120, device="cuda", dtype=data.dtype)
    padded_scale = torch.empty(64, 40, device="cuda")

    def projection():
        compiled().pad_quantized(data, scale, padded, padded_scale)
        return fp8._project_quantized(data, scale, weight, weight_scale)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        projection()
        projection()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = projection()
    for multiplier in (0.5, 1.25):
        torch.cuda.current_stream().wait_stream(stream)
        data.copy_(torch.randn_like(data.float()).to(data.dtype))
        scale.mul_(multiplier)
        weight_scale.mul_(multiplier)
        padded.view(torch.uint8).fill_(127)
        padded_scale.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(padded[:4].view(torch.uint8), data.view(torch.uint8))
        torch.testing.assert_close(padded_scale[:4], scale, rtol=0, atol=0)
        assert torch.count_nonzero(padded[4:].view(torch.uint8)).item() == 0
        assert torch.count_nonzero(padded_scale[4:]).item() == 0
        activation = data.float() * scale.repeat_interleave(128, dim=1)
        weights = weight.float() * weight_scale.repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        torch.testing.assert_close(
            actual.float(), activation @ weights.T, rtol=0.01, atol=0.01
        )
    del graph
    torch.cuda.synchronize()


def test_small_padding_rejects_aliased_output_before_launch():
    from oh_my_vllm.kernels.cuda_backend import compiled

    padded = torch.empty(64, 5120, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(4, 40, device="cuda")
    padded_scale = torch.empty(64, 40, device="cuda")
    with pytest.raises(Exception, match="Check failed"):
        compiled().pad_quantized(padded[:4], scale, padded, padded_scale)


@pytest.mark.parametrize("rows", [33, 63, 64, 65])
def test_tail_tiles_preserve_output_canaries_and_live_scales(rows):
    from oh_my_vllm.kernels.cuda_backend import groupwise

    data = torch.randn(rows, 512, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(256, 512, device="cuda").to(data.dtype)
    scale = torch.rand(rows, 4, device="cuda") * 0.01
    weight_scale = torch.rand(2, 4, device="cuda") * 0.01
    storage = torch.full((rows + 2, 256), 19, device="cuda", dtype=torch.bfloat16)
    output = storage[1:-1]
    workspace = groupwise._workspace(data.device)
    module = groupwise.compiled()

    def projection():
        module.groupwise_fp8(
            data, weight, scale, weight_scale, output, workspace, 1, True, True
        )

    projection()
    projection()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        projection()
    for multiplier in (0.5, 1.25):
        data.copy_(torch.randn_like(data.float()).to(data.dtype))
        scale.mul_(multiplier)
        weight_scale.mul_(multiplier)
        output.fill_(float("nan"))
        graph.replay()
        activation = data.float() * scale.repeat_interleave(128, dim=1)
        weights = weight.float() * weight_scale.repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        torch.testing.assert_close(
            output.float(), activation @ weights.T, rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(
            storage[[0, -1]], torch.full_like(storage[[0, -1]], 19), rtol=0, atol=0
        )
    del graph
    torch.cuda.synchronize()
