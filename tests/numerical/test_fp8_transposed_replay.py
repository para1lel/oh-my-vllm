"""Narrow projections retain live scales, caller streams, and output bounds."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("rows", [1, 4, 20, 32])
@pytest.mark.parametrize("columns", [34816, 16384, 14336])
@pytest.mark.parametrize("pdl", [False, True])
def test_transposed_projection_replays_live_inputs_and_scales(
    rows, columns, pdl, monkeypatch
):
    from oh_my_vllm.kernels import fp8
    from oh_my_vllm.kernels.cuda_backend import groupwise

    monkeypatch.setattr(groupwise, "execution_policy", lambda: SimpleNamespace(pdl=pdl))
    torch.manual_seed(rows)
    data = torch.randn(rows, 5120, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(columns, 5120, device="cuda").to(data.dtype)
    scale = torch.rand(rows, 40, device="cuda") * 0.01
    weight_scale = torch.rand(columns // 128, 40, device="cuda") * 0.01
    storage = torch.full((rows + 2, columns), 19, device="cuda", dtype=torch.bfloat16)
    output = storage[1:-1]
    workspace = groupwise._workspace(data.device)
    module = groupwise.compiled()

    def projection():
        module.groupwise_fp8_transposed(
            data, weight, scale, weight_scale, output, workspace, pdl
        )
        return fp8._project_quantized(data, scale, weight, weight_scale)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        projection()
        projection()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = projection()
    sampled_columns = torch.cat(
        (
            torch.tensor([0, 127, 128, columns - 1], device="cuda"),
            torch.linspace(256, columns - 2, 60, device="cuda").long(),
        )
    )
    for multiplier in (0.5, 1.25):
        torch.cuda.current_stream().wait_stream(stream)
        data.copy_(torch.randn_like(data.float()).to(data.dtype))
        weight.copy_(torch.randn_like(weight.float()).to(weight.dtype))
        scale.mul_(multiplier)
        weight_scale.mul_(multiplier)
        output.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        activation = data.double().reshape(rows, 40, 128)
        weights = weight[sampled_columns].double().reshape(64, 40, 128)
        blocks = (activation[:, None] * weights[None]).sum(dim=-1)
        expected = (
            blocks
            * scale.double()[:, None]
            * weight_scale[sampled_columns // 128].double()[None]
        ).sum(dim=-1)
        torch.testing.assert_close(
            output[:, sampled_columns].double(), expected, rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(actual, output, rtol=0, atol=0)
        torch.testing.assert_close(
            storage[[0, -1]], torch.full_like(storage[[0, -1]], 19), rtol=0, atol=0
        )
    del graph
    torch.cuda.synchronize()


@pytest.mark.parametrize("fault", ["alias", "alignment", "shape", "scale_stride"])
def test_transposed_projection_rejects_invalid_storage_before_launch(fault):
    from oh_my_vllm.kernels.cuda_backend import groupwise

    data = torch.zeros(4, 5120, device="cuda", dtype=torch.float8_e4m3fn)
    weight = torch.zeros(14336, 5120, device="cuda", dtype=data.dtype)
    scale = torch.ones(4, 40, device="cuda")
    weight_scale = torch.ones(112, 40, device="cuda")
    output = torch.empty(4, 14336, device="cuda", dtype=torch.bfloat16)
    if fault == "alias":
        output = weight.view(torch.bfloat16).flatten()[: output.numel()].view_as(output)
    elif fault == "alignment":
        data = torch.empty(data.numel() + 1, device="cuda", dtype=data.dtype)[
            1:
        ].view_as(data)
    elif fault == "shape":
        output = output[:3]
    else:
        scale = torch.empty(40, 4, device="cuda").T
    with pytest.raises(Exception, match="Check failed"):
        groupwise.compiled().groupwise_fp8_transposed(
            data,
            weight,
            scale,
            weight_scale,
            output,
            groupwise._workspace(data.device),
            True,
        )
    torch.cuda.synchronize()
