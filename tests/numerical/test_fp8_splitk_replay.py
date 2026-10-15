"""K segmentation keeps live graph inputs and the final BF16 projection boundary."""

from types import SimpleNamespace

import pytest
import torch

from development.kernels import oracles
from tests.support.fp64 import close, cpu

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("rows", [1, 8, 9, 20, 32, 33])
@pytest.mark.parametrize("width", [6144, 17408])
@pytest.mark.parametrize("pdl", [False, True])
def test_projection_replays_changed_inputs_and_scales(rows, width, pdl, monkeypatch):
    from oh_my_vllm.kernels import fp8
    from oh_my_vllm.kernels.cuda_backend import groupwise

    monkeypatch.setattr(groupwise, "execution_policy", lambda: SimpleNamespace(pdl=pdl))
    data = torch.randn(rows, width, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(rows, width // 128, device="cuda") * 0.1 + 0.01
    weight = torch.randn(5120, width, device="cuda").to(data.dtype)
    weight_scale = torch.rand(40, width // 128, device="cuda") * 0.1 + 0.01
    tensors = (data, scale, weight, weight_scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fp8._project_quantized(data, scale, weight, weight_scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fp8._project_quantized(data, scale, weight, weight_scale)
    for multiplier in (0.5, 1.25):
        torch.cuda.current_stream().wait_stream(stream)
        data.copy_(torch.randn_like(data.float()).to(data.dtype))
        weight.copy_(torch.randn_like(weight.float()).to(weight.dtype))
        scale.mul_(multiplier)
        weight_scale.mul_(multiplier)
        snapshots = [tensor.view(torch.uint8).clone() for tensor in tensors]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        expected = oracles.groupwise(
            *cpu((data, weight, scale, weight_scale)), dtype=torch.float64
        )
        close(actual, expected, rtol=0.01, atol=0.01, check_dtype=False)
        for tensor, snapshot in zip(tensors, snapshots, strict=True):
            torch.testing.assert_close(tensor.view(torch.uint8), snapshot)
    del graph
    torch.cuda.synchronize()


@pytest.mark.parametrize("splits", [2, 8])
def test_native_partial_capacity_alias_and_output_canaries(splits):
    from oh_my_vllm.kernels.cuda_backend import groupwise

    rows, width = 9, 17408
    data = torch.randn(rows, width, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(5120, width, device="cuda").to(data.dtype)
    scale = torch.rand(rows, width // 128, device="cuda") * 0.1
    weight_scale = torch.rand(40, width // 128, device="cuda") * 0.1
    partial_storage = torch.full(
        (splits * rows * 5120 + 64,), 19, device="cuda", dtype=torch.float32
    )
    partials = partial_storage[32:-32].view(splits, rows, 5120)
    output_storage = torch.full(
        (rows * 5120 + 64,), 23, device="cuda", dtype=torch.bfloat16
    )
    output = output_storage[32:-32].view(rows, 5120)
    module = groupwise.compiled()
    workspace = groupwise._workspace(data.device)

    def project(partials=partials, output=output, splits=splits):
        module.groupwise_fp8_splitk(
            data, weight, scale, weight_scale, partials, output, workspace, splits, True
        )

    with pytest.raises(Exception, match="Check failed"):
        project(partials=partials[0])
    with pytest.raises(Exception, match="Check failed"):
        project(partials=partials[:-1])
    with pytest.raises(Exception, match="Check failed"):
        project(partials=torch.empty(splits, rows, 10240, device="cuda")[:, :, ::2])
    with pytest.raises(Exception, match="Check failed"):
        project(output=output.to(torch.float16))
    with pytest.raises(Exception, match="Check failed"):
        project(
            output=partials.flatten()
            .view(torch.bfloat16)[: rows * 5120]
            .view_as(output)
        )
    with pytest.raises(Exception, match="Check failed"):
        project(splits=3)
    partials.fill_(float("nan"))
    output.fill_(float("nan"))
    project()
    torch.cuda.synchronize()
    assert torch.isfinite(partials).all().item()
    expected = oracles.groupwise(
        *cpu((data, weight, scale, weight_scale)), dtype=torch.float64
    )
    close(output, expected, rtol=0.01, atol=0.01, check_dtype=False)
    for storage, expected in ((partial_storage, 19), (output_storage, 23)):
        torch.testing.assert_close(
            storage[:32], torch.full_like(storage[:32], expected), rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage[-32:], torch.full_like(storage[-32:], expected), rtol=0, atol=0
        )
