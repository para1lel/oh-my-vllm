"""Column clusters must preserve boundary shapes and live graph inputs."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def replay_stream():
    from oh_my_vllm.kernels.cuda_backend import groupwise

    # This test module owns a short-lived stream, separate from worker streams.
    with pytest.MonkeyPatch.context() as context:
        context.setattr(groupwise, "_WORKSPACES", {})
        yield torch.cuda.Stream()
        torch.cuda.synchronize()


@pytest.mark.parametrize("rows", [623, 624, 2496, 2497])
@pytest.mark.parametrize("pdl", [False, True])
def test_cluster_boundaries_replay_changed_data_and_scales(
    rows, pdl, replay_stream, monkeypatch
):
    from flashinfer.gemm import gemm_fp8_nt_groupwise
    from oh_my_vllm.kernels.cuda_backend import groupwise

    monkeypatch.setattr(groupwise, "execution_policy", lambda: SimpleNamespace(pdl=pdl))
    generator = torch.Generator(device="cuda").manual_seed(932 + rows)
    data = torch.randn(rows, 5120, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    weight = torch.randn(34816, 5120, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.rand(rows, 40, device="cuda", generator=generator) * 0.01
    scale_weight = torch.rand(272, 40, device="cuda", generator=generator) * 0.01
    weight_before = weight.view(torch.uint8).clone()

    def projection():
        return groupwise.gemm(data, weight, scale_a, scale_weight, mma_sm=1)

    replay_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(replay_stream):
        projection()
        projection()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = projection()
    torch.cuda.current_stream().wait_stream(replay_stream)
    for step in range(2):
        data.copy_(
            torch.randn(rows, 5120, device="cuda", generator=generator).to(
                torch.float8_e4m3fn
            )
        )
        scale_a.mul_(0.75 + step)
        scale_weight.mul_(1.25)
        data_before = data.view(torch.uint8).clone()
        activation_scale_before = scale_a.clone()
        scale_before = scale_weight.clone()
        # This library route has its own unmodified one-CTA CUTLASS launcher.
        expected = gemm_fp8_nt_groupwise(
            data,
            weight,
            scale_a,
            scale_weight,
            scale_major_mode="K",
            mma_sm=1,
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert result.shape == (rows, 34816)
        assert result.is_contiguous()
        assert torch.equal(data.view(torch.uint8), data_before)
        assert torch.equal(weight.view(torch.uint8), weight_before)
        assert torch.equal(scale_a, activation_scale_before)
        assert torch.equal(scale_weight, scale_before)
