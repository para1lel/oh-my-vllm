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


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "columns,inner",
    [(34816, 5120), (16384, 5120), (14336, 5120), (5120, 17408), (5120, 6144)],
)
def test_identity_epilogue_replay_preserves_inputs_and_output_guards(
    columns, inner, pdl, replay_stream
):
    from flashinfer.gemm import gemm_fp8_nt_groupwise
    from oh_my_vllm.kernels.cuda_backend import groupwise

    rows = 624
    generator = torch.Generator(device="cuda").manual_seed(948 + columns + inner)
    data = torch.randn(rows, inner, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    weight = torch.randn(columns, inner, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.rand(rows, inner // 128, device="cuda", generator=generator) * 0.01
    scale_weight = (
        torch.rand(columns // 128, inner // 128, device="cuda", generator=generator)
        * 0.01
    )
    # Eight BF16 values keep the output pointer 16-byte aligned. The surrounding
    # words detect writes past the logical matrix, including the last CTA tail.
    storage = torch.full((rows * columns + 16,), 123, device="cuda", dtype=torch.int16)
    result = storage[8:-8].view(torch.bfloat16).view(rows, columns)
    module = groupwise.compiled()

    def projection():
        module.groupwise_fp8(
            data,
            weight,
            scale_a,
            scale_weight,
            result,
            groupwise._workspace(data.device),
            1,
            pdl,
            True,
        )

    replay_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(replay_stream):
        projection()
        projection()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection()
    torch.cuda.current_stream().wait_stream(replay_stream)
    for step in range(2):
        data.copy_(
            torch.randn(rows, inner, device="cuda", generator=generator).to(
                torch.float8_e4m3fn
            )
        )
        weight.copy_(
            torch.randn(columns, inner, device="cuda", generator=generator).to(
                torch.float8_e4m3fn
            )
        )
        scale_a.mul_(0.75 + step)
        scale_weight.mul_(1.25)
        protected = [
            tensor.view(torch.uint8).clone()
            for tensor in (data, weight, scale_a, scale_weight)
        ]
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
        # The conversion callback must ignore old D, even when it contains NaN.
        result.fill_(float("nan"))
        replay_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(replay_stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(replay_stream)
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(storage[:8], torch.full_like(storage[:8], 123))
        assert torch.equal(storage[-8:], torch.full_like(storage[-8:], 123))
        for tensor, before in zip(
            (data, weight, scale_a, scale_weight), protected, strict=True
        ):
            assert torch.equal(tensor.view(torch.uint8), before)
