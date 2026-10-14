"""Large split reductions keep weighted means, empty heads and graph inputs."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("rows,heads", [(1, 24), (4, 24), (20, 24), (32, 24), (5, 16)])
def test_large_split_merge_weighted_mean_guards_and_changed_stream_replay(rows, heads):
    from oh_my_vllm.kernels.cuda_backend import compiled

    generator = torch.Generator(device="cuda").manual_seed(784 + rows + heads)
    partial = torch.empty(rows, heads, 128, 256, device="cuda", dtype=torch.float32)
    lse = torch.empty(rows, heads, 128, device="cuda", dtype=torch.float32)
    storage = torch.full(
        (rows * heads * 256 + 4,), 17, device="cuda", dtype=torch.bfloat16
    )
    output = storage[2:-2].view(rows, heads, 256)
    module = compiled()

    def set_inputs(pattern):
        partial.copy_(torch.randn(partial.shape, generator=generator, device="cuda"))
        lse.copy_(torch.randn(lse.shape, generator=generator, device="cuda") * 4)
        if pattern == "cancellation":
            signs = torch.where(torch.arange(128, device="cuda") % 2 == 0, 1.0, -1.0)
            partial.mul_(0.25).add_(signs[None, None, :, None] * 4096)
            lse.zero_()
        elif pattern == "dominant":
            lse.fill_(-80)
            lse[..., 63] = 80
        elif pattern == "mixed_empty":
            lse[..., 73:] = -torch.inf
            lse[0].fill_(-torch.inf)
        elif pattern == "all_empty":
            lse.fill_(-torch.inf)

    def expected():
        logs = lse.double()
        maximum = logs.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0)
        weights = torch.exp2(logs - maximum)
        total = weights.sum(-1, keepdim=True)
        numerator = (weights.unsqueeze(-1) * partial.double()).sum(-2)
        return torch.where(total > 0, numerator / total.clamp_min(1e-200), 0)

    def check(pattern, snapshots):
        torch.testing.assert_close(output.double(), expected(), atol=0.03, rtol=0.03)
        for guard in (storage[:2], storage[-2:]):
            torch.testing.assert_close(
                guard, torch.full_like(guard, 17), atol=0, rtol=0
            )
        for tensor, before in zip((partial, lse), snapshots, strict=True):
            torch.testing.assert_close(tensor.view(torch.uint8), before, atol=0, rtol=0)
        if pattern == "all_empty":
            assert torch.count_nonzero(output).item() == 0
        elif pattern == "mixed_empty":
            assert torch.count_nonzero(output[0]).item() == 0

    patterns = ("random", "cancellation", "dominant", "mixed_empty", "all_empty")
    for pattern in patterns:
        set_inputs(pattern)
        before = [tensor.view(torch.uint8).clone() for tensor in (partial, lse)]
        output.fill_(torch.nan)
        module.attention_merge(partial, lse, output)
        check(pattern, before)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.attention_merge(partial, lse, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            module.attention_merge(partial, lse, output)
    for pattern in ("random", "cancellation", "mixed_empty", "all_empty"):
        torch.cuda.current_stream().wait_stream(stream)
        set_inputs(pattern)
        before = [tensor.view(torch.uint8).clone() for tensor in (partial, lse)]
        output.fill_(torch.nan)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        check(pattern, before)
    torch.cuda.synchronize()
    graph.reset()
