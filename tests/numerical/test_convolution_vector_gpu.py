"""Vector convolution preserves taps, snapshots, metadata widths, and alignment."""

import itertools

import pytest
import torch
from oh_my_vllm.kernels.cuda_backend import compiled

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows", [127, 128, 129, 624, 4095, 4096, 4097, 8192])
@pytest.mark.parametrize(
    "types",
    [(torch.int64, torch.int32, torch.int64), (torch.int32, torch.int64, torch.int32)],
)
@torch.inference_mode()
def test_vector_convolution_matches_scalar_with_mixed_sequences(rows, types):
    torch.manual_seed(71)
    counts = [1, 3, rows - 4]
    width, stride = 10240, 16384
    backing = torch.randn(rows * stride + 4, device="cuda", dtype=torch.bfloat16)
    # The same values with a different address select the scalar model fallback.
    x = backing[: rows * stride].view(rows, stride)[:, :width]
    scalar_backing = torch.empty_like(backing)
    scalar = torch.as_strided(scalar_backing[1:], (rows, width), (stride, 1))
    scalar.copy_(x)
    weight = torch.randn(width, 4, device="cuda", dtype=torch.bfloat16)
    ids = torch.cat(
        [
            torch.full((n,), i, device="cuda", dtype=types[0])
            for i, n in enumerate(counts)
        ]
    )
    starts = torch.tensor(
        [0, *itertools.accumulate(counts)], device="cuda", dtype=types[1]
    )
    sources = torch.randn(3, width, 3, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(6, width, 3, device="cuda", dtype=torch.bfloat16)
    reference_pool = pool.clone()
    writes = torch.full((rows,), -1, device="cuda", dtype=types[2])
    destinations = [0, 1, 2, 3, rows - 2, rows - 1]
    writes[destinations] = torch.arange(6, device="cuda", dtype=types[2])
    out, reference = torch.empty_like(x), torch.empty_like(x)
    kernel = compiled().convolution
    kernel(scalar, weight, reference_pool, ids, starts, sources, writes, reference)
    kernel(x, weight, pool, ids, starts, sources, writes, out)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
    # Captured writes keep the same stream dependencies and addresses.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, weight, pool, ids, starts, sources, writes, out)
    graph.replay()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
    # Replay must observe new data and destination indices at fixed addresses.
    x.add_(0.25)
    scalar.copy_(x)
    sources.mul_(0.75)
    writes.fill_(-1)
    writes[[0, 2, rows - 1]] = torch.tensor([5, 4, 3], device="cuda", dtype=types[2])
    pool.fill_(0.5)
    reference_pool.copy_(pool)
    kernel(scalar, weight, reference_pool, ids, starts, sources, writes, reference)
    graph.replay()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
