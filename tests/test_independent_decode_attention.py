"""Paged decode reference tests, including shifted MTP cache and graph replay."""

import pytest
import torch
from oh_my_vllm.kernels.decode_attention import decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def reference(query, cache, tables, lengths, first):
    q, pool = query.cpu().double(), cache.cpu().double()
    out = []
    for i, length in enumerate(lengths.cpu().tolist()):
        slots = torch.arange(first, length)
        pages = tables[i].cpu()[slots // 784].long()
        keys = pool[pages, 0, slots % 784].repeat_interleave(6, 1)
        values = pool[pages, 1, slots % 784].repeat_interleave(6, 1)
        scores = torch.einsum("hd,thd->ht", q[i], keys) / 16
        out.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values))
    return torch.stack(out)


def inputs():
    torch.manual_seed(33)
    query = torch.randn(3, 24, 256, dtype=torch.bfloat16, device="cuda")
    cache = torch.randn(7, 2, 784, 4, 256, dtype=torch.bfloat16, device="cuda")
    tables = torch.tensor([[2, 5], [1, 4], [6, 3]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([785, 37, 2], dtype=torch.int32, device="cuda")
    return query, cache, tables, lengths


@pytest.mark.parametrize("first", [0, 1])
def test_paged_decode(first):
    query, cache, tables, lengths = inputs()
    before = cache.clone()
    actual = decode(query, cache, tables, lengths, first=first, max_tokens=1568)
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, first),
        rtol=0.03,
        atol=0.03,
    )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    if first == 1:
        # Cache position zero is deliberately absent in the MTP representation.
        cache[tables[:, 0].long(), :, 0] = 100
        changed = decode(query, cache, tables, lengths, first=1, max_tokens=1568)
        torch.testing.assert_close(changed, actual, rtol=0, atol=0)


def test_graph_replay_reads_updated_lengths_and_tables():
    query, cache, tables, lengths = inputs()
    decode(query, cache, tables, lengths, first=1, max_tokens=1568)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = decode(query, cache, tables, lengths, first=1, max_tokens=1568)
    lengths.copy_(torch.tensor([800, 783, 45], dtype=torch.int32, device="cuda"))
    tables[0, 1] = 3
    graph.replay()
    torch.testing.assert_close(
        out.cpu().double(),
        reference(query, cache, tables, lengths, 1),
        rtol=0.03,
        atol=0.03,
    )
