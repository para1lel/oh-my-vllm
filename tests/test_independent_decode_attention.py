"""Paged decode reference tests, including shifted MTP cache and graph replay."""

import pytest
import torch
from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.worker.decode_graph import DecodeAttention

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


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("native", [False, True])
def test_grouped_verification_preserves_ragged_causality(first, native):
    torch.manual_seed(91)
    starts = torch.tensor([0, 5, 6, 9], device="cuda", dtype=torch.int32)
    query = torch.randn(9, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(7, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor(
        [[2, 5]] * 5 + [[1, 4]] + [[6, 3]] * 3, device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(
        [782, 783, 784, 785, 786, 2, 34, 35, 36], device="cuda", dtype=torch.int32
    )

    attention = DecodeAttention(tables, lengths, 1568)
    attention.first, attention.starts = first, starts

    def run():
        if native:
            attention.prepare()
            return attention(query, cache)
        return decode(
            query, cache, tables, lengths, first=first, max_tokens=1568, starts=starts
        )

    actual = run()
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, first),
        rtol=0.03,
        atol=0.03,
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        actual = run()
    # Same total/request shape, different row grouping, tables and lengths.
    starts.copy_(torch.tensor([0, 2, 6, 9], device="cuda", dtype=torch.int32))
    tables[:2] = tables[6]
    tables[2:6] = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
    lengths.copy_(torch.tensor([800, 801, 70, 71, 72, 73, 90, 91, 92], device="cuda"))
    graph.replay()
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, first),
        rtol=0.03,
        atol=0.03,
    )


@pytest.mark.parametrize("first", [0, 1])
def test_single_group_verification(first):
    torch.manual_seed(92)
    query = torch.randn(5, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(3, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[2, 1]] * 5, device="cuda", dtype=torch.int32)
    lengths = torch.arange(782, 787, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    actual = decode(
        query, cache, tables, lengths, first=first, max_tokens=1568, starts=starts
    )
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, first),
        rtol=0.03,
        atol=0.03,
    )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_cache_addresses_beyond_signed_int32(grouped, native):
    # The last page starts above 2**31 BF16 elements. Casting after multiplication
    # would be too late: both decode kernels must widen the page ID first.
    torch.manual_seed(144)
    cache = torch.empty(1400, 2, 784, 4, 256, dtype=torch.bfloat16, device="cuda")
    small = torch.randn(1, 2, 784, 4, 256, dtype=torch.bfloat16, device="cuda")
    cache[1399].copy_(small[0])
    from oh_my_vllm.kernels.attention import append

    # Also cover int32 slot inputs: multiplication must widen before addressing.
    keys = torch.randn(5, 4, 256, dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(keys)
    slots = 1399 * 784 + torch.arange(5, dtype=torch.int32, device="cuda")
    append(cache, keys, values, slots)
    small[0, 0, :5].copy_(keys)
    small[0, 1, :5].copy_(values)
    query = torch.randn(5, 24, 256, dtype=torch.bfloat16, device="cuda")
    tables = torch.full((5, 1), 1399, dtype=torch.int32, device="cuda")
    lengths = torch.arange(1, 6, dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 5], dtype=torch.int32, device="cuda") if grouped else None
    if native:
        attention = DecodeAttention(tables, lengths, 784)
        attention.starts = starts
        attention.prepare()
        actual = attention(query, cache)
    else:
        actual = decode(query, cache, tables, lengths, max_tokens=784, starts=starts)
    expected = reference(query, small, torch.zeros_like(tables), lengths, 0)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=0.03, rtol=0.03)
