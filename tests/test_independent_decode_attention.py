"""Paged decode reference tests, including shifted MTP cache and graph replay."""

import os
import subprocess
import sys
from unittest.mock import patch

import oh_my_vllm.kernels.decode_attention as decode_attention
import pytest
import torch
from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.worker.decode_graph import DecodeAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("entry", ["append", "prepare_attention"])
@pytest.mark.parametrize("backend", ["cuda", "tilelang"])
def test_out_of_range_fa_write_slot_raises(entry, backend):
    environment = os.environ | {"OH_MY_VLLM_KERNEL_BACKEND": backend}
    process = subprocess.run(
        [sys.executable, "-m", "tests.gpu_slot_guard_case", entry],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_attention_merge_all_empty_splits_writes_zero():
    from oh_my_vllm.kernels.cuda_backend import compiled

    partial = torch.zeros(1, 24, 16, 256, device="cuda", dtype=torch.float32)
    lse = torch.full((1, 24, 16), -torch.inf, device="cuda")
    out = torch.full((1, 24, 256), float("nan"), device="cuda", dtype=torch.bfloat16)
    partial[0, 0, 0] = 1
    lse[0, 0, 0] = 0
    compiled().attention_merge(partial, lse, out)
    torch.testing.assert_close(out[0, 0], torch.ones_like(out[0, 0]))
    torch.testing.assert_close(out[0, 1:], torch.zeros_like(out[0, 1:]))


@pytest.mark.parametrize("backend", ["cuda", "tilelang"])
def test_grouped_decode_rejects_group_size_above_5(backend):
    environment = os.environ | {"OH_MY_VLLM_KERNEL_BACKEND": backend}
    process = subprocess.run(
        [sys.executable, "-m", "tests.gpu_slot_guard_case", "grouped_decode"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_eight_target_rows_match_independent_causal_reference(batch):
    generator = torch.Generator(device="cuda").manual_seed(784)
    query = torch.randn(
        batch * 8, 24, 256, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    cache = torch.randn(
        1 + batch * 2,
        2,
        784,
        4,
        256,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    tables = torch.tensor(
        [[1 + row * 2, 2 + row * 2] for row in range(batch)],
        device="cuda",
        dtype=torch.int32,
    ).repeat_interleave(8, 0)
    lengths = torch.tensor(
        list(range(779, 787)) * batch, device="cuda", dtype=torch.int32
    )
    starts = torch.arange(0, batch * 8 + 1, 8, device="cuda", dtype=torch.int32)
    actual = decode(
        query, cache, tables, lengths, max_tokens=1568, starts=starts, max_query_len=8
    )
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, 0),
        atol=0.03,
        rtol=0.03,
    )


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


@pytest.mark.parametrize("length", [1, 15, 16, 17, 783, 784, 785, 4095, 4096])
def test_flashinfer_native_decode_ignores_workspace_and_stale_subpage_bytes(length):
    """Test first-use workspace and unwritten KV tails on TRT-LLM decode."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    generator = torch.Generator(device="cpu").manual_seed(1000 + length)
    query = torch.ones(1, 24, 256, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(length, 4, 256, generator=generator).bfloat16()
    values = torch.randn(length, 4, 256, generator=generator).bfloat16()
    pages = (length + 783) // 784
    mapping = list(range(1, pages + 1))[::-1]
    caches = []
    for stale in (0, 128):
        cache = torch.full(
            (pages + 1, 2, 784, 4, 256),
            stale,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for logical, physical in enumerate(mapping):
            start, end = logical * 784, min((logical + 1) * 784, length)
            cache[physical, 0, : end - start] = keys[start:end].cuda()
            cache[physical, 1, : end - start] = values[start:end].cuda()
        caches.append(cache)
    tables = torch.tensor([mapping], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    outputs = []
    with patch(
        "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
        wraps=trtllm_batch_decode_with_kv_cache,
    ) as backend:
        for cache, workspace_value in (
            (caches[0], 0),
            (caches[0], 0xA5),
            (caches[1], 0),
            (caches[1], 0xA5),
        ):
            attention = DecodeAttention(tables, lengths, extent=max(4096, length))
            attention.prepare()
            assert attention.native_metadata is not None
            attention.workspace.fill_(workspace_value)
            first = attention(query, cache).detach().clone()
            repeated = attention(query, cache).detach().clone()
            torch.testing.assert_close(repeated, first, atol=0, rtol=0)
            outputs.append(first)
        assert backend.call_count == 8
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=0, rtol=0)
    torch.testing.assert_close(
        outputs[0].cpu().double(),
        reference(query, caches[0], tables, lengths, first=0),
        atol=0.03,
        rtol=0.03,
    )


@pytest.mark.parametrize(
    "batch,group,length", [(4, 1, 32769), (1, 5, 785), (1, 1, 131073)]
)
def test_flashinfer_native_decode_first_use_poison_at_production_shapes(
    batch, group, length
):
    """Test poisoned scratch/tails in batched, grouped and long decode."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    pages = (length + 783) // 784
    cache_shape = (1 + batch * pages, 2, 784, 4, 256)
    valid = torch.randn(cache_shape, device="cuda", dtype=torch.bfloat16)
    tables = []
    for row in range(batch):
        mapping = list(range(1 + row * pages, 1 + (row + 1) * pages))[::-1]
        tables.extend([mapping] * group)
    compact = torch.tensor(tables, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [
            position
            for _ in range(batch)
            for position in range(length - group + 1, length + 1)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    caches = []
    for stale in (0, 128):
        cache = valid.clone()
        for row in range(batch):
            physical = tables[row * group][-1]
            cache[physical, :, length % 784 :] = stale
        caches.append(cache)
    query = torch.ones(batch * group, 24, 256, device="cuda", dtype=torch.bfloat16)
    starts = (
        torch.arange(0, batch * group + 1, group, device="cuda", dtype=torch.int32)
        if group > 1
        else None
    )
    outputs = []
    with patch(
        "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
        wraps=trtllm_batch_decode_with_kv_cache,
    ) as backend:
        for cache, workspace_value in (
            (caches[0], 0),
            (caches[0], 0xA5),
            (caches[1], 0),
            (caches[1], 0xA5),
        ):
            attention = DecodeAttention(compact, lengths, extent=max(4096, length))
            attention.starts = starts
            attention.prepare()
            assert attention.native_metadata is not None
            attention.workspace.fill_(workspace_value)
            first = attention(query, cache).detach().clone()
            repeated = attention(query, cache).detach().clone()
            torch.testing.assert_close(repeated, first, atol=0, rtol=0)
            outputs.append(first)
        assert backend.call_count == 8
    for output in outputs:
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, outputs[0], atol=0, rtol=0)


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


@pytest.mark.parametrize("compiled", [False, True])
def test_native_query_bucket_replay_changes_request_boundaries(compiled):
    """One bound-five graph serves [1,4] and [2,3] with dynamic causality."""
    from oh_my_vllm.ir import compile_forward

    generator = torch.Generator(device="cuda").manual_seed(1568)
    query = torch.randn(
        5, 24, 256, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    cache = torch.randn(
        5, 2, 784, 4, 256, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    before = cache.clone()
    tables = torch.tensor([[1, 2]] + [[3, 4]] * 4, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([21, 782, 783, 784, 785], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 1, 5], device="cuda", dtype=torch.int32)
    # Production allocates scratch before compiling the opaque attention plan.
    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    attention = DecodeAttention(
        tables, lengths, 1568, workspace=workspace, max_query_len=5
    )
    attention.starts = starts

    def run():
        attention.prepare()
        return attention(query, cache)

    unit = compile_forward(run, unit="query_bucket_replay") if compiled else run
    initial = unit().clone()
    torch.testing.assert_close(
        initial.cpu().double(),
        reference(query, cache, tables, lengths, 0),
        atol=0.03,
        rtol=0.03,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit()

    # Change the group boundaries, both page mappings, and positions in place.
    starts.copy_(torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32))
    tables.copy_(
        torch.tensor([[4, 1]] * 2 + [[2, 3]] * 3, device="cuda", dtype=torch.int32)
    )
    lengths.copy_(
        torch.tensor([783, 784, 784, 785, 786], device="cuda", dtype=torch.int32)
    )
    query.copy_(
        torch.randn(query.shape, generator=generator, device="cuda", dtype=query.dtype)
    )
    graph.replay()
    replayed = output.clone()
    expected = reference(query, cache, tables, lengths, 0)
    torch.testing.assert_close(replayed.cpu().double(), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(replayed, run(), atol=0.03, rtol=0.03)
    assert not torch.allclose(replayed, initial, atol=0.03, rtol=0.03)
    torch.testing.assert_close(cache, before, atol=0, rtol=0)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("position64", [False, True])
def test_cuda_partial_tile_crosses_page_with_masked_tail(grouped, position64):
    from oh_my_vllm.kernels.cuda_backend import compiled

    torch.manual_seed(94)
    query = torch.ones(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(3, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    cache[2, 0, 0].fill_(1)
    cache[2, 1, 0].fill_(16)
    cache[0, 0, 0].fill_(1)
    cache[0, 1, 0].fill_(-16)
    tables = torch.tensor([[1, 2]] * 3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([783, 784, 785], device="cuda", dtype=torch.int32)
    starts = (
        torch.tensor([0, 3], device="cuda", dtype=torch.int32) if grouped else lengths
    )
    partial = torch.empty(3, 24, 64, 256, device="cuda", dtype=torch.float32)
    lse = torch.empty(3, 24, 64, device="cuda", dtype=torch.float32)
    out = torch.empty_like(query)
    cuda = compiled()

    def launch():
        cuda.attention_partial(
            query,
            cache,
            tables,
            lengths,
            starts,
            partial,
            lse,
            0,
            grouped,
            position64,
            5,
        )
        cuda.attention_merge(partial, lse, out)

    launch()
    expected = reference(query, cache, tables, lengths, 0)
    torch.testing.assert_close(out.cpu().double(), expected, atol=0.03, rtol=0.03)
    assert expected[2].mean() > 8

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch()
    tables[:, 1] = 0
    graph.replay()
    expected = reference(query, cache, tables, lengths, 0)
    torch.testing.assert_close(out.cpu().double(), expected, atol=0.03, rtol=0.03)
    assert expected[2].mean() < -8


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


@pytest.mark.parametrize("unaligned", [(True, False), (False, True), (True, True)])
def test_contiguous_inputs_with_unaligned_storage_offset(
    unaligned, caplog, monkeypatch
):
    monkeypatch.setattr(decode_attention, "_unaligned_cache_clone_count", 0)
    query, cache, tables, lengths = inputs()
    tensors = [query, cache]
    for index, enabled in enumerate(unaligned):
        if enabled:
            original = tensors[index]
            storage = torch.empty(
                original.numel() + 1, device=original.device, dtype=original.dtype
            )
            tensors[index] = storage[1:].view_as(original).copy_(original)
            assert (
                tensors[index].is_contiguous() and tensors[index].data_ptr() % 16 != 0
            )
    actual = decode(*tensors, tables, lengths, first=1, max_tokens=1568)
    copied_cache = unaligned[1]
    assert decode_attention.unaligned_cache_clone_count() == int(copied_cache)
    assert ("unaligned FA cache clone" in caplog.text) == copied_cache
    torch.testing.assert_close(
        actual.cpu().double(),
        reference(query, cache, tables, lengths, 1),
        rtol=0.03,
        atol=0.03,
    )


def test_unaligned_cache_clone_warning_and_count_accumulate(caplog, monkeypatch):
    monkeypatch.setattr(decode_attention, "_unaligned_cache_clone_count", 0)
    query, cache, tables, lengths = inputs()
    storage = torch.empty(cache.numel() + 1, device="cuda", dtype=cache.dtype)
    unaligned_cache = storage[1:].view_as(cache).copy_(cache)
    assert unaligned_cache.data_ptr() % 16 != 0
    for expected_count, warnings in ((1, 1), (2, 2), (3, 2), (4, 3)):
        actual = decode(
            query, unaligned_cache, tables, lengths, first=1, max_tokens=1568
        )
        torch.testing.assert_close(
            actual.cpu().double(),
            reference(query, cache, tables, lengths, 1),
            rtol=0.03,
            atol=0.03,
        )
        assert decode_attention.unaligned_cache_clone_count() == expected_count
        assert caplog.text.count("unaligned FA cache clone") == warnings
