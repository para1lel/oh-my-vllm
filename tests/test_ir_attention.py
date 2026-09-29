"""Planned attention stays opaque to fullgraph compilation."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward, operations
from oh_my_vllm.ir.attention import (
    MTP_DECODE,
    MTP_PREFILL,
    PAGED_PREFILL,
    TARGET_NATIVE_DECODE,
    TARGET_OWNED_DECODE,
    planned_attention,
    register_plan,
)
from oh_my_vllm.worker.decode_graph import DecodeAttention


def test_attention_reference_and_live_plan_handle():
    query = torch.randn(1, 24, 256, dtype=torch.bfloat16)
    cache = torch.randn(2, 2, 784, 4, 256, dtype=torch.bfloat16)
    tables = torch.tensor([[0]], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)
    plan = DecodeAttention(tables, lengths, 784)
    actual = operations()["planned_attention"].reference(
        query,
        cache,
        plan._ir_handle,
        TARGET_OWNED_DECODE,
        tables,
        lengths,
        None,
        None,
        None,
    )
    keys = cache[0, 0, :3].repeat_interleave(6, dim=1).double()
    values = cache[0, 1, :3].repeat_interleave(6, dim=1).double()
    scores = torch.einsum("hd,lhd->hl", query[0].double(), keys) / 16
    expected = torch.einsum("hl,lhd->hd", scores.softmax(-1), values).bfloat16()
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_attention_reference_crosses_page_and_excludes_mtp_row_zero():
    class Plan:
        pass

    query = torch.randn(2, 24, 256, dtype=torch.bfloat16)
    cache = torch.randn(2, 2, 784, 4, 256, dtype=torch.bfloat16)
    paged = Plan()
    paged._ir_reference_plan = (
        torch.tensor([0, 2]),
        torch.tensor([0, 2]),
        torch.tensor([0, 1]),
        torch.tensor([1]),
    )
    reference = operations()["planned_attention"].reference
    paged_handle = register_plan(paged)
    original = reference(
        query,
        cache,
        paged_handle,
        PAGED_PREFILL,
        None,
        None,
        None,
        None,
        None,
    )
    changed = cache.clone()
    changed[1, 0, 0].fill_(100)
    changed[1, 1, 0].fill_(100)
    actual = reference(
        query,
        changed,
        paged_handle,
        PAGED_PREFILL,
        None,
        None,
        None,
        None,
        None,
    )
    torch.testing.assert_close(original[0], actual[0], rtol=0, atol=0)
    assert not torch.equal(original[1], actual[1])

    mtp = Plan()
    mtp._ir_reference_plan = [0, 2], [[0, 1]], [783, 784]
    mtp_handle = register_plan(mtp)
    tables = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    lengths = torch.tensor([784, 785], dtype=torch.int32)
    for route in (MTP_PREFILL, MTP_DECODE):
        route_tables = tables if route == MTP_DECODE else None
        route_lengths = lengths if route == MTP_DECODE else None
        original = reference(
            query,
            cache,
            mtp_handle,
            route,
            route_tables,
            route_lengths,
            None,
            None,
            None,
        )
        changed = cache.clone()
        changed[0, :, 0].fill_(100)
        actual = reference(
            query,
            changed,
            mtp_handle,
            route,
            route_tables,
            route_lengths,
            None,
            None,
            None,
        )
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    changed_tables = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
    changed_lengths = torch.tensor([5, 5], dtype=torch.int32)
    changed_decode = reference(
        query,
        cache,
        mtp_handle,
        MTP_DECODE,
        changed_tables,
        changed_lengths,
        None,
        None,
        None,
    )
    keys = cache[1, 0, 1:5].repeat_interleave(6, dim=1).double()
    values = cache[1, 1, 1:5].repeat_interleave(6, dim=1).double()
    scores = torch.einsum("hd,lhd->hl", query[0].double(), keys) / 16
    expected = torch.einsum("hl,lhd->hd", scores.softmax(-1), values).bfloat16()
    torch.testing.assert_close(changed_decode[0], expected, rtol=0, atol=0)


def test_native_decode_reference_uses_explicit_metadata():
    query = torch.randn(1, 24, 256, dtype=torch.bfloat16)
    cache = torch.randn(2, 2, 784, 4, 256, dtype=torch.bfloat16)
    tables = torch.tensor([[0]], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)
    plan = DecodeAttention(tables, lengths, 784)
    subpages = torch.arange(49, dtype=torch.int32)[None]
    native_starts = torch.tensor([0, 1], dtype=torch.int32)
    reference = operations()["planned_attention"].reference
    native = reference(
        query,
        cache,
        plan._ir_handle,
        TARGET_NATIVE_DECODE,
        tables,
        lengths,
        subpages,
        lengths,
        native_starts,
    )
    owned = reference(
        query,
        cache,
        plan._ir_handle,
        TARGET_OWNED_DECODE,
        tables,
        lengths,
        None,
        None,
        None,
    )
    torch.testing.assert_close(native, owned, rtol=0, atol=0)
    changed_logical_table = torch.tensor([[1]], dtype=torch.int32)
    changed_logical_length = torch.tensor([10], dtype=torch.int32)
    native_length = torch.tensor([2], dtype=torch.int32)
    native = reference(
        query,
        cache,
        plan._ir_handle,
        TARGET_NATIVE_DECODE,
        changed_logical_table,
        changed_logical_length,
        subpages,
        native_length,
        native_starts,
    )
    expected = reference(
        query,
        cache,
        plan._ir_handle,
        TARGET_OWNED_DECODE,
        tables,
        native_length,
        None,
        None,
        None,
    )
    torch.testing.assert_close(native, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_owned_decode_attention_compiles_with_dynamic_metadata():
    query = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([7], device="cuda", dtype=torch.int32)
    plan = DecodeAttention(tables, lengths, 784)
    expected = plan._run(query, cache, tables, lengths)
    eager = plan(query, cache)
    compiled_fn = compile_forward(
        lambda q, c, t, length_values: planned_attention(
            q, c, plan._ir_handle, TARGET_OWNED_DECODE, t, length_values
        )
    )
    compiled = compiled_fn(query, cache, tables, lengths)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)
    lengths.fill_(5)
    changed = compiled_fn(query, cache, tables, lengths)
    torch.testing.assert_close(
        changed, plan._run(query, cache, tables, lengths), rtol=0, atol=0
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_flashinfer_paged_prefill_compiles():
    from oh_my_vllm.kernels.attention import PagedAttention

    plan = PagedAttention()
    plan.plan(
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        24,
        4,
        256,
    )
    query = torch.randn(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    expected = plan._run(query, cache)
    eager = plan(query, cache)
    compiled = compile_forward(lambda q, c: plan(q, c))(query, cache)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_native_decode_reuses_compiled_graph_with_changed_metadata():
    query = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([7], device="cuda", dtype=torch.int32)
    plan = DecodeAttention(tables, lengths, 784)
    plan.prepare()

    def forward(q, c, t, length_values, subpages, native_lengths, starts):
        return planned_attention(
            q,
            c,
            plan._ir_handle,
            TARGET_NATIVE_DECODE,
            t,
            length_values,
            subpages,
            native_lengths,
            starts,
        )

    compiled = compile_forward(forward)
    first_metadata = plan.native_metadata
    first = compiled(query, cache, tables, lengths, *first_metadata)
    torch.testing.assert_close(
        first,
        plan._run(query, cache, tables, lengths, *first_metadata),
        rtol=0,
        atol=0,
    )

    changed_tables = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    offsets = torch.arange(49, device="cuda", dtype=torch.int32)
    changed_subpages = (changed_tables[:, :, None] * 98 + offsets).flatten(1)
    changed_lengths = torch.tensor([5], device="cuda", dtype=torch.int32)
    changed_metadata = changed_subpages, changed_lengths, first_metadata[2]
    second = compiled(query, cache, changed_tables, changed_lengths, *changed_metadata)
    torch.testing.assert_close(
        second,
        plan._run(query, cache, changed_tables, changed_lengths, *changed_metadata),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(first, second)
