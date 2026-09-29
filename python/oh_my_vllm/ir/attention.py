"""Opaque attention IR node over host-planned FlashInfer and owned decode paths.

The CPU plan handle names a live batch plan. Host planning remains outside the
compiled GPU unit; the handle value is data, so changing plans does not change
provider selection. A fixed route code identifies FlashInfer or the owned
kernel backend and is specialized with the graph's static metadata.
"""

from __future__ import annotations

import itertools
import math
import weakref
from typing import Any

import torch

from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec

# Route values are stable schema data, not tensor-content-based decisions.
PAGED_PREFILL = 0
MTP_PREFILL = 1
MTP_DECODE = 2
TARGET_NATIVE_DECODE = 3
TARGET_OWNED_DECODE = 4

_PLANS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_NEXT_PLAN = itertools.count(1)


def register_plan(plan: Any) -> torch.Tensor:
    """Create a non-reused CPU handle while a batch/graph owns the plan."""
    key = next(_NEXT_PLAN)
    _PLANS[key] = plan
    return torch.tensor(key, dtype=torch.int64)


def _lookup(handle: torch.Tensor) -> Any:
    if handle.device.type != "cpu" or handle.numel() != 1:
        raise ValueError("attention plan handle must be one CPU integer")
    key = int(handle.item())
    try:
        return _PLANS[key]
    except KeyError as exc:
        raise RuntimeError(f"attention plan {key} is no longer live") from exc


def _fake_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    torch._check(query.ndim == 3 and query.shape[-1] == 256)
    torch._check(cache.ndim == 5 and cache.shape[1:3] == (2, 784))
    torch._check(cache.shape[-1] == query.shape[-1])
    torch._check(plan_handle.shape == ())
    torch._check(0 <= route <= 4)
    torch._check((tables is None) == (lengths is None))
    torch._check(
        (tables is not None)
        == (route in (MTP_DECODE, TARGET_NATIVE_DECODE, TARGET_OWNED_DECODE))
    )
    if tables is not None:
        torch._check(tables.shape[0] == query.shape[0])
        torch._check(lengths.shape == (query.shape[0],))
    native = (native_tables, native_lengths, native_starts)
    torch._check(
        all(item is not None for item in native) == (route == TARGET_NATIVE_DECODE)
    )
    if route == TARGET_NATIVE_DECODE:
        torch._check(native_tables.ndim == 2)
        torch._check(native_lengths.ndim == 1 and native_starts.ndim == 1)
        torch._check(native_tables.shape[0] == native_lengths.shape[0])
        torch._check(native_starts.shape[0] == native_lengths.shape[0] + 1)
        torch._check(all(item.dtype == torch.int32 for item in native))
        torch._check(all(item.device == query.device for item in native))
    return torch.empty(query.shape, device=query.device, dtype=query.dtype)


def _attend_row(
    query: torch.Tensor,
    cache: torch.Tensor,
    page_table: Any,
    first: int,
    last: int,
) -> torch.Tensor:
    positions = torch.arange(first, last + 1, device=query.device, dtype=torch.long)
    pages = torch.as_tensor(page_table, device=query.device, dtype=torch.long)
    physical = pages[positions // 784]
    offsets = positions % 784
    keys = cache[physical, 0, offsets].repeat_interleave(
        query.shape[0] // cache.shape[3], dim=1
    )
    values = cache[physical, 1, offsets].repeat_interleave(
        query.shape[0] // cache.shape[3], dim=1
    )
    return _weighted_attention(query, keys, values)


def _weighted_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("hd,lhd->hl", query.double(), keys.double())
    scores = scores / math.sqrt(query.shape[-1])
    weights = scores.softmax(dim=-1)
    return torch.einsum("hl,lhd->hd", weights, values.double()).to(query.dtype)


def _attend_native_row(
    query: torch.Tensor,
    cache: torch.Tensor,
    subpages: torch.Tensor,
    last: int,
) -> torch.Tensor:
    positions = torch.arange(last + 1, device=query.device, dtype=torch.long)
    pages = subpages[positions // 16].long() // 98
    offsets = positions % 784
    keys = cache[pages, 0, offsets].repeat_interleave(
        query.shape[0] // cache.shape[3], dim=1
    )
    values = cache[pages, 1, offsets].repeat_interleave(
        query.shape[0] // cache.shape[3], dim=1
    )
    return _weighted_attention(query, keys, values)


def _reference_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    _fake_attention(
        query,
        cache,
        plan_handle,
        route,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )
    plan = _lookup(plan_handle)
    outputs = []
    if route == PAGED_PREFILL:
        starts, page_starts, pages, last_lengths = plan._ir_reference_plan
        starts = starts.tolist()
        page_starts = page_starts.tolist()
        pages = pages.tolist()
        last_lengths = last_lengths.tolist()
        for seq in range(len(starts) - 1):
            begin, end = starts[seq : seq + 2]
            table = pages[page_starts[seq] : page_starts[seq + 1]]
            length = (len(table) - 1) * 784 + last_lengths[seq]
            first_query = length - (end - begin)
            for row in range(begin, end):
                outputs.append(
                    _attend_row(query[row], cache, table, 0, first_query + row - begin)
                )
    elif route == MTP_PREFILL:
        starts, page_tables, positions = plan._ir_reference_plan
        for seq, table in enumerate(page_tables):
            for row in range(starts[seq], starts[seq + 1]):
                outputs.append(_attend_row(query[row], cache, table, 1, positions[row]))
    elif route == MTP_DECODE:
        for row in range(len(query)):
            outputs.append(
                _attend_row(
                    query[row], cache, tables[row], 1, int(lengths[row].item()) - 1
                )
            )
    elif route == TARGET_NATIVE_DECODE:
        if any(
            value is None for value in (native_tables, native_lengths, native_starts)
        ):
            raise ValueError("native decode requires explicit metadata")
        offsets = native_starts.cpu().tolist()
        native_lengths = native_lengths.cpu().tolist()
        for group, length in enumerate(native_lengths):
            begin, end = offsets[group : group + 2]
            for row in range(begin, end):
                outputs.append(
                    _attend_native_row(
                        query[row],
                        cache,
                        native_tables[group],
                        length - (end - row),
                    )
                )
    else:
        if tables is None or lengths is None:
            raise ValueError("decode attention requires explicit tables and lengths")
        for row in range(len(query)):
            outputs.append(
                _attend_row(
                    query[row],
                    cache,
                    tables[row],
                    plan.first,
                    int(lengths[row].item()) - 1,
                )
            )
    return torch.stack(outputs)


@torch.library.custom_op("oh_my_vllm_ir::planned_attention", mutates_args=())
def _ir_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    args = (
        query,
        cache,
        plan_handle,
        route,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )
    return _ATTENTION.select(*args).op(*args)


_ir_attention.register_fake(_fake_attention)


@torch.library.custom_op("oh_my_vllm_native::planned_attention", mutates_args=())
def _native_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    return _reference_attention(
        query,
        cache,
        plan_handle,
        route,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )


_native_attention.register_fake(_fake_attention)


@torch.library.custom_op("oh_my_vllm_flashinfer::planned_attention", mutates_args=())
def _flashinfer_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    return _lookup(plan_handle)._run(
        query,
        cache,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )


_flashinfer_attention.register_fake(_fake_attention)


@torch.library.custom_op("oh_my_vllm_kernel::planned_attention", mutates_args=())
def _owned_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None,
    lengths: torch.Tensor | None,
    native_tables: torch.Tensor | None,
    native_lengths: torch.Tensor | None,
    native_starts: torch.Tensor | None,
) -> torch.Tensor:
    return _lookup(plan_handle)._run(
        query,
        cache,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )


_owned_attention.register_fake(_fake_attention)


_ATTENTION = Operation(
    "planned_attention",
    _ir_attention,
    _native_attention,
    default_priority=("flashinfer", NAME),
)


def _supports(route: tuple[int, ...]):
    def check(
        query: TensorSpec,
        cache: TensorSpec,
        handle: TensorSpec,
        selected_route: int,
        tables: TensorSpec | None,
        lengths: TensorSpec | None,
        native_tables: TensorSpec | None,
        native_lengths: TensorSpec | None,
        native_starts: TensorSpec | None,
    ) -> bool:
        return (
            selected_route in route
            and query.device.type == "cuda"
            and cache.device == query.device
            and handle.device.type == "cpu"
            and (tables is None) == (lengths is None)
            and all(
                item is not None
                for item in (native_tables, native_lengths, native_starts)
            )
            == (selected_route == TARGET_NATIVE_DECODE)
        )

    return check


_ATTENTION.register_impl(
    "flashinfer",
    _flashinfer_attention,
    supports=_supports((PAGED_PREFILL, MTP_PREFILL, TARGET_NATIVE_DECODE)),
)
_ATTENTION.register_impl(
    NAME,
    _owned_attention,
    supports=_supports((MTP_DECODE, TARGET_OWNED_DECODE)),
)


def planned_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    plan_handle: torch.Tensor,
    route: int,
    tables: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    native_tables: torch.Tensor | None = None,
    native_lengths: torch.Tensor | None = None,
    native_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    return _ATTENTION(
        query,
        cache,
        plan_handle,
        route,
        tables,
        lengths,
        native_tables,
        native_lengths,
        native_starts,
    )
