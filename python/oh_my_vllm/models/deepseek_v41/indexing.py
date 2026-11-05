"""Bound temporary index scores without changing the per-head math."""

from collections.abc import Iterator
from numbers import Integral

import torch

INDEX_SCORE_WORK_BYTES = 256 * 1024 * 1024


def topk_indices(scores, count, *, request_widths=None, finite_scores=False):
    """Select ascending indices with Torch semantics by default, including NaNs.

    finite_scores=True asserts finite producer arithmetic before -inf masks.
    Only that explicit contract permits native CUDA BF16 count512 selection.
    No fixed cutoff-tie subset is promised for the native path. Prepare it
    with an Eager call before Graph recording; the graph pool retains captured
    padding and output allocations at fixed addresses until graph release.
    """
    if scores.ndim != 3 or type(count) is not int or not 0 <= count <= scores.size(-1):
        raise ValueError("index top-k requires [batch, query, width] and a valid count")
    if request_widths is None:
        width = scores.size(-1)
        if (
            finite_scores is True
            and scores.is_cuda
            and scores.dtype == torch.bfloat16
            and count == 512
            and 512 <= width < 2**23
            and scores.size(0) * scores.size(1) <= 32768
            and scores.is_contiguous()
            and scores.data_ptr() % 16 == 0
        ):
            from .kernel import _transform_module

            physical = scores
            if width % 512:
                physical = torch.empty(
                    (*scores.shape[:2], (width + 511) // 512 * 512),
                    dtype=scores.dtype,
                    device=scores.device,
                )
                physical[..., :width].copy_(scores)
                physical[..., width:].fill_(-torch.inf)
            output = torch.empty(
                (*scores.shape[:2], 512), dtype=torch.int64, device=scores.device
            )
            _transform_module().topk_bf16(
                physical.view(-1, physical.size(-1)), output.view(-1, 512), width
            )
            return output
        return scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    if len(request_widths) != scores.size(0) or any(
        type(width) is not int or not 0 <= width <= scores.size(-1)
        for width in request_widths
    ):
        raise ValueError(
            "index top-k widths must cover each request within the score table"
        )
    indices = torch.full(
        (*scores.shape[:2], count),
        scores.size(-1),
        dtype=torch.int64,
        device=scores.device,
    )
    for request, width in enumerate(request_widths):
        selected = min(count, width)
        if selected:
            indices[request, :, :selected] = (
                scores[request, :, :width]
                .topk(selected, dim=-1, sorted=False)
                .indices.sort(dim=-1)
                .values
            )
    return indices


def _rank_score_chunks(query, keys, weights, chunk_rows, work_bytes):
    import torch.distributed as dist

    from .kernel import _transform_module

    batch, rows, heads, dim = query.shape
    width = keys.size(1)
    chunk_rows = min(chunk_rows, rows)
    row_bytes = batch * (2 * heads * (dim + width + 1) + 8 * width)
    pipeline = (
        rows > chunk_rows
        and row_bytes * chunk_rows <= work_bytes
        and not torch.cuda.is_current_stream_capturing()
    )
    buffers = [
        torch.empty(
            batch * chunk_rows * width, dtype=torch.float32, device=query.device
        )
        for _ in range(2 if pipeline else 1)
    ]
    module = _transform_module()
    stream = torch.cuda.current_stream(query.device)
    pending = None
    try:
        for chunk, first in enumerate(range(0, rows, chunk_rows)):
            last = min(first + chunk_rows, rows)
            total = buffers[chunk % len(buffers)][
                : batch * (last - first) * width
            ].view(batch, last - first, width)
            with torch.no_grad():
                packed_query = query[:, first:last].contiguous()
                per_head = torch.einsum("bshd,btd->bsht", packed_query, keys)
                del packed_query
                module.weighted_heads(
                    per_head.view(-1, 8, width),
                    weights[:, first:last].contiguous().view(-1, 8),
                    total.view(-1, width),
                )
                del per_head
                if not pipeline:
                    dist.all_reduce(total)
                    scores = total.to(query.dtype, copy=True)
                else:
                    work = dist.all_reduce(total, async_op=True)
                    previous = pending
                    pending = first, last, total, work
                    if previous is None:
                        continue
                    begin, end, values, previous_work = previous
                    previous_work.wait()
                    scores = values.to(query.dtype, copy=True)
            if pipeline:
                yield begin, end, scores
            else:
                yield first, last, scores
            del scores
        if pending is not None:
            first, last, values, work = pending
            work.wait()
            pending = None
            yield first, last, values.to(query.dtype, copy=True)
    finally:
        if pending is not None:
            with torch.cuda.stream(stream):
                pending[3].wait()


def score_chunks(
    query: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    *,
    work_bytes: int = INDEX_SCORE_WORK_BYTES,
    rank_sum: bool = False,
    head_partitions: int = 1,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield query ranges and scores for [B,S,H,D], [B,T,D], [B,S,H].

    Inputs must have one device and one FP16, BF16, FP32, or FP64 dtype.
    H and D must be positive. Autocast must not mix FP16 and BF16.
    This inference helper does not record gradients. Each row reserves
    B * max(H*(D+T), 2*H*T, (H+1)*T) * element_size bytes for query packing,
    the per-head product, and the head reduction in separate lifetime phases.
    Input storage, caller-held outputs, and backend workspaces are not included.
    With rank_sum, FP16/BF16 rank sums use FP32 and round once to the score dtype.
    With head_partitions, each group rounds before local accumulation.
    Workspace terms use H/head_partitions and include the accumulator.
    A reusable accumulator adds B*T*accumulator_size bytes per row to every
    phase. CUDA BF16 eight-head rank sums fuse weighting and head reduction.
    Two accumulators can overlap scoring and rank reduction within this budget.
    Returned scores do not share the accumulator storage.
    Chunk ranges depend only on shapes, dtype size, and the positive integer
    budget. Empty S yields nothing; empty B or T yields one empty score chunk.
    """
    if isinstance(work_bytes, bool) or not isinstance(work_bytes, Integral):
        raise TypeError("work_bytes must be an integer, not a bool or tensor")
    if work_bytes <= 0:
        raise ValueError("work_bytes must be positive")
    if type(rank_sum) is not bool:
        raise TypeError("rank_sum must be a bool")
    tensors = (query, keys, weights)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("query, keys, and weights must be tensors")
    if query.ndim != 4 or keys.ndim != 3 or weights.ndim != 3:
        raise ValueError("expected query [B,S,H,D], keys [B,T,D], weights [B,S,H]")
    batch, rows, heads, dim = query.shape
    if heads == 0 or dim == 0:
        raise ValueError("query head count H and head dimension D must be positive")
    if keys.size(0) != batch or keys.size(2) != dim:
        raise ValueError("keys must match query batch B and head dimension D")
    if weights.shape != query.shape[:3]:
        raise ValueError("weights must match query [B,S,H]")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("query must have FP16, BF16, FP32, or FP64 dtype")
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise TypeError("query, keys, and weights must have the same dtype")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("query, keys, and weights must be on the same device")
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise ValueError("query, keys, and weights must have strided layouts")
    if (
        query.device.type in ("cpu", "cuda")
        and query.dtype in (torch.float16, torch.bfloat16)
        and torch.is_autocast_enabled(query.device.type)
        and torch.get_autocast_dtype(query.device.type) != query.dtype
    ):
        raise ValueError("autocast must not mix FP16 and BF16 index score dtypes")
    if rows == 0:
        return
    length = keys.size(1)
    if batch == 0 or length == 0:
        yield 0, rows, query.new_empty((batch, rows, length))
        return
    accumulator_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    partition_heads = heads // head_partitions
    local_sum = head_partitions > 1
    row_bytes = (
        batch
        * max(
            partition_heads * (dim + length),
            2 * partition_heads * length,
            (partition_heads + 1) * length,
        )
        * query.element_size()
    )
    if rank_sum or local_sum:
        accumulator_size = (
            4
            if query.dtype in (torch.float16, torch.bfloat16)
            else query.element_size()
        )
        row_bytes += batch * length * accumulator_size
    chunk_rows = int(work_bytes) // row_bytes
    if chunk_rows == 0:
        raise ValueError(
            f"work_bytes={work_bytes} is too small; need at least {row_bytes} bytes "
            "for one query row"
        )
    # Width one uses PyTorch's contiguous-head reduction order.
    if (
        rank_sum
        and not local_sum
        and query.is_cuda
        and query.dtype == torch.bfloat16
        and heads == 8
        and length > 1
        and weights.is_contiguous()
        and weights.data_ptr() % 16 == 0
    ):
        yield from _rank_score_chunks(query, keys, weights, chunk_rows, int(work_bytes))
        return
    if rank_sum:
        import torch.distributed as dist
    if rank_sum or local_sum:
        accumulator = torch.empty(
            batch * min(chunk_rows, rows) * length,
            dtype=accumulator_dtype,
            device=query.device,
        )
    if local_sum:
        grouped_query = query.unflatten(2, (head_partitions, partition_heads))
        grouped_weights = weights.unflatten(2, (head_partitions, partition_heads))
    for first in range(0, rows, chunk_rows):
        last = min(first + chunk_rows, rows)
        with torch.no_grad():
            if local_sum:
                total = accumulator[: batch * (last - first) * length].view(
                    batch, last - first, length
                )
                for partition in range(head_partitions):
                    packed_query = grouped_query[:, first:last, partition].contiguous()
                    per_head = torch.einsum("bshd,btd->bsht", packed_query, keys)
                    del packed_query
                    weighted = per_head.relu_() * grouped_weights[
                        :, first:last, partition
                    ].unsqueeze(-1)
                    del per_head
                    partial = weighted.sum(dim=2)
                    del weighted
                    if partition == 0:
                        total.copy_(partial)
                    else:
                        total.add_(partial)
                    del partial
                if rank_sum:
                    dist.all_reduce(total)
                scores = total.to(query.dtype, copy=True)
                del total
            else:
                packed_query = query[:, first:last].contiguous()
                per_head = torch.einsum("bshd,btd->bsht", packed_query, keys)
                del packed_query
                weighted = per_head.relu_() * weights[:, first:last].unsqueeze(-1)
                del per_head
                scores = weighted.sum(dim=2)
                del weighted
                if rank_sum:
                    dtype = scores.dtype
                    total = accumulator[: batch * (last - first) * length].view(
                        batch, last - first, length
                    )
                    total.copy_(scores)
                    del scores
                    dist.all_reduce(total)
                    scores = total.to(dtype, copy=True)
                    del total
        yield first, last, scores
        del scores
