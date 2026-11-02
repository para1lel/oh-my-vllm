"""Bound temporary index scores without changing the per-head math."""

from collections.abc import Iterator
from numbers import Integral

import torch

INDEX_SCORE_WORK_BYTES = 256 * 1024 * 1024


def score_chunks(
    query: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    *,
    work_bytes: int = INDEX_SCORE_WORK_BYTES,
    rank_sum: bool = False,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield query ranges and scores for [B,S,H,D], [B,T,D], [B,S,H].

    Inputs must have one device and one FP16, BF16, FP32, or FP64 dtype.
    H and D must be positive. Autocast must not mix FP16 and BF16.
    This inference helper does not record gradients. Each row reserves
    B * max(H*(D+T), 2*H*T, (H+1)*T) * element_size bytes for query packing,
    the per-head product, and the head reduction in separate lifetime phases.
    Input storage, caller-held outputs, and backend workspaces are not included.
    With rank_sum, FP16/BF16 rank sums use FP32 and round once to the score dtype.
    A reusable rank accumulator adds B*T*accumulator_size bytes per row to every
    phase. Returned scores do not share this accumulator's storage.
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
    row_bytes = (
        batch
        * max(
            heads * (dim + length),
            2 * heads * length,
            (heads + 1) * length,
        )
        * query.element_size()
    )
    if rank_sum:
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
    if rank_sum:
        import torch.distributed as dist

        accumulator = torch.empty(
            batch * min(chunk_rows, rows) * length,
            dtype=accumulator_dtype,
            device=query.device,
        )
    for first in range(0, rows, chunk_rows):
        last = min(first + chunk_rows, rows)
        with torch.no_grad():
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
