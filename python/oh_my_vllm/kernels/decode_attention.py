"""Split-KV GQA decode with an explicit first valid cache position.

MTP cache rows are indexed by their input token (target position + 1), so row
zero is absent. Excluding it here keeps prefix cache keys aligned with tokens
without modifying shared boundary rows or changing the 784-token page size.
"""

import torch

from .backend import NAME, kernel

_partials = kernel("decode_attention", "_partials")


_merge = kernel("decode_attention", "_merge")


def decode(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    lengths: torch.Tensor,
    *,
    first: int = 0,
    max_tokens: int = 65536,
    starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged causal decode, optionally sharing KV reads across verification rows.

    Host planning validates table IDs, lengths and (when supplied) starts: each
    group has 1..5 contiguous query rows, identical tables and increasing lengths.
    The per-row length still masks every candidate's attention independently.
    """
    if query.ndim != 3 or query.shape[-1] != 256 or query.numel() == 0:
        raise ValueError("decode query must be nonempty [requests,heads,256]")
    if cache.ndim != 5 or cache.shape[1:3] != (2, 784) or cache.shape[-1] != 256:
        raise ValueError("decode cache must be [pages,2,784,kv_heads,256]")
    requests, heads, dim = query.shape
    kv_heads = cache.shape[3]
    if kv_heads == 0 or heads % kv_heads or not 0 < heads // kv_heads <= 16:
        raise ValueError("unsupported decode head ratio")
    if tables.ndim != 2 or tables.shape[0] != requests or lengths.shape != (requests,):
        raise ValueError("decode metadata must match request count")
    if first not in (0, 1) or max_tokens <= first or tables.shape[1] * 784 < max_tokens:
        raise ValueError("invalid decode extent or first position")
    if any(t.dtype != torch.bfloat16 for t in (query, cache)) or any(
        t.dtype not in (torch.int32, torch.int64) for t in (tables, lengths)
    ):
        raise ValueError("decode requires BF16 data and integer metadata")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != query.device
        for t in (query, cache, tables, lengths)
    ):
        raise ValueError("decode tensors must be contiguous on one CUDA device")
    if starts is not None and (
        starts.ndim != 1
        or starts.numel() < 2
        or starts.dtype not in (torch.int32, torch.int64)
        or starts.device != query.device
        or not starts.is_contiguous()
    ):
        raise ValueError("grouped decode starts must be contiguous CUDA integers")
    # Both backends issue16-byte KV copies. Preserve support for contiguous
    # views that start at an unaligned BF16 storage offset.
    if cache.data_ptr() % 16:
        cache = cache.clone()
    # The frozen TileLang code vector-loads Q. Contiguous BF16 storage-offset
    # views need an aligned copy; native CUDA handles their scalar load directly.
    if NAME == "tilelang" and query.data_ptr() % 16:
        query = query.clone()
    # Keep the reduction traffic bounded for small batches; both kernels are
    # included when selecting split counts, not just the partial attention.
    splits = 64
    if starts is not None and starts.numel() >= 5 and max_tokens <= 65536:
        # Verification shares each KV tile among several query rows. Larger
        # batches fill the GPU with fewer splits and less merge traffic.
        splits = 16
    block = 64
    if starts is None and max_tokens >= 16384:
        block, splits = 32, 128
    partial = torch.empty((requests, heads, splits, dim), device=query.device)
    lse = torch.empty((requests, heads, splits), device=query.device)
    out = torch.empty_like(query)
    offsets = starts if starts is not None else lengths
    types = tuple(
        str(t.dtype).removeprefix("torch.") for t in (tables, lengths, offsets)
    )
    query_tile = (
        max(16, 1 << (5 * (heads // kv_heads) - 1).bit_length())
        if starts is not None
        else 16
    )
    _partials(
        heads,
        kv_heads,
        len(cache),
        tables.shape[1],
        splits,
        first,
        block,
        query_tile,
        starts is not None,
        types,
        "int32" if max_tokens <= 2**31 - splits * block else "int64",
    )(query, cache, tables, lengths, offsets, partial, lse)
    _merge(heads, splits)(partial, lse, out)
    return out
