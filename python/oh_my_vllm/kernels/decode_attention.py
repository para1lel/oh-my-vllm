"""Split-KV GQA decode with an explicit first valid cache position.

MTP cache rows are indexed by their input token (target position + 1), so row
zero is absent. Excluding it here keeps prefix cache keys aligned with tokens
without modifying shared boundary rows or changing the 784-token page size.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partials(
    Q,
    Cache,
    Tables,
    Lengths,
    Partial,
    LSE,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    TableWidth: tl.constexpr,
    Splits: tl.constexpr,
    Chunk: tl.constexpr,
    First: tl.constexpr,
    BK: tl.constexpr,
):
    req, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = kv_head * (H // HK) + tl.arange(0, 16)
    active_heads = tl.arange(0, 16) < H // HK
    dims = tl.arange(0, D)
    q = tl.load(
        Q + (req * H + heads[:, None]) * D + dims[None, :], active_heads[:, None], 0
    )
    length = tl.load(Lengths + req)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    accumulator = tl.full((16, D), 0, tl.float32)
    begin = First + split * Chunk
    for block in range(begin, tl.minimum(begin + Chunk, length), BK):
        positions = block + tl.arange(0, BK)
        valid = positions < length
        page = tl.load(Tables + req * TableWidth + positions // 784, valid, 0)
        offset = page * (2 * 784 * HK * D) + (positions % 784) * HK * D + kv_head * D
        k = tl.load(Cache + offset[None, :] + dims[:, None], valid[None, :], 0)
        v = tl.load(
            Cache + offset[:, None] + 784 * HK * D + dims[None, :], valid[:, None], 0
        )
        score = tl.dot(q, k) * (D**-0.5 * 1.4426950408889634)
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(score - safe_max[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        accumulator = accumulator * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
        maximum = next_max
    normalizer = tl.where(denominator > 0, denominator, 1)
    output = accumulator / normalizer[:, None]
    logsum = maximum + tl.log2(normalizer)
    address = (req * H + heads) * Splits + split
    tl.store(
        Partial + address[:, None] * D + dims[None, :], output, active_heads[:, None]
    )
    tl.store(LSE + address, logsum, active_heads)


@triton.jit
def _merge(Partial, LSE, Out, D: tl.constexpr, Splits: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, Splits)
    dim = tl.arange(0, D)
    lse = tl.load(LSE + row * Splits + split)
    maximum = tl.max(lse, 0)
    weight = tl.exp2(lse - maximum)
    partial = tl.load(Partial + (row * Splits + split[:, None]) * D + dim[None, :])
    out = tl.sum(partial * weight[:, None], 0) / tl.sum(weight, 0)
    tl.store(Out + row * D + dim, out)


def decode(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    lengths: torch.Tensor,
    *,
    first: int = 0,
    max_tokens: int = 65536,
) -> torch.Tensor:
    """One query per sequence. Host planning validates table IDs and lengths."""
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
    splits = 32
    partial = torch.empty((requests, heads, splits, dim), device=query.device)
    lse = torch.empty((requests, heads, splits), device=query.device)
    out = torch.empty_like(query)
    chunk = triton.cdiv(max_tokens, splits * 128) * 128
    _partials[(requests, kv_heads, splits)](
        query,
        cache,
        tables,
        lengths,
        partial,
        lse,
        heads,
        kv_heads,
        dim,
        tables.shape[1],
        splits,
        chunk,
        first,
        128,
        num_warps=4,
    )
    _merge[(requests * heads,)](partial, lse, out, dim, splits, num_warps=4)
    return out
