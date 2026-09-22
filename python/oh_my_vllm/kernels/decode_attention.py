"""Split-KV GQA decode with an explicit first valid cache position.

MTP cache rows are indexed by their input token (target position + 1), so row
zero is absent. Excluding it here keeps prefix cache keys aligned with tokens
without modifying shared boundary rows or changing the 784-token page size.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _partials(
    h: int,
    hk: int,
    pages: int,
    table_width: int,
    splits: int,
    first: int,
    bk: int,
    bq: int,
    grouped: bool,
    index_types: tuple,
):
    table_dtype, length_dtype, start_dtype = index_types
    n = T.dynamic("n")
    groups = T.dynamic("groups") if grouped else n
    starts_size = groups + 1 if grouped else n
    ratio = h // hk
    # Read multiple KV rows per warp group with coalesced eight-element vectors.
    gather_layout = tilelang.layout.Fragment(
        (bk, 256),
        forward_thread_fn=lambda i, j: (i % 8) * 32 + j // 8,
        forward_index_fn=lambda i, j: (i // 8) * 8 + j % 8,
    )

    @T.prim_func
    def kernel(
        query: T.Tensor((n, h, 256), "bfloat16"),
        cache: T.Tensor((pages, 2, 784, hk, 256), "bfloat16"),
        tables: T.Tensor((n, table_width), table_dtype),
        lengths: T.Tensor((n,), length_dtype),
        starts: T.Tensor((starts_size,), start_dtype),
        partial: T.Tensor((n, h, splits, 256), "float32"),
        lse: T.Tensor((n, h, splits), "float32"),
    ):
        with T.Kernel(groups, hk, splits, threads=256) as (seq, kh, split):
            q_shared = T.alloc_shared((bq, 256), "bfloat16")
            k_shared = T.alloc_shared((bk, 256), "bfloat16")
            v_shared = T.alloc_shared((bk, 256), "bfloat16")
            p_shared = T.alloc_shared((bq, bk), "bfloat16")
            score = T.alloc_fragment((bq, bk), "float32")
            acc = T.alloc_fragment((bq, 256), "float32")
            maximum = T.alloc_fragment((bq,), "float32")
            next_max = T.alloc_fragment((bq,), "float32")
            denom = T.alloc_fragment((bq,), "float32")
            block_sum = T.alloc_fragment((bq,), "float32")
            alpha = T.alloc_fragment((bq,), "float32")
            start = T.if_then_else(grouped, starts[seq], seq)
            end = T.if_then_else(grouped, starts[seq + 1], seq + 1)
            last = lengths[end - 1]
            # Partition the live extent on device; graph replay can change lengths.
            chunk = T.ceildiv(T.max(last - first, 0), splits * bk) * bk
            begin = first + split * chunk
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(bq, 256):
                row = start + i // ratio
                q_shared[i, j] = T.if_then_else(
                    row < end, query[row, kh * ratio + i % ratio, j], 0
                )
            for step in T.serial(T.ceildiv(T.max(T.min(chunk, last - begin), 0), bk)):
                for i, j in T.Parallel(bk, 256, loop_layout=gather_layout):
                    pos = begin + step * bk + i
                    if pos < last:
                        page = tables[start, pos // 784].astype("int64")
                        k_shared[i, j] = cache[page, 0, pos % 784, kh, j]
                        v_shared[i, j] = cache[page, 1, pos % 784, kh, j]
                    else:
                        k_shared[i, j] = 0
                        v_shared[i, j] = 0
                T.gemm(q_shared, k_shared, score, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(bq, bk):
                    row = start + i // ratio
                    pos = begin + step * bk + j
                    if row < end:
                        score[i, j] = T.if_then_else(
                            pos < lengths[row],
                            score[i, j] * (256**-0.5 * 1.4426950408889634),
                            -T.infinity("float32"),
                        )
                    else:
                        score[i, j] = -T.infinity("float32")
                T.reduce_max(score, next_max, dim=1)
                for i in T.Parallel(bq):
                    next_max[i] = T.max(maximum[i], next_max[i])
                    safe_max = T.if_then_else(
                        next_max[i] == -T.infinity("float32"), 0, next_max[i]
                    )
                    alpha[i] = T.exp2(maximum[i] - safe_max)
                for i, j in T.Parallel(bq, bk):
                    safe_max = T.if_then_else(
                        next_max[i] == -T.infinity("float32"), 0, next_max[i]
                    )
                    score[i, j] = T.exp2(score[i, j] - safe_max)
                    p_shared[i, j] = score[i, j]
                T.reduce_sum(score, block_sum, dim=1)
                for i in T.Parallel(bq):
                    denom[i] = denom[i] * alpha[i] + block_sum[i]
                    maximum[i] = next_max[i]
                for i, j in T.Parallel(bq, 256):
                    acc[i, j] *= alpha[i]
                T.gemm(p_shared, v_shared, acc)
            for i, j in T.Parallel(bq, 256):
                row = start + i // ratio
                if row < end:
                    partial[row, kh * ratio + i % ratio, split, j] = acc[
                        i, j
                    ] / T.if_then_else(denom[i] > 0, denom[i], 1)
            for i in T.Parallel(bq):
                row = start + i // ratio
                if row < end:
                    lse[row, kh * ratio + i % ratio, split] = maximum[i] + T.log2(
                        T.if_then_else(denom[i] > 0, denom[i], 1)
                    )

    return kernel


@tilelang.jit
def _merge(h: int, splits: int):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        partial: T.Tensor((n, h, splits, 256), "float32"),
        lse: T.Tensor((n, h, splits), "float32"),
        out: T.Tensor((n, h, 256), "bfloat16"),
    ):
        with T.Kernel(n, h, threads=128) as (row, head):
            logs = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            total = T.alloc_fragment((1,), "float32")
            weighted = T.alloc_fragment((splits, 256), "float32")
            result = T.alloc_fragment((256,), "float32")
            for i in T.Parallel(splits):
                logs[i] = lse[row, head, i]
            T.reduce_max(logs, maximum, dim=0)
            for i in T.Parallel(splits):
                logs[i] = T.exp2(logs[i] - maximum[0])
            T.reduce_sum(logs, total, dim=0)
            for i, j in T.Parallel(splits, 256):
                weighted[i, j] = partial[row, head, i, j] * logs[i]
            T.reduce_sum(weighted, result, dim=0)
            for j in T.Parallel(256):
                out[row, head, j] = result[j] / total[0]

    return kernel


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
    splits = 64 if starts is not None else 128
    block = 64
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
    )(query, cache, tables, lengths, offsets, partial, lse)
    _merge(heads, splits)(partial, lse, out)
    return out
