"""Supplemental DSpark comparison; the original TileLang snapshot is unchanged.

The Qwen3 BF16 norm boundary, full head-128 YaRN and bidirectional seven-row
block differ from the target model's frozen kernels. This separate source is
pinned before formal timing and is never a CUDA provider fallback.
"""

import math

import tilelang
import tilelang.language as T
import torch

from .tilelang_reference.attention import append as _frozen_append


@tilelang.jit
def _rms(eps: float, weight_dtype: str):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        x: T.Tensor((n, 5120), "bfloat16"),
        weight: T.Tensor((5120,), weight_dtype),
        out: T.Tensor((n, 5120), "bfloat16"),
    ):
        with T.Kernel(n, threads=128) as row:
            values = T.alloc_fragment((8192,), "float32")
            squares = T.alloc_fragment((8192,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(8192):
                values[i] = T.if_then_else(i < 5120, x[row, i].astype("float32"), 0)
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(8192):
                if i < 5120:
                    normalized = (values[i] * T.rsqrt(total[0] / 5120 + eps)).astype(
                        "bfloat16"
                    )
                    multiplier = weight[i].astype("bfloat16")
                    out[row, i] = normalized.astype("float32") * multiplier.astype(
                        "float32"
                    )

    return kernel


def rms_norm(x, weight, eps=1e-6):
    if x.ndim < 2 or x.shape[-1] != 5120 or not x.numel():
        raise ValueError("DSpark hidden RMS requires nonempty [...,5120] rows")
    if (
        x.dtype != torch.bfloat16
        or weight.shape != (5120,)
        or weight.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("DSpark hidden RMS needs BF16 rows and 5120 multipliers")
    if not 0 < eps < 1 or any(
        not t.is_cuda or not t.is_contiguous() or t.device != x.device
        for t in (x, weight)
    ):
        raise ValueError("invalid DSpark hidden RMS epsilon or CUDA layout")
    out = torch.empty_like(x)
    _rms(eps, str(weight.dtype).removeprefix("torch."))(
        x.reshape(-1, 5120), weight, out.reshape(-1, 5120)
    )
    return out


@tilelang.jit
def _norm_rope(heads: int, strides: tuple, index_dtype: str, factor: float, eps: float):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        x: T.StridedTensor((n, heads, 128), strides, "bfloat16"),
        weight: T.Tensor((128,), "float32"),
        positions: T.Tensor((n,), index_dtype),
        inv_freq: T.Tensor((64,), "float32"),
        out: T.Tensor((n, heads, 128), "bfloat16"),
    ):
        with T.Kernel(n * heads, threads=128) as row:
            values = T.alloc_fragment((128,), "float32")
            squares = T.alloc_fragment((128,), "float32")
            weighted = T.alloc_fragment((128,), "bfloat16")
            total = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(128):
                values[i] = x[row // heads, row % heads, i].astype("float32")
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(128):
                normalized = (values[i] * T.rsqrt(total[0] / 128 + eps)).astype(
                    "bfloat16"
                )
                multiplier = weight[i].astype("bfloat16")
                weighted[i] = normalized.astype("float32") * multiplier.astype(
                    "float32"
                )
            for i in T.Parallel(128):
                angle64 = positions[row // heads].astype("float64") * inv_freq[
                    i % 64
                ].astype("float64")
                angle = (
                    angle64
                    - T.round(angle64 / T.float64(math.tau)) * T.float64(math.tau)
                ).astype("float32")
                cosine = (T.cos(angle) * factor).astype("bfloat16").astype("float32")
                sine = (T.sin(angle) * factor).astype("bfloat16").astype("float32")
                other = T.if_then_else(i < 64, i + 64, i - 64)
                # Read the paired column from global memory rather than asking
                # a fragment lane for a value owned by another thread/warp.
                other_normalized = (
                    x[row // heads, row % heads, other].astype("float32")
                    * T.rsqrt(total[0] / 128 + eps)
                ).astype("bfloat16")
                other_weighted = (
                    other_normalized.astype("float32")
                    * weight[other].astype("bfloat16").astype("float32")
                ).astype("bfloat16")
                own_product = (weighted[i].astype("float32") * cosine).astype(
                    "bfloat16"
                )
                other_product = (other_weighted.astype("float32") * sine).astype(
                    "bfloat16"
                )
                out[row // heads, row % heads, i] = own_product.astype("float32") + (
                    T.if_then_else(i < 64, -1, 1) * other_product.astype("float32")
                )

    return kernel


def normalize_rope(x, weight, positions, inv_freq, attention_factor, eps=1e-6):
    if x.ndim != 3 or x.shape[-1] != 128 or not x.numel():
        raise ValueError("DSpark Q/K rows must be nonempty [tokens,heads,128]")
    if (
        x.dtype != torch.bfloat16
        or weight.shape != (128,)
        or weight.dtype != torch.float32
    ):
        raise ValueError("DSpark Q/K requires BF16 data and 128 FP32 multipliers")
    if positions.shape != (len(x),) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("DSpark Q/K needs one integer position per token")
    if inv_freq.shape != (64,) or inv_freq.dtype != torch.float32:
        raise ValueError("DSpark YaRN needs 64 FP32 inverse frequencies")
    if not math.isfinite(attention_factor) or attention_factor <= 0 or not 0 < eps < 1:
        raise ValueError("invalid DSpark YaRN scale or norm epsilon")
    if x.stride(2) != 1 or x.stride(1) != 128 or x.stride(0) < x.shape[1] * 128:
        raise ValueError("DSpark Q/K requires dense nonoverlapping head rows")
    if any(
        not t.is_cuda or t.device != x.device for t in (x, weight, positions, inv_freq)
    ):
        raise ValueError("DSpark Q/K inputs must share a CUDA device")
    if any(not t.is_contiguous() for t in (weight, positions, inv_freq)):
        raise ValueError("DSpark Q/K parameters must be contiguous")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _norm_rope(
        x.shape[1],
        x.stride(),
        str(positions.dtype).removeprefix("torch."),
        attention_factor,
        eps,
    )(x, weight, positions, inv_freq, out)
    return out


def append(cache, key, value, slots):
    """The immutable generic append already supports eight head-128 KV heads."""
    _frozen_append(cache, key, value, slots)


@tilelang.jit
def _partials(pages: int, table_width: int, splits: int):
    requests = T.dynamic("requests")
    bq, bk, heads, kv_heads, dim, ratio = 32, 64, 32, 8, 128, 4

    @T.prim_func
    def kernel(
        query: T.Tensor((requests, 7, heads, dim), "bfloat16"),
        cache: T.Tensor((pages, 2, 784, kv_heads, dim), "bfloat16"),
        tables: T.Tensor((requests, table_width), "int32"),
        contexts: T.Tensor((requests,), "int32"),
        block_key: T.Tensor((requests, 7, kv_heads, dim), "bfloat16"),
        block_value: T.Tensor((requests, 7, kv_heads, dim), "bfloat16"),
        partial: T.Tensor((requests * 7, heads, splits, dim), "float32"),
        lse: T.Tensor((requests * 7, heads, splits), "float32"),
    ):
        with T.Kernel(requests, kv_heads, splits, threads=128) as (seq, kh, split):
            q_shared = T.alloc_shared((bq, dim), "bfloat16")
            k_shared = T.alloc_shared((bk, dim), "bfloat16")
            v_shared = T.alloc_shared((bk, dim), "bfloat16")
            p_shared = T.alloc_shared((bq, bk), "bfloat16")
            score = T.alloc_fragment((bq, bk), "float32")
            acc = T.alloc_fragment((bq, dim), "float32")
            maximum = T.alloc_fragment((bq,), "float32")
            next_max = T.alloc_fragment((bq,), "float32")
            denom = T.alloc_fragment((bq,), "float32")
            block_sum = T.alloc_fragment((bq,), "float32")
            alpha = T.alloc_fragment((bq,), "float32")
            context = contexts[seq]
            last = context + 7
            chunk = T.ceildiv(last, splits * bk) * bk
            begin = split * chunk
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(bq, dim):
                q_shared[i, j] = T.if_then_else(
                    i < 7 * ratio, query[seq, i // ratio, kh * ratio + i % ratio, j], 0
                )
            for step in T.serial(T.ceildiv(T.max(T.min(chunk, last - begin), 0), bk)):
                for i, j in T.Parallel(bk, dim // 8):
                    pos = begin + step * bk + i
                    if pos < context:
                        page = tables[seq, pos // 784].astype("int64")
                        T.ptx_cp_async(
                            T.access_ptr(k_shared[i, j * 8], "w", 8),
                            T.access_ptr(cache[page, 0, pos % 784, kh, j * 8], "r", 8),
                            8,
                            True,
                        )
                        T.ptx_cp_async(
                            T.access_ptr(v_shared[i, j * 8], "w", 8),
                            T.access_ptr(cache[page, 1, pos % 784, kh, j * 8], "r", 8),
                            8,
                            True,
                        )
                    else:
                        local = T.min(pos - context, 6)
                        T.ptx_cp_async(
                            T.access_ptr(k_shared[i, j * 8], "w", 8),
                            T.access_ptr(block_key[seq, local, kh, j * 8], "r", 8),
                            8,
                            pos < last,
                        )
                        T.ptx_cp_async(
                            T.access_ptr(v_shared[i, j * 8], "w", 8),
                            T.access_ptr(block_value[seq, local, kh, j * 8], "r", 8),
                            8,
                            pos < last,
                        )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.gemm(q_shared, k_shared, score, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(bq, bk):
                    score[i, j] = T.if_then_else(
                        i < 7 * ratio and begin + step * bk + j < last,
                        score[i, j] * (128**-0.5 * 1.4426950408889634),
                        -T.infinity("float32"),
                    )
                T.reduce_max(score, next_max, dim=1)
                for i in T.Parallel(bq):
                    next_max[i] = T.max(maximum[i], next_max[i])
                    safe = T.if_then_else(
                        next_max[i] == -T.infinity("float32"), 0, next_max[i]
                    )
                    alpha[i] = T.exp2(maximum[i] - safe)
                for i, j in T.Parallel(bq, bk):
                    safe = T.if_then_else(
                        next_max[i] == -T.infinity("float32"), 0, next_max[i]
                    )
                    score[i, j] = T.exp2(score[i, j] - safe)
                    p_shared[i, j] = score[i, j]
                T.reduce_sum(score, block_sum, dim=1)
                for i in T.Parallel(bq):
                    denom[i] = denom[i] * alpha[i] + block_sum[i]
                    maximum[i] = next_max[i]
                for i, j in T.Parallel(bq, dim):
                    acc[i, j] *= alpha[i]
                T.gemm(p_shared, v_shared, acc)
            for i, j in T.Parallel(bq, dim):
                if i < 7 * ratio:
                    partial[seq * 7 + i // ratio, kh * ratio + i % ratio, split, j] = (
                        acc[i, j] / (T.if_then_else(denom[i] > 0, denom[i], 1))
                    )
            for i in T.Parallel(bq):
                if i < 7 * ratio:
                    lse[seq * 7 + i // ratio, kh * ratio + i % ratio, split] = maximum[
                        i
                    ] + T.log2(T.if_then_else(denom[i] > 0, denom[i], 1))

    return kernel


@tilelang.jit
def _merge(splits: int):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        partial: T.Tensor((n, 32, splits, 128), "float32"),
        lse: T.Tensor((n, 32, splits), "float32"),
        out: T.Tensor((n, 32, 128), "bfloat16"),
    ):
        with T.Kernel(n, 32, threads=128) as (row, head):
            logs = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            total = T.alloc_fragment((1,), "float32")
            weights = T.alloc_shared((splits,), "float32")
            result = T.alloc_fragment((128,), "float32")
            for i in T.Parallel(splits):
                logs[i] = lse[row, head, i]
            T.reduce_max(logs, maximum, dim=0)
            for i in T.Parallel(splits):
                logs[i] = T.exp2(logs[i] - maximum[0])
            T.reduce_sum(logs, total, dim=0)
            T.copy(logs, weights)
            T.clear(result)
            for i in T.serial(splits):
                for j in T.Parallel(128):
                    result[j] += partial[row, head, i, j] * weights[i]
            for j in T.Parallel(128):
                out[row, head, j] = result[j] / total[0]

    return kernel


def attention(query, cache, tables, context_lengths, block_key, block_value):
    if query.ndim != 4 or query.shape[1:] != (7, 32, 128) or not len(query):
        raise ValueError("DSpark queries must be nonempty [requests,7,32,128]")
    if cache.ndim != 5 or cache.shape[1:] != (2, 784, 8, 128):
        raise ValueError("DSpark cache must be [pages,2,784,8,128]")
    if (
        block_key.shape != (len(query), 7, 8, 128)
        or block_value.shape != block_key.shape
    ):
        raise ValueError("DSpark block KV must match the seven query rows")
    if tables.ndim != 2 or tables.shape[0] != len(query) or not tables.shape[1]:
        raise ValueError("DSpark attention needs one nonempty page table per request")
    if context_lengths.shape != (len(query),) or any(
        t.dtype != torch.int32 for t in (tables, context_lengths)
    ):
        raise ValueError("DSpark attention metadata must be int32")
    data = (query, cache, block_key, block_value)
    if any(t.dtype != torch.bfloat16 for t in data) or any(
        not t.is_cuda or not t.is_contiguous() or t.device != query.device
        for t in (*data, tables, context_lengths)
    ):
        raise ValueError(
            "DSpark attention needs contiguous BF16 data on one CUDA device"
        )
    torch._assert_async(
        torch.all((context_lengths >= 0) & (context_lengths <= tables.shape[1] * 784)),
        "DSpark context length exceeds its page table",
    )
    # This comparator uses vector staging. Include necessary aligned snapshots
    # in the measured operation; CUDA's scalar layout dispatch needs no copy.
    query, cache, block_key, block_value = tuple(
        x.clone() if x.data_ptr() % 16 else x for x in data
    )
    splits = 16 if len(query) >= 4 and tables.shape[1] * 784 <= 65536 else 64
    partial = torch.empty((len(query) * 7, 32, splits, 128), device=query.device)
    lse = torch.empty((len(query) * 7, 32, splits), device=query.device)
    out = torch.empty_like(query)
    _partials(len(cache), tables.shape[1], splits)(
        query, cache, tables, context_lengths, block_key, block_value, partial, lse
    )
    _merge(splits)(partial, lse, out.reshape(-1, 32, 128))
    return out
