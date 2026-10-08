"""DSpark head-128 preparation and context-plus-block attention.

Context KV is persistent and token-indexed in 784-token pages. The seven-row
draft block stays in separate tensors: all queries can see all its rows, and
rejected noise never enters a shared context page.
"""

import math

import torch

from .cuda_backend import compiled


def rms_norm(x, weight, eps=1e-6):
    """Qwen3 hidden RMS with BF16 rounding before the learned multiplier."""
    from oh_my_vllm.ir.dspark import hidden_norm

    return hidden_norm(x, weight, eps)


def _validate_rms(x, weight, eps):
    if x.ndim < 2 or x.shape[-1] != 5120 or not x.numel():
        raise ValueError("DSpark hidden RMS requires nonempty [...,5120] rows")
    if (
        x.dtype != torch.bfloat16
        or weight.shape != (5120,)
        or weight.dtype
        not in (
            torch.bfloat16,
            torch.float32,
        )
    ):
        raise ValueError("DSpark hidden RMS needs BF16 data and 5120 multipliers")
    if not 0 < eps < 1:
        raise ValueError("DSpark hidden RMS epsilon must be in (0,1)")
    if any(
        not tensor.is_cuda or tensor.device != x.device or not tensor.is_contiguous()
        for tensor in (x, weight)
    ):
        raise ValueError(
            "DSpark hidden RMS requires contiguous tensors on one CUDA device"
        )


def _rms_cuda(x, weight, eps):
    _validate_rms(x, weight, eps)
    output = torch.empty_like(x)
    compiled().dspark_rms_norm(
        x.reshape(-1, 5120), weight, output.reshape(-1, 5120), eps
    )
    return output


def normalize_rope(x, weight, positions, inv_freq, attention_factor, eps=1e-6):
    """RMS, checkpoint BF16 multiplier, and full NeoX YaRN rotation.

    Preserve BF16 rounding after normalization, multiplication, cosine/sine and
    each rotary product. Positions times the supplied FP32 frequencies use FP64
    phase reduction so large positions do not lose precision before trigonometry.
    """
    from oh_my_vllm.ir.dspark import norm_rope

    return norm_rope(x, weight, positions, inv_freq, attention_factor, eps)


def prepare_qk(
    q, k, q_weight, k_weight, positions, inv_freq, attention_factor, eps=1e-6
):
    """Prepare either projection independently; context injection needs only K."""
    flat_positions = positions.reshape(-1)
    query = (
        normalize_rope(
            q.reshape(-1, q.shape[-2], 128),
            q_weight,
            flat_positions,
            inv_freq,
            attention_factor,
            eps,
        ).reshape(q.shape)
        if q is not None
        else None
    )
    key = (
        normalize_rope(
            k.reshape(-1, k.shape[-2], 128),
            k_weight,
            flat_positions,
            inv_freq,
            attention_factor,
            eps,
        ).reshape(k.shape)
        if k is not None
        else None
    )
    return query, key


def append(cache, key, value, slots):
    """Write retained context KV; negative slots suppress an individual write."""
    from oh_my_vllm.ir.dspark import append_context

    append_context(cache, key, value, slots)


def attention(query, cache, tables, context_lengths, block_key, block_value):
    """Return [requests,7,32,128] from each context plus its complete own block."""
    from oh_my_vllm.ir.dspark import block_attention

    return block_attention(
        query, cache, tables, context_lengths, block_key, block_value
    )


def _validate_norm(x, weight, positions, inv_freq, attention_factor, eps):
    if x.ndim != 3 or x.shape[-1] != 128 or x.numel() == 0:
        raise ValueError("DSpark normalization requires nonempty [tokens,heads,128]")
    if (
        x.dtype != torch.bfloat16
        or weight.shape != (128,)
        or weight.dtype != torch.float32
    ):
        raise ValueError(
            "DSpark normalization needs BF16 rows and FP32 checkpoint multipliers"
        )
    if inv_freq.shape != (64,) or inv_freq.dtype != torch.float32:
        raise ValueError("DSpark full RoPE requires 64 FP32 inverse frequencies")
    if positions.shape != (len(x),) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("DSpark normalization needs one integer position per token")
    if not math.isfinite(attention_factor) or attention_factor <= 0 or not 0 < eps < 1:
        raise ValueError("invalid DSpark attention factor or RMS epsilon")
    if x.stride(2) != 1 or x.stride(1) != 128 or x.stride(0) < x.shape[1] * 128:
        raise ValueError(
            "DSpark normalization requires dense non-overlapping head rows"
        )
    if any(
        not t.is_cuda or t.device != x.device for t in (x, weight, positions, inv_freq)
    ):
        raise ValueError("DSpark normalization inputs must share one CUDA device")
    if any(not t.is_contiguous() for t in (weight, positions, inv_freq)):
        raise ValueError("DSpark normalization parameters must be contiguous")


def _normalize_rope_cuda(x, weight, positions, inv_freq, attention_factor, eps):
    _validate_norm(x, weight, positions, inv_freq, attention_factor, eps)
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    compiled().dspark_norm_rope(
        x, weight, positions, inv_freq, output, attention_factor, eps
    )
    return output


def _append_cuda(cache, key, value, slots):
    if cache.ndim != 5 or cache.shape[1:] != (2, 784, 8, 128):
        raise ValueError("DSpark context cache must be [pages,2,784,8,128]")
    if (
        key.ndim != 3
        or key.shape[1:] != (8, 128)
        or value.shape != key.shape
        or not len(key)
    ):
        raise ValueError("DSpark context KV must be nonempty [tokens,8,128]")
    if slots.shape != (len(key),) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("DSpark context append needs one integer slot per token")
    data = (cache, key, value)
    if any(t.dtype != torch.bfloat16 for t in data):
        raise ValueError("DSpark context KV must be BF16")
    if any(
        not t.is_cuda or t.device != cache.device or not t.is_contiguous()
        for t in (*data, slots)
    ):
        raise ValueError(
            "DSpark context append inputs must be contiguous on one CUDA device"
        )
    compiled().dspark_append(key, value, cache, slots)


def _validate_attention(query, cache, tables, context_lengths, block_key, block_value):
    if query.ndim != 4 or query.shape[1:] != (7, 32, 128) or not len(query):
        raise ValueError("DSpark queries must be nonempty [requests,7,32,128]")
    requests = len(query)
    if cache.ndim != 5 or cache.shape[1:] != (2, 784, 8, 128):
        raise ValueError("DSpark context cache must be [pages,2,784,8,128]")
    if block_key.shape != (requests, 7, 8, 128) or block_value.shape != block_key.shape:
        raise ValueError("DSpark noise KV must be [requests,7,8,128]")
    if tables.ndim != 2 or tables.shape[0] != requests or not tables.shape[1]:
        raise ValueError("DSpark attention needs one nonempty page table per request")
    if context_lengths.shape != (requests,) or any(
        t.dtype != torch.int32 for t in (tables, context_lengths)
    ):
        raise ValueError("DSpark context lengths and tables must use int32")
    data = (query, cache, block_key, block_value)
    if any(t.dtype != torch.bfloat16 for t in data):
        raise ValueError("DSpark attention data must be BF16")
    if any(
        not t.is_cuda or t.device != query.device or not t.is_contiguous()
        for t in (*data, tables, context_lengths)
    ):
        raise ValueError(
            "DSpark attention inputs must be contiguous on one CUDA device"
        )


def _attention_cuda(query, cache, tables, context_lengths, block_key, block_value):
    _validate_attention(query, cache, tables, context_lengths, block_key, block_value)
    requests = len(query)
    splits = 16 if requests >= 4 and tables.shape[1] * 784 <= 65536 else 64
    partial = torch.empty(
        (requests * 7, 32, splits, 128), device=query.device, dtype=torch.float32
    )
    lse = torch.empty(
        (requests * 7, 32, splits), device=query.device, dtype=torch.float32
    )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    implementation = compiled()
    implementation.dspark_attention_partial(
        query, cache, tables, context_lengths, block_key, block_value, partial, lse
    )
    implementation.dspark_attention_merge(partial, lse, output)
    return output
