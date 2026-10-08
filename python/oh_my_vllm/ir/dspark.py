"""DSpark semantic nodes with explicit CUDA or supplemental TileLang providers."""

from __future__ import annotations

import math

import torch

from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec


def _fake_rms(x, weight, eps):
    torch._check(x.ndim >= 2 and x.shape[-1] == 5120)
    torch._check(x.dtype == torch.bfloat16)
    torch._check(weight.shape == (5120,))
    torch._check(weight.dtype in (torch.bfloat16, torch.float32))
    torch._check(0 < eps < 1)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _reference_rms(x, weight, eps):
    values = x.double()
    normalized = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    ).bfloat16()
    return (normalized * weight.bfloat16()).bfloat16()


def _fake_norm(x, weight, positions, inv_freq, attention_factor, eps):
    torch._check(x.ndim == 3 and x.shape[-1] == 128)
    torch._check(weight.shape == (128,) and weight.dtype == torch.float32)
    torch._check(positions.shape == (x.shape[0],))
    torch._check(positions.dtype in (torch.int32, torch.int64))
    torch._check(inv_freq.shape == (64,) and inv_freq.dtype == torch.float32)
    torch._check(x.dtype == torch.bfloat16)
    torch._check(attention_factor > 0 and 0 < eps < 1)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _reference_norm(x, weight, positions, inv_freq, attention_factor, eps):
    """FP64 arithmetic, retaining each checkpoint BF16 rounding boundary."""
    values = x.double()
    normalized = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    ).bfloat16()
    normalized = (normalized * weight.bfloat16()).bfloat16()
    phase = positions.double()[:, None, None] * inv_freq.double()[None, None]
    cosine = (phase.cos() * attention_factor).bfloat16()
    sine = (phase.sin() * attention_factor).bfloat16()
    left, right = normalized[..., :64], normalized[..., 64:]
    first = (left * cosine).bfloat16() - (right * sine).bfloat16()
    second = (right * cosine).bfloat16() + (left * sine).bfloat16()
    return torch.cat((first, second), -1).bfloat16()


def _fake_append(cache, key, value, slots):
    torch._check(cache.ndim == 5 and cache.shape[1:] == (2, 784, 8, 128))
    torch._check(key.ndim == 3 and key.shape[1:] == (8, 128))
    torch._check(value.shape == key.shape and slots.shape == (key.shape[0],))
    torch._check(key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16)
    torch._check(cache.dtype == torch.bfloat16)
    torch._check(slots.dtype in (torch.int32, torch.int64))


def _reference_append(cache, key, value, slots):
    valid = slots >= 0
    selected = slots[valid].long()
    cache[selected // 784, 0, selected % 784] = key[valid]
    cache[selected // 784, 1, selected % 784] = value[valid]


def _fake_attention(query, cache, tables, context_lengths, block_key, block_value):
    torch._check(query.ndim == 4 and query.shape[1:] == (7, 32, 128))
    torch._check(cache.ndim == 5 and cache.shape[1:] == (2, 784, 8, 128))
    torch._check(tables.ndim == 2 and tables.shape[0] == query.shape[0])
    torch._check(context_lengths.shape == (query.shape[0],))
    torch._check(tables.dtype == torch.int32 and context_lengths.dtype == torch.int32)
    torch._check(block_key.shape == (query.shape[0], 7, 8, 128))
    torch._check(block_value.shape == block_key.shape)
    torch._check(
        all(x.dtype == torch.bfloat16 for x in (query, cache, block_key, block_value))
    )
    return torch.empty(query.shape, device=query.device, dtype=query.dtype)


def _reference_attention(query, cache, tables, context_lengths, block_key, block_value):
    """All block queries see the same retained prefix and all seven block rows."""
    outputs = []
    for request, length in enumerate(context_lengths.cpu().tolist()):
        if not 0 <= length <= tables.shape[1] * 784:
            raise ValueError("DSpark context length exceeds its page table")
        positions = torch.arange(length, device=query.device)
        pages = tables[request].long()[positions // 784]
        key = torch.cat(
            (cache[pages, 0, positions % 784], block_key[request]), 0
        ).double()
        value = torch.cat(
            (cache[pages, 1, positions % 784], block_value[request]), 0
        ).double()
        rows = query[request].double().reshape(7, 8, 4, 128)
        scores = torch.einsum("qhrd,khd->qhrk", rows, key) / math.sqrt(128)
        out = torch.einsum("qhrk,khd->qhrd", scores.softmax(-1), value)
        outputs.append(out.reshape(7, 32, 128).to(query.dtype))
    return torch.stack(outputs)


def _cuda_inputs(*args):
    return all(
        not isinstance(arg, TensorSpec) or arg.device.type == "cuda" for arg in args
    )


@torch.library.custom_op("oh_my_vllm_ir::dspark_rms_norm", mutates_args=())
def _ir_dspark_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return _DSPARK_RMS_NORM.select(x, weight, eps).op(x, weight, eps)


_ir_dspark_rms_norm.register_fake(_fake_rms)


@torch.library.custom_op("oh_my_vllm_native::dspark_rms_norm", mutates_args=())
def _native_dspark_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    _fake_rms(x, weight, eps)
    return _reference_rms(x, weight, eps)


_native_dspark_rms_norm.register_fake(_fake_rms)


@torch.library.custom_op("oh_my_vllm_kernel::dspark_rms_norm", mutates_args=())
def _kernel_dspark_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    if NAME == "tilelang":
        from oh_my_vllm.kernels.dspark_tilelang_reference import rms_norm

        return rms_norm(x, weight, eps)
    from oh_my_vllm.kernels.dspark_attention import _rms_cuda

    return _rms_cuda(x, weight, eps)


_kernel_dspark_rms_norm.register_fake(_fake_rms)
_DSPARK_RMS_NORM = Operation(
    "dspark_rms_norm",
    _ir_dspark_rms_norm,
    _native_dspark_rms_norm,
    default_priority=(NAME,),
)
_DSPARK_RMS_NORM.register_impl(NAME, _kernel_dspark_rms_norm, supports=_cuda_inputs)


def hidden_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _DSPARK_RMS_NORM(x, weight, eps)


@torch.library.custom_op("oh_my_vllm_ir::dspark_norm_rope", mutates_args=())
def _ir_dspark_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_factor: float,
    eps: float,
) -> torch.Tensor:
    return _DSPARK_NORM_ROPE.select(
        x, weight, positions, inv_freq, attention_factor, eps
    ).op(x, weight, positions, inv_freq, attention_factor, eps)


_ir_dspark_norm_rope.register_fake(_fake_norm)


@torch.library.custom_op("oh_my_vllm_native::dspark_norm_rope", mutates_args=())
def _native_dspark_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_factor: float,
    eps: float,
) -> torch.Tensor:
    _fake_norm(x, weight, positions, inv_freq, attention_factor, eps)
    return _reference_norm(x, weight, positions, inv_freq, attention_factor, eps)


_native_dspark_norm_rope.register_fake(_fake_norm)


@torch.library.custom_op("oh_my_vllm_kernel::dspark_norm_rope", mutates_args=())
def _kernel_dspark_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_factor: float,
    eps: float,
) -> torch.Tensor:
    if NAME == "tilelang":
        from oh_my_vllm.kernels.dspark_tilelang_reference import normalize_rope

        return normalize_rope(x, weight, positions, inv_freq, attention_factor, eps)
    from oh_my_vllm.kernels.dspark_attention import _normalize_rope_cuda

    return _normalize_rope_cuda(x, weight, positions, inv_freq, attention_factor, eps)


_kernel_dspark_norm_rope.register_fake(_fake_norm)
_DSPARK_NORM_ROPE = Operation(
    "dspark_norm_rope",
    _ir_dspark_norm_rope,
    _native_dspark_norm_rope,
    default_priority=(NAME,),
)
_DSPARK_NORM_ROPE.register_impl(NAME, _kernel_dspark_norm_rope, supports=_cuda_inputs)


def norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_factor: float,
    eps: float,
) -> torch.Tensor:
    return _DSPARK_NORM_ROPE(x, weight, positions, inv_freq, attention_factor, eps)


@torch.library.custom_op("oh_my_vllm_ir::dspark_append", mutates_args=("cache",))
def _ir_dspark_append(
    cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor
) -> None:
    return _DSPARK_APPEND.select(cache, key, value, slots).op(cache, key, value, slots)


_ir_dspark_append.register_fake(_fake_append)


@torch.library.custom_op("oh_my_vllm_native::dspark_append", mutates_args=("cache",))
def _native_dspark_append(
    cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor
) -> None:
    _fake_append(cache, key, value, slots)
    return _reference_append(cache, key, value, slots)


_native_dspark_append.register_fake(_fake_append)


@torch.library.custom_op("oh_my_vllm_kernel::dspark_append", mutates_args=("cache",))
def _kernel_dspark_append(
    cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor
) -> None:
    if NAME == "tilelang":
        from oh_my_vllm.kernels.dspark_tilelang_reference import append

        return append(cache, key, value, slots)
    from oh_my_vllm.kernels.dspark_attention import _append_cuda

    return _append_cuda(cache, key, value, slots)


_kernel_dspark_append.register_fake(_fake_append)
_DSPARK_APPEND = Operation(
    "dspark_append", _ir_dspark_append, _native_dspark_append, default_priority=(NAME,)
)
_DSPARK_APPEND.register_impl(NAME, _kernel_dspark_append, supports=_cuda_inputs)


def append_context(
    cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor
) -> None:
    return _DSPARK_APPEND(cache, key, value, slots)


@torch.library.custom_op("oh_my_vllm_ir::dspark_attention", mutates_args=())
def _ir_dspark_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
) -> torch.Tensor:
    return _DSPARK_ATTENTION.select(
        query, cache, tables, context_lengths, block_key, block_value
    ).op(query, cache, tables, context_lengths, block_key, block_value)


_ir_dspark_attention.register_fake(_fake_attention)


@torch.library.custom_op("oh_my_vllm_native::dspark_attention", mutates_args=())
def _native_dspark_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
) -> torch.Tensor:
    _fake_attention(query, cache, tables, context_lengths, block_key, block_value)
    return _reference_attention(
        query, cache, tables, context_lengths, block_key, block_value
    )


_native_dspark_attention.register_fake(_fake_attention)


@torch.library.custom_op("oh_my_vllm_kernel::dspark_attention", mutates_args=())
def _kernel_dspark_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
) -> torch.Tensor:
    if NAME == "tilelang":
        from oh_my_vllm.kernels.dspark_tilelang_reference import attention

        return attention(query, cache, tables, context_lengths, block_key, block_value)
    from oh_my_vllm.kernels.dspark_attention import _attention_cuda

    return _attention_cuda(
        query, cache, tables, context_lengths, block_key, block_value
    )


_kernel_dspark_attention.register_fake(_fake_attention)
_DSPARK_ATTENTION = Operation(
    "dspark_attention",
    _ir_dspark_attention,
    _native_dspark_attention,
    default_priority=(NAME,),
)
_DSPARK_ATTENTION.register_impl(NAME, _kernel_dspark_attention, supports=_cuda_inputs)


def block_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
) -> torch.Tensor:
    return _DSPARK_ATTENTION(
        query, cache, tables, context_lengths, block_key, block_value
    )
