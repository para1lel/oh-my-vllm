"""State-writing IR operations with explicit torch.library mutation schemas."""

from __future__ import annotations

import torch

from oh_my_vllm.kernels.attention_prepare import prepare_attention as kernel_prepare
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec


def _reference_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """FP64 phase/reference arithmetic with the model's BF16 norm boundary."""
    values = x.double()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    normalized = (normalized * weight.double()).to(x.dtype).double()
    frequencies = 10000000.0 ** (
        -torch.arange(32, device=x.device, dtype=torch.float64) / 32
    )
    angle = positions.double()[:, None, None] * frequencies[None, None]
    cosine, sine = angle.cos(), angle.sin()
    left, right = normalized[..., :32], normalized[..., 32:64]
    first = left * cosine - right * sine
    second = right * cosine + left * sine
    return torch.cat((first, second, normalized[..., 64:]), dim=-1).to(x.dtype)


def _fake_prepare(
    packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    torch._check(packed.ndim == 2 and packed.shape[1] == 14336)
    torch._check(cache.ndim == 5 and cache.shape[1:] == (2, 784, 4, 256))
    torch._check(q_weight.shape == (256,) and k_weight.shape == (256,))
    torch._check(positions.shape == (packed.shape[0],))
    torch._check(slots.shape == (packed.shape[0],))
    torch._check(packed.dtype == torch.bfloat16 and cache.dtype == torch.bfloat16)
    return torch.empty(
        (packed.shape[0], 24, 256), device=packed.device, dtype=packed.dtype
    )


@torch.library.custom_op("oh_my_vllm_ir::prepare_attention", mutates_args=("cache",))
def _ir_prepare(
    packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    return _PREPARE.select(packed, q_weight, k_weight, positions, cache, slots).op(
        packed, q_weight, k_weight, positions, cache, slots
    )


_ir_prepare.register_fake(_fake_prepare)


@torch.library.custom_op(
    "oh_my_vllm_native::prepare_attention", mutates_args=("cache",)
)
def _native_prepare(
    packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    _fake_prepare(packed, q_weight, k_weight, positions, cache, slots)
    qg, k, v = packed.split([12288, 1024, 1024], dim=-1)
    q = _reference_norm_rope(qg.reshape(-1, 24, 512)[..., :256], q_weight, positions)
    key = _reference_norm_rope(k.reshape(-1, 4, 256), k_weight, positions)
    value = v.reshape(-1, 4, 256)
    valid = slots >= 0
    selected = slots[valid].long()
    cache[selected // 784, 0, selected % 784] = key[valid]
    cache[selected // 784, 1, selected % 784] = value[valid]
    return q


_native_prepare.register_fake(_fake_prepare)


@torch.library.custom_op(
    "oh_my_vllm_kernel::prepare_attention", mutates_args=("cache",)
)
def _kernel_prepare(
    packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    return kernel_prepare(packed, q_weight, k_weight, positions, cache, slots)


_kernel_prepare.register_fake(_fake_prepare)


_PREPARE = Operation(
    "prepare_attention", _ir_prepare, _native_prepare, default_priority=(NAME,)
)
_PREPARE.register_impl(
    NAME,
    _kernel_prepare,
    supports=lambda packed, q_weight, k_weight, positions, cache, slots: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda"
        for spec in (packed, q_weight, k_weight, positions, cache, slots)
    ),
)


def prepare_attention(
    packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Write K/V at nonnegative physical slots and return normalized/rotated Q."""
    return _PREPARE(packed, q_weight, k_weight, positions, cache, slots)
