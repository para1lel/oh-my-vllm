"""Separate live-query computation from the context rows required by future tokens."""

import torch

from oh_my_vllm.kernels.backend import NAME
from oh_my_vllm.kernels.partial_attention import (
    prepare_context as kernel_context,
)
from oh_my_vllm.kernels.partial_attention import (
    prepare_query as kernel_query,
)

from .core import Operation, TensorSpec
from .state import _reference_norm_rope


def _fake_context(packed, weight, positions, cache, slots):
    torch._check(packed.ndim == 2 and packed.shape[1] == 2048)
    torch._check(weight.shape == (256,) and weight.dtype == torch.float32)
    torch._check(cache.ndim == 5 and cache.shape[1:] == (2, 784, 4, 256))
    torch._check(positions.shape == slots.shape == (packed.shape[0],))
    torch._check(packed.dtype == cache.dtype == torch.bfloat16)
    torch._check(positions.dtype in (torch.int32, torch.int64))
    torch._check(slots.dtype in (torch.int32, torch.int64))


def _fake_query(packed, weight, positions):
    torch._check(packed.ndim == 2 and packed.shape[1] == 12288)
    torch._check(weight.shape == (256,) and weight.dtype == torch.float32)
    torch._check(positions.shape == (packed.shape[0],))
    torch._check(packed.dtype == torch.bfloat16)
    torch._check(positions.dtype in (torch.int32, torch.int64))
    return torch.empty(
        (packed.shape[0], 24, 256), device=packed.device, dtype=packed.dtype
    )


@torch.library.custom_op("oh_my_vllm_ir::prepare_context", mutates_args=("cache",))
def _ir_context(
    packed: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    _CONTEXT.select(packed, weight, positions, cache, slots).op(
        packed, weight, positions, cache, slots
    )


_ir_context.register_fake(_fake_context)


@torch.library.custom_op("oh_my_vllm_native::prepare_context", mutates_args=("cache",))
def _native_context(
    packed: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    _fake_context(packed, weight, positions, cache, slots)
    keys = _reference_norm_rope(packed[:, :1024].reshape(-1, 4, 256), weight, positions)
    values = packed[:, 1024:].reshape(-1, 4, 256)
    valid = slots >= 0
    selected = slots[valid].long()
    cache[selected // 784, 0, selected % 784] = keys[valid]
    cache[selected // 784, 1, selected % 784] = values[valid]


_native_context.register_fake(_fake_context)


@torch.library.custom_op("oh_my_vllm_kernel::prepare_context", mutates_args=("cache",))
def _kernel_context(
    packed: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    _fake_context(packed, weight, positions, cache, slots)
    kernel_context(packed, weight, positions, cache, slots)


_kernel_context.register_fake(_fake_context)


@torch.library.custom_op("oh_my_vllm_ir::prepare_query", mutates_args=())
def _ir_query(
    packed: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    return _QUERY.select(packed, weight, positions).op(packed, weight, positions)


_ir_query.register_fake(_fake_query)


@torch.library.custom_op("oh_my_vllm_native::prepare_query", mutates_args=())
def _native_query(
    packed: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    _fake_query(packed, weight, positions)
    return _reference_norm_rope(
        packed.reshape(-1, 24, 512)[..., :256], weight, positions
    )


_native_query.register_fake(_fake_query)


@torch.library.custom_op("oh_my_vllm_kernel::prepare_query", mutates_args=())
def _kernel_query(
    packed: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    _fake_query(packed, weight, positions)
    return kernel_query(packed, weight, positions)


_kernel_query.register_fake(_fake_query)


def _cuda(*specs):
    return all(isinstance(s, TensorSpec) and s.device.type == "cuda" for s in specs)


_CONTEXT = Operation(
    "prepare_context", _ir_context, _native_context, default_priority=(NAME,)
)
_CONTEXT.register_impl(NAME, _kernel_context, supports=_cuda)
_QUERY = Operation("prepare_query", _ir_query, _native_query, default_priority=(NAME,))
_QUERY.register_impl(NAME, _kernel_query, supports=_cuda)


def prepare_context(packed, weight, positions, cache, slots):
    """Write all retained K/V rows, without a query or attention output."""
    return _CONTEXT(packed, weight, positions, cache, slots)


def prepare_query(packed, weight, positions):
    """Normalize and rotate the 24 live queries in packed Q/gate rows."""
    return _QUERY(packed, weight, positions)
