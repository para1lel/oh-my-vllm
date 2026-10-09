"""Residual RMS and FP8 projection with explicit BF16 rounding boundaries."""

import torch

from oh_my_vllm.kernels import fp8 as kernel_fp8
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec
from .fp8 import _native_linear
from .pointwise import _native_add_rms_norm


def _fake(x, residual, gamma, weight, scale):
    torch._check(x.ndim == 2 and x.shape[1] == 5120)
    torch._check(residual.shape == x.shape and gamma.shape == (5120,))
    torch._check(weight.ndim == 2 and weight.shape[1] == 5120)
    torch._check(weight.shape[0] % 128 == 0)
    torch._check(scale.shape == (weight.shape[0] // 128, 40))
    return (
        torch.empty_like(x),
        torch.empty(
            (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
        ),
    )


@torch.library.custom_op("oh_my_vllm_ir::add_norm_fp8_linear", mutates_args=())
def _ir(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _OP.select(x, residual, gamma, weight, scale).op(
        x, residual, gamma, weight, scale
    )


_ir.register_fake(_fake)


@torch.library.custom_op("oh_my_vllm_native::add_norm_fp8_linear", mutates_args=())
def _native(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed, normalized = _native_add_rms_norm(x, residual, gamma)
    return summed, _native_linear(normalized, weight, scale, False)


_native.register_fake(_fake)


@torch.library.custom_op("oh_my_vllm_kernel::add_norm_fp8_linear", mutates_args=())
def _kernel(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return kernel_fp8.add_norm_linear(x, residual, gamma, weight, scale)


_kernel.register_fake(_fake)

_OP = Operation("add_norm_fp8_linear", _ir, _native, default_priority=(NAME,))
_OP.register_impl(
    NAME,
    _kernel,
    supports=lambda *specs: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda" for spec in specs
    ),
)


def add_norm_fp8_linear(x, residual, gamma, weight, scale):
    """Return the rounded residual sum, then its normalized FP8 projection.

    x and residual are BF16 [rows,5120] features. gamma is the effective FP32
    RMS multiplier. weight is checkpoint FP8 [columns,5120], and scale holds
    FP32 [columns/128,40] multipliers for its 128x128 blocks.
    """
    return _OP(x, residual, gamma, weight, scale)


def _gated_fake(x, gamma, gate, weight, scale):
    torch._check(x.ndim == 3 and x.shape[1:] == (48, 128))
    torch._check(gate.shape == x.shape and gamma.shape == (128,))
    torch._check(weight.ndim == 2 and weight.shape[1] == 6144)
    torch._check(weight.shape[0] % 128 == 0)
    torch._check(scale.shape == (weight.shape[0] // 128, 48))
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )


@torch.library.custom_op("oh_my_vllm_ir::gated_norm_fp8_linear", mutates_args=())
def _gated_ir(
    x: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _GATED_OP.select(x, gamma, gate, weight, scale).op(
        x, gamma, gate, weight, scale
    )


_gated_ir.register_fake(_gated_fake)


@torch.library.custom_op("oh_my_vllm_native::gated_norm_fp8_linear", mutates_args=())
def _gated_native(
    x: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    from .pointwise import _native_rms_norm

    normalized = _native_rms_norm(x, gamma, 1e-6, gate)
    return _native_linear(normalized.flatten(1), weight, scale, False)


_gated_native.register_fake(_gated_fake)


@torch.library.custom_op("oh_my_vllm_kernel::gated_norm_fp8_linear", mutates_args=())
def _gated_kernel(
    x: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return kernel_fp8.gated_norm_linear(x, gamma, gate, weight, scale)


_gated_kernel.register_fake(_gated_fake)
_GATED_OP = Operation(
    "gated_norm_fp8_linear", _gated_ir, _gated_native, default_priority=(NAME,)
)
_GATED_OP.register_impl(
    NAME,
    _gated_kernel,
    supports=lambda *specs: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda" for spec in specs
    ),
)


def gated_norm_fp8_linear(x, gamma, gate, weight, scale):
    """Project RMS-normalized GDN heads after their SiLU gate and BF16 rounding.

    x holds BF16 GDN output [rows,48,128]. gamma is the effective FP32 [128]
    RMS multiplier. gate is BF16 [rows,48,128], including packed token views.
    weight is checkpoint FP8 [columns,6144]; scale is FP32 [columns/128,48].
    Each 128x128 weight block keeps its checkpoint scale. The returned BF16
    [rows,columns] tensor has independent storage and contiguous rows.
    """
    return _GATED_OP(x, gamma, gate, weight, scale)
