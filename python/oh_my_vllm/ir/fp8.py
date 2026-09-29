"""Block FP8 quantization and fused FlashInfer GEMM semantic operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from oh_my_vllm.kernels import fp8 as kernel_fp8
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec


def _reference_quantize(
    x: torch.Tensor,
    column_major: bool,
    silu_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if silu_gate:
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate.float()).to(x.dtype) * up
    values = x.float().reshape(x.shape[0], -1, 128)
    magnitudes = torch.nan_to_num(
        values.abs(), nan=0.0, posinf=float("inf"), neginf=float("inf")
    )
    maximum = magnitudes.amax(dim=-1)
    scales = torch.fmax(maximum, torch.full_like(maximum, 1e-10)) / 448.0
    divided = values / scales[..., None]
    data = torch.fmin(
        torch.full_like(divided, 448),
        torch.fmax(torch.full_like(divided, -448), divided),
    )
    data = data.reshape(x.shape).to(torch.float8_e4m3fn)
    if column_major:
        scales = scales.T.contiguous().T
    return data, scales


def _fake_quantize(
    x: torch.Tensor,
    column_major: bool,
    silu_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(x.ndim == 2 and x.shape[1] % 128 == 0)
    width = x.shape[1] // (2 if silu_gate else 1)
    torch._check(width % 128 == 0)
    data = torch.empty((x.shape[0], width), device=x.device, dtype=torch.float8_e4m3fn)
    if column_major:
        scales = torch.empty(
            (width // 128, x.shape[0]), device=x.device, dtype=torch.float32
        ).T
    else:
        scales = torch.empty(
            (x.shape[0], width // 128), device=x.device, dtype=torch.float32
        )
    return data, scales


@torch.library.custom_op("oh_my_vllm_ir::fp8_quantize", mutates_args=())
def _ir_quantize(
    x: torch.Tensor,
    column_major: bool,
    silu_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _QUANTIZE.select(x, column_major, silu_gate).op(x, column_major, silu_gate)


_ir_quantize.register_fake(_fake_quantize)


@torch.library.custom_op("oh_my_vllm_native::fp8_quantize", mutates_args=())
def _native_quantize(
    x: torch.Tensor,
    column_major: bool,
    silu_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reference_quantize(x, column_major, silu_gate)


_native_quantize.register_fake(_fake_quantize)


@torch.library.custom_op("oh_my_vllm_kernel::fp8_quantize", mutates_args=())
def _kernel_quantize(
    x: torch.Tensor,
    column_major: bool,
    silu_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return kernel_fp8.quantize(x, column_major=column_major, silu_gate=silu_gate)


_kernel_quantize.register_fake(_fake_quantize)


_QUANTIZE = Operation(
    "fp8_quantize",
    _ir_quantize,
    _native_quantize,
    default_priority=(NAME,),
)
_QUANTIZE.register_impl(
    NAME,
    _kernel_quantize,
    supports=lambda x, column_major, silu_gate: (
        isinstance(x, TensorSpec) and x.device.type == "cuda"
    ),
)


def _fake_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    silu_gate: bool,
) -> torch.Tensor:
    torch._check(x.ndim == 2 and weight.ndim == 2 and weight_scale.ndim == 2)
    width = x.shape[1] // (2 if silu_gate else 1)
    torch._check(weight.shape[1] == width)
    torch._check(weight_scale.shape == (weight.shape[0] // 128, width // 128))
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )


@torch.library.custom_op("oh_my_vllm_ir::fp8_linear", mutates_args=())
def _ir_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    silu_gate: bool,
) -> torch.Tensor:
    return _LINEAR.select(x, weight, weight_scale, silu_gate).op(
        x, weight, weight_scale, silu_gate
    )


_ir_linear.register_fake(_fake_linear)


@torch.library.custom_op("oh_my_vllm_native::fp8_linear", mutates_args=())
def _native_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    silu_gate: bool,
) -> torch.Tensor:
    data, scales = _reference_quantize(x, x.shape[0] <= 32, silu_gate)
    activations = data.float() * scales.repeat_interleave(128, dim=1)
    expanded_scales = weight_scale.repeat_interleave(128, dim=0)
    expanded_scales = expanded_scales.repeat_interleave(128, dim=1)
    weights = weight.float() * expanded_scales
    return (activations @ weights.T).to(torch.bfloat16)


_native_linear.register_fake(_fake_linear)


@torch.library.custom_op("oh_my_vllm_kernel::fp8_linear", mutates_args=())
def _kernel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    silu_gate: bool,
) -> torch.Tensor:
    return kernel_fp8.linear(x, weight, weight_scale, silu_gate=silu_gate)


_kernel_linear.register_fake(_fake_linear)


_LINEAR = Operation("fp8_linear", _ir_linear, _native_linear, default_priority=(NAME,))
_LINEAR.register_impl(
    NAME,
    _kernel_linear,
    supports=lambda x, weight, scale, silu_gate: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda"
        for spec in (x, weight, scale)
    ),
)


def quantize(
    x: torch.Tensor,
    *,
    column_major: bool = False,
    silu_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _QUANTIZE(x, column_major, silu_gate)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    silu_gate: bool = False,
) -> torch.Tensor:
    return _LINEAR(x, weight, weight_scale, silu_gate)
