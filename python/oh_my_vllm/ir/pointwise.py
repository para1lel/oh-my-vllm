"""Executable semantics and providers for Qwen pointwise CUDA operators.

Each IR node has a native PyTorch reference, a production CUDA provider, and
one torch.library schema shared by both. The reference requires explicit
provider selection; the default always executes the existing CUDA kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from oh_my_vllm.kernels import elementwise, normalization
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec

_BACKEND_PROVIDER = NAME


def _cuda(*specs: TensorSpec | None) -> bool:
    return all(spec is None or spec.device.type == "cuda" for spec in specs)


def _rms_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    values = x.float()
    factor = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
    result = values * factor * weight.float()
    if gate is not None:
        result = result * F.silu(gate.float())
    # Match the production provider and fake kernel's contiguous output layout.
    return result.to(x.dtype).contiguous()


@torch.library.custom_op("oh_my_vllm_ir::silu_mul", mutates_args=())
def _ir_silu_mul(packed: torch.Tensor) -> torch.Tensor:
    return _SILU.select(packed).op(packed)


def _fake_silu_mul(packed: torch.Tensor) -> torch.Tensor:
    torch._check(packed.ndim == 2 and packed.shape[1] % 2 == 0)
    return torch.empty(
        (packed.shape[0], packed.shape[1] // 2),
        device=packed.device,
        dtype=packed.dtype,
    )


_ir_silu_mul.register_fake(_fake_silu_mul)


@torch.library.custom_op("oh_my_vllm_native::silu_mul", mutates_args=())
def _native_silu_mul(packed: torch.Tensor) -> torch.Tensor:
    gate, up = packed.chunk(2, dim=-1)
    return F.silu(gate.float()).to(packed.dtype) * up


_native_silu_mul.register_fake(_fake_silu_mul)


@torch.library.custom_op("oh_my_vllm_kernel::silu_mul", mutates_args=())
def _kernel_silu_mul(packed: torch.Tensor) -> torch.Tensor:
    return elementwise.silu_mul(packed)


_kernel_silu_mul.register_fake(_fake_silu_mul)


_SILU = Operation(
    "silu_mul",
    _ir_silu_mul,
    _native_silu_mul,
    default_priority=(_BACKEND_PROVIDER,),
)
_SILU.register_impl(_BACKEND_PROVIDER, _kernel_silu_mul, supports=_cuda)


@torch.library.custom_op("oh_my_vllm_ir::delta_gates", mutates_args=())
def _ir_delta_gates(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _GATES.select(ba, a_log, bias).op(ba, a_log, bias)


def _fake_delta_gates(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(ba.ndim == 2 and ba.shape[1] == 96)
    torch._check(a_log.shape == (48,) and bias.shape == (48,))
    shape = (ba.shape[0], 48)
    return (
        torch.empty(shape, device=ba.device, dtype=torch.float32),
        torch.empty(shape, device=ba.device, dtype=torch.float32),
    )


_ir_delta_gates.register_fake(_fake_delta_gates)


@torch.library.custom_op("oh_my_vllm_native::delta_gates", mutates_args=())
def _native_delta_gates(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, a = ba.float().chunk(2, dim=-1)
    return -a_log.exp() * F.softplus(a + bias), b.sigmoid()


_native_delta_gates.register_fake(_fake_delta_gates)


@torch.library.custom_op("oh_my_vllm_kernel::delta_gates", mutates_args=())
def _kernel_delta_gates(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return elementwise.delta_gates(ba, a_log, bias)


_kernel_delta_gates.register_fake(_fake_delta_gates)


_GATES = Operation(
    "delta_gates",
    _ir_delta_gates,
    _native_delta_gates,
    default_priority=(_BACKEND_PROVIDER,),
)
_GATES.register_impl(_BACKEND_PROVIDER, _kernel_delta_gates, supports=_cuda)


@torch.library.custom_op("oh_my_vllm_ir::rms_norm", mutates_args=())
def _ir_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    return _RMS.select(x, weight, epsilon, gate).op(x, weight, epsilon, gate)


def _fake_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    torch._check(weight.shape == (x.shape[-1],))
    if gate is not None:
        torch._check(gate.shape == x.shape)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


_ir_rms_norm.register_fake(_fake_rms_norm)


@torch.library.custom_op("oh_my_vllm_native::rms_norm", mutates_args=())
def _native_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    return _rms_reference(x, weight, epsilon, gate)


_native_rms_norm.register_fake(_fake_rms_norm)


@torch.library.custom_op("oh_my_vllm_kernel::rms_norm", mutates_args=())
def _kernel_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    return normalization.rms_norm(x, weight, epsilon, gate)


_kernel_rms_norm.register_fake(_fake_rms_norm)


_RMS = Operation(
    "rms_norm",
    _ir_rms_norm,
    _native_rms_norm,
    default_priority=(_BACKEND_PROVIDER,),
)
_RMS.register_impl(
    _BACKEND_PROVIDER,
    _kernel_rms_norm,
    supports=lambda x, weight, epsilon, gate: _cuda(x, weight, gate),
)


@torch.library.custom_op("oh_my_vllm_ir::add_rms_norm", mutates_args=())
def _ir_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ADD_RMS.select(x, residual, weight).op(x, residual, weight)


def _fake_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(x.shape == residual.shape)
    torch._check(weight.shape == (x.shape[-1],))
    return (
        torch.empty(x.shape, device=x.device, dtype=x.dtype),
        torch.empty(x.shape, device=x.device, dtype=x.dtype),
    )


_ir_add_rms_norm.register_fake(_fake_add_rms_norm)


@torch.library.custom_op("oh_my_vllm_native::add_rms_norm", mutates_args=())
def _native_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = (x.float() + residual.float()).to(x.dtype)
    return summed, _rms_reference(summed, weight, 1e-6, None)


_native_add_rms_norm.register_fake(_fake_add_rms_norm)


@torch.library.custom_op("oh_my_vllm_kernel::add_rms_norm", mutates_args=())
def _kernel_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return normalization.add_rms_norm(x, residual, weight)


_kernel_add_rms_norm.register_fake(_fake_add_rms_norm)


_ADD_RMS = Operation(
    "add_rms_norm",
    _ir_add_rms_norm,
    _native_add_rms_norm,
    default_priority=(_BACKEND_PROVIDER,),
)
_ADD_RMS.register_impl(_BACKEND_PROVIDER, _kernel_add_rms_norm, supports=_cuda)


def silu_mul(packed: torch.Tensor) -> torch.Tensor:
    return _SILU(packed)


def delta_gates(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _GATES(ba, a_log, bias)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    return _RMS(x, weight, epsilon, gate)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ADD_RMS(x, residual, weight)
