"""Attention sigmoid and product retain their separate BF16 rounding boundaries."""

import torch

from oh_my_vllm.kernels import elementwise
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec


def _fake(input, gate):
    torch._check(input.ndim == 3 and input.shape == gate.shape)
    torch._check(input.dtype == torch.bfloat16 and gate.dtype == input.dtype)
    return torch.empty(input.shape, device=input.device, dtype=input.dtype)


@torch.library.custom_op("oh_my_vllm_ir::attention_gate", mutates_args=())
def _ir(input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _OP.select(input, gate).op(input, gate)


_ir.register_fake(_fake)


@torch.library.custom_op("oh_my_vllm_native::attention_gate", mutates_args=())
def _native(input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (input * gate.sigmoid()).contiguous()


_native.register_fake(_fake)


@torch.library.custom_op("oh_my_vllm_kernel::attention_gate", mutates_args=())
def _kernel(input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return elementwise.attention_gate(input, gate)


_kernel.register_fake(_fake)
_OP = Operation("attention_gate", _ir, _native, default_priority=(NAME,))
_OP.register_impl(
    NAME,
    _kernel,
    supports=lambda *specs: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda" for spec in specs
    ),
)


def attention_gate(input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _OP(input, gate)
