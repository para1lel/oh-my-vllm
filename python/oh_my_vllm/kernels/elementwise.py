"""Fused pointwise model operations with explicit BF16 rounding boundaries."""

import torch

from .backend import kernel

_silu_mul = kernel("elementwise", "_silu_mul")


def silu_mul(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.shape[1] % 2 or packed.dtype != torch.bfloat16:
        raise ValueError("packed gate/up must be a BF16 matrix of even width")
    if not packed.is_cuda or not packed.is_contiguous():
        raise ValueError("packed gate/up must be contiguous CUDA data")
    rows, width = packed.shape[0], packed.shape[1] // 2
    out = torch.empty((rows, width), dtype=packed.dtype, device=packed.device)
    block = 1024 if rows >= 128 else 256
    _silu_mul(width, block)(packed, out)
    return out


_gates = kernel("elementwise", "_gates")


def delta_gates(ba: torch.Tensor, a_log: torch.Tensor, bias: torch.Tensor):
    if ba.ndim != 2 or ba.shape[1] != 96 or ba.dtype != torch.bfloat16:
        raise ValueError("GDN b/a projection must be BF16 [tokens,96]")
    if any(t.shape != (48,) or t.dtype != torch.float32 for t in (a_log, bias)):
        raise ValueError("GDN gate parameters must be FP32 [48]")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != ba.device
        for t in (ba, a_log, bias)
    ):
        raise ValueError("GDN gates must be contiguous on one CUDA device")
    decay = torch.empty((len(ba), 48), device=ba.device)
    beta = torch.empty_like(decay)
    _gates()(ba, a_log, bias, decay, beta)
    return decay, beta
