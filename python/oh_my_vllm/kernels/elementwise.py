"""Fused pointwise model operations with explicit BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul(X, Out, Width: tl.constexpr, Total: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // Width, i % Width
    gate = tl.load(X + row * 2 * Width + col, i < Total, 0).to(tl.float32)
    up = tl.load(X + row * 2 * Width + Width + col, i < Total, 0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + i, activated * up, i < Total)


def silu_mul(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.shape[1] % 2 or packed.dtype != torch.bfloat16:
        raise ValueError("packed gate/up must be a BF16 matrix of even width")
    if not packed.is_cuda or not packed.is_contiguous():
        raise ValueError("packed gate/up must be contiguous CUDA data")
    rows, width = packed.shape[0], packed.shape[1] // 2
    out = torch.empty((rows, width), dtype=packed.dtype, device=packed.device)
    block = 1024 if rows >= 128 else 256
    _silu_mul[(triton.cdiv(out.numel(), block),)](
        packed, out, width, out.numel(), block
    )
    return out


@triton.jit
def _gates(BA, ALog, Bias, Decay, Beta, Total: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, head = i // 48, i % 48
    b = tl.load(BA + row * 96 + head, i < Total, 0).to(tl.float32)
    a = tl.load(BA + row * 96 + 48 + head, i < Total, 0).to(tl.float32)
    log = tl.load(ALog + head)
    bias = tl.load(Bias + head)
    x = a + bias
    softplus = tl.where(x > 20, x, libdevice.log1p(tl.exp(x)))
    tl.store(Decay + i, -tl.exp(log) * softplus, i < Total)
    tl.store(Beta + i, tl.sigmoid(b), i < Total)


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
    _gates[(triton.cdiv(decay.numel(), 128),)](
        ba, a_log, bias, decay, beta, decay.numel(), 128
    )
    return decay, beta
