"""RMS normalization and text-only partial NeoX rotary embedding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms(
    X,
    Weight,
    Gate,
    Out,
    D: tl.constexpr,
    Eps: tl.constexpr,
    Gated: tl.constexpr,
    Block: tl.constexpr,
    Heads: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    XD: tl.constexpr,
    GS: tl.constexpr,
    GH: tl.constexpr,
    GD: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    offset = row // Heads * XS + row % Heads * XH + col * XD
    x = tl.load(X + offset, col < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + Eps)
    w = tl.load(Weight + col, col < D, 0).to(tl.float32)
    value = x * inv * w
    if Gated:
        offset_gate = row // Heads * GS + row % Heads * GH + col * GD
        gate = tl.load(Gate + offset_gate, col < D, 0).to(tl.float32)
        value *= gate * tl.sigmoid(gate)
    tl.store(Out + row * D + col, value, col < D)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight is the effective multiplier (Qwen offset weights add one at load)."""
    if x.ndim < 1 or x.numel() == 0 or weight.shape != (x.shape[-1],):
        raise ValueError("RMS normalization weight must match the nonempty last axis")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("RMS normalization requires BF16 input and FP32 weight")
    if not 0 < epsilon < 1:
        raise ValueError("RMS normalization epsilon must be in (0,1)")
    tensors = (x, weight) if gate is None else (x, weight, gate)
    if gate is not None and (gate.shape != x.shape or gate.dtype != x.dtype):
        raise ValueError("RMS normalization gate must match input shape/dtype")
    if any(not t.is_cuda or t.device != x.device for t in tensors):
        raise ValueError("RMS normalization tensors must share one CUDA device")
    if not weight.is_contiguous():
        raise ValueError("RMS normalization weight must be contiguous")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    # Preserve packed projection views; flatten only uncommon higher-rank inputs.
    def rows(tensor):
        if tensor.ndim == 3:
            return tensor
        return tensor.reshape(-1, 1, tensor.shape[-1])

    x = rows(x)
    gate = rows(gate) if gate is not None else None
    gs = gate.stride() if gate is not None else (0, 0, 0)
    _rms[(x.numel() // x.shape[-1],)](
        x,
        weight,
        gate if gate is not None else x,
        out,
        x.shape[-1],
        epsilon,
        gate is not None,
        triton.next_power_of_2(x.shape[-1]),
        x.shape[1],
        *x.stride(),
        *gs,
    )
    return out


@triton.jit
def _rope(
    X,
    Positions,
    Out,
    Heads: tl.constexpr,
    D: tl.constexpr,
    Rotary: tl.constexpr,
    Theta: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    pos = tl.load(Positions + row // Heads).to(tl.float32)
    frequency = tl.exp(-tl.log(Theta) * (col % (Rotary // 2)) * 2.0 / Rotary)
    angle = pos * frequency
    x = tl.load(X + row * D + col, col < D, 0)
    other_col = tl.where(col < Rotary // 2, col + Rotary // 2, col - Rotary // 2)
    other = tl.load(X + row * D + other_col, col < Rotary, 0)
    rotated = x.to(tl.float32) * tl.cos(angle) + tl.where(
        col < Rotary // 2, -other, other
    ) * tl.sin(angle)
    tl.store(Out + row * D + col, tl.where(col < Rotary, rotated, x), col < D)


def rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int = 64,
    theta: float = 10000000.0,
) -> torch.Tensor:
    if x.ndim != 3 or x.numel() == 0 or x.dtype != torch.bfloat16:
        raise ValueError(
            "rotary input must be a nonempty BF16 [tokens,heads,dim] tensor"
        )
    if (
        type(rotary_dim) is not int
        or not 0 < rotary_dim <= x.shape[-1]
        or rotary_dim % 2
    ):
        raise ValueError(
            "rotary dimension must be positive, even and no larger than head size"
        )
    if positions.shape != (x.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("rotary requires one integer position per token")
    if not x.is_cuda or positions.device != x.device or not positions.is_contiguous():
        raise ValueError("rotary positions must be contiguous on the input CUDA device")
    if not 1 < theta < float("inf"):
        raise ValueError("rotary theta must be finite and greater than one")
    x = x.contiguous()
    out = torch.empty_like(x)
    _rope[(x.shape[0] * x.shape[1],)](
        x,
        positions,
        out,
        x.shape[1],
        x.shape[2],
        rotary_dim,
        theta,
        triton.next_power_of_2(x.shape[2]),
    )
    return out


@triton.jit
def _add_rms(X, Residual, Weight, Sum, Out, D: tl.constexpr, Block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    residual = tl.load(Residual + row * D + col, col < D, 0).to(tl.float32)
    # Preserve the materialized BF16 residual before computing RMS in FP32.
    value = (x + residual).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 0) / D + 1e-6)
    weight = tl.load(Weight + col, col < D, 0)
    tl.store(Sum + row * D + col, value, col < D)
    tl.store(Out + row * D + col, value * inv * weight, col < D)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the rounded residual sum and its RMS-normalized projection input."""
    if x.ndim != 2 or x.numel() == 0 or residual.shape != x.shape:
        raise ValueError("residual RMS requires two nonempty matrices of equal shape")
    if x.dtype != torch.bfloat16 or residual.dtype != x.dtype:
        raise ValueError("residual RMS inputs must be BF16")
    if weight.shape != (x.shape[1],) or weight.dtype != torch.float32:
        raise ValueError("residual RMS weight must be FP32 and match the row width")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != x.device
        for t in (x, residual, weight)
    ):
        raise ValueError("residual RMS tensors must be contiguous on one CUDA device")
    summed, normalized = torch.empty_like(x), torch.empty_like(x)
    _add_rms[(len(x),)](
        x,
        residual,
        weight,
        summed,
        normalized,
        x.shape[1],
        triton.next_power_of_2(x.shape[1]),
        num_warps=8,
    )
    return summed, normalized
