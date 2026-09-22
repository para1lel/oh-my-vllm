"""RMS normalization and text-only partial NeoX rotary embedding."""

import math

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _rms(h: int, d: int, eps: float, gated: bool, xs: tuple, gs: tuple):
    n = T.dynamic("n")
    block = 1 << (d - 1).bit_length()

    @T.prim_func
    def kernel(
        x: T.StridedTensor((n, h, d), xs, "bfloat16"),
        weight: T.Tensor((d,), "float32"),
        gate: T.StridedTensor((n, h, d), gs, "bfloat16"),
        out: T.Tensor((n, h, d), "bfloat16"),
    ):
        with T.Kernel(n * h, threads=32 if d <= 256 else 128) as row:
            values = T.alloc_fragment((block,), "float32")
            squares = T.alloc_fragment((block,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(block):
                values[i] = T.if_then_else(
                    i < d, x[row // h, row % h, i].astype("float32"), 0
                )
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(block):
                if i < d:
                    value = values[i] * T.rsqrt(total[0] / d + eps) * weight[i]
                    if gated:
                        g = gate[row // h, row % h, i].astype("float32")
                        out[row // h, row % h, i] = value * g / (1 + T.exp(-g))
                    else:
                        out[row // h, row % h, i] = value

    return kernel


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
    gs = gate.stride() if gate is not None else x.stride()
    _rms(*x.shape[1:], epsilon, gate is not None, x.stride(), gs)(
        x, weight, gate if gate is not None else x, out.view(x.shape)
    )
    return out


@tilelang.jit
def _rope(h: int, d: int, rotary: int, theta: float, index_dtype: str):
    n = T.dynamic("n")
    block = 1 << (d - 1).bit_length()

    @T.prim_func
    def kernel(
        x: T.Tensor((n, h, d), "bfloat16"),
        positions: T.Tensor((n,), index_dtype),
        out: T.Tensor((n, h, d), "bfloat16"),
    ):
        with T.Kernel(n * h, threads=128) as row:
            for i in T.Parallel(block):
                if i < d:
                    value = x[row // h, row % h, i].astype("float32")
                    if i < rotary:
                        pos = positions[row // h].astype("float32")
                        freq = T.exp(-T.log(theta) * (i % (rotary // 2)) * 2.0 / rotary)
                        other_col = T.if_then_else(
                            i < rotary // 2, i + rotary // 2, i - rotary // 2
                        )
                        other = x[row // h, row % h, other_col].astype("float32")
                        rotated_other = T.if_then_else(i < rotary // 2, -other, other)
                        out[row // h, row % h, i] = value * T.cos(
                            pos * freq
                        ) + rotated_other * T.sin(pos * freq)
                    else:
                        out[row // h, row % h, i] = value

    return kernel


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
    _rope(*x.shape[1:], rotary_dim, theta, str(positions.dtype).removeprefix("torch."))(
        x, positions, out
    )
    return out


@tilelang.jit
def _rms_rotary(h: int, strides: tuple, index_dtype: str):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        x: T.StridedTensor((n, h, 256), strides, "bfloat16"),
        weight: T.Tensor((256,), "float32"),
        positions: T.Tensor((n,), index_dtype),
        out: T.Tensor((n, h, 256), "bfloat16"),
    ):
        with T.Kernel(n * h, threads=256) as row:
            values = T.alloc_fragment((256,), "float32")
            squares = T.alloc_fragment((256,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(256):
                values[i] = x[row // h, row % h, i].astype("float32")
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(256):
                values[i] = (
                    (values[i] * T.rsqrt(total[0] / 256 + 1e-6) * weight[i])
                    .astype("bfloat16")
                    .astype("float32")
                )
            for i in T.Parallel(256):
                if i < 64:
                    # Preserve normalization's BF16 boundary before rotation.
                    other_col = T.if_then_else(i < 32, i + 32, i - 32)
                    other = (
                        (
                            x[row // h, row % h, other_col].astype("float32")
                            * T.rsqrt(total[0] / 256 + 1e-6)
                            * weight[other_col]
                        )
                        .astype("bfloat16")
                        .astype("float32")
                    )
                    # Reduce the phase in FP64: FP32 frequency/angle error grows
                    # enough near the maximum context to exceed BF16 tolerances.
                    angle64 = positions[row // h].astype("float64") * T.exp(
                        T.float64(-math.log(10000000.0)) * (i % 32) / 32
                    )
                    angle = (
                        angle64
                        - T.round(angle64 / T.float64(math.tau)) * T.float64(math.tau)
                    ).astype("float32")
                    out[row // h, row % h, i] = values[i] * T.cos(
                        angle
                    ) + T.if_then_else(i < 32, -other, other) * T.sin(angle)
                else:
                    out[row // h, row % h, i] = values[i]

    return kernel


def rms_rotary(
    x: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Fuse Qwen's 256-wide Q/K RMS and 64-wide partial NeoX rotation."""
    if x.ndim != 3 or x.shape[-1] != 256 or x.numel() == 0:
        raise ValueError("fused Q/K normalization requires [tokens,heads,256]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("fused Q/K normalization requires BF16 data/FP32 weight")
    if weight.shape != (256,) or not weight.is_contiguous():
        raise ValueError("fused Q/K normalization requires a contiguous256 weight")
    if positions.shape != (x.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "fused Q/K normalization requires one integer position per row"
        )
    if not positions.is_contiguous() or any(
        not t.is_cuda or t.device != x.device for t in (x, weight, positions)
    ):
        raise ValueError("fused Q/K tensors must share a GPU with contiguous positions")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rms_rotary(x.shape[1], x.stride(), str(positions.dtype).removeprefix("torch."))(
        x, weight, positions, out
    )
    return out


@tilelang.jit
def _add_rms(d: int):
    n = T.dynamic("n")
    # Avoid padding the model's residual width to 8192 reduction elements.
    block = d if d == 5120 else 1 << (d - 1).bit_length()

    @T.prim_func
    def kernel(
        x: T.Tensor((n, d), "bfloat16"),
        residual: T.Tensor((n, d), "bfloat16"),
        weight: T.Tensor((d,), "float32"),
        summed: T.Tensor((n, d), "bfloat16"),
        out: T.Tensor((n, d), "bfloat16"),
    ):
        with T.Kernel(n, threads=256) as row:
            values = T.alloc_fragment((block,), "float32")
            squares = T.alloc_fragment((block,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(block):
                values[i] = 0
                if i < d:
                    value = x[row, i].astype("float32") + residual[row, i].astype(
                        "float32"
                    )
                    values[i] = value.astype("bfloat16").astype("float32")
                    summed[row, i] = values[i]
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(block):
                if i < d:
                    out[row, i] = values[i] * T.rsqrt(total[0] / d + 1e-6) * weight[i]

    return kernel


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
    _add_rms(x.shape[1])(x, residual, weight, summed, normalized)
    return summed, normalized
