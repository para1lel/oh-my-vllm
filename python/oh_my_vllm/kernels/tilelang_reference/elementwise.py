"""Fused pointwise model operations with explicit BF16 rounding boundaries."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _silu_mul(width: int, block: int):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        x: T.Tensor((rows, 2 * width), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * width, block), threads=128) as bx:
            for j in T.Parallel(block):
                i = bx * block + j
                if i < rows * width:
                    row, col = i // width, i % width
                    gate = x[row, col].astype("float32")
                    up = x[row, width + col].astype("float32")
                    activated = (gate / (1 + T.exp(-gate))).astype("bfloat16")
                    out[row, col] = activated.astype("float32") * up

    return kernel


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


@tilelang.jit
def _gates():
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        ba: T.Tensor((rows, 96), "bfloat16"),
        a_log: T.Tensor((48,), "float32"),
        bias: T.Tensor((48,), "float32"),
        decay: T.Tensor((rows, 48), "float32"),
        beta: T.Tensor((rows, 48), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * 48, 128), threads=128) as bx:
            for j in T.Parallel(128):
                i = bx * 128 + j
                if i < rows * 48:
                    row, head = i // 48, i % 48
                    b = ba[row, head].astype("float32")
                    a = ba[row, head + 48].astype("float32") + bias[head]
                    softplus = T.if_then_else(
                        a > 20, a, T.call_extern("float32", "log1pf", T.exp(a))
                    )
                    decay[row, head] = -T.exp(a_log[head]) * softplus
                    beta[row, head] = 1 / (1 + T.exp(-b))

    return kernel


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
