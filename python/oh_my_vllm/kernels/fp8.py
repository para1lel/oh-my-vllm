"""Block-scaled FP8 linear operations using independent FlashInfer kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, Scales, Rows: tl.constexpr, Width: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * 128 + tl.arange(0, 128)
    values = tl.load(X + row * Width + columns).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(values), 0), 1e-10), 448.0)
    quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale), -448.0), 448.0)
    tl.store(Q + row * Width + columns, quantized)
    # CUTLASS K-major uses contiguous K groups, despite FlashInfer 0.6.18
    # describing this argument as column-major. Multi-row FP64 tests cover it.
    tl.store(Scales + row * (Width // 128) + group, scale)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("FP8 input must be BF16, FP16 or FP32")
    if (
        x.ndim != 2
        or x.numel() == 0
        or x.shape[1] % 128
        or not x.is_contiguous()
        or not x.is_cuda
    ):
        raise ValueError(
            "FP8 requires a contiguous CUDA matrix with K divisible by 128"
        )
    rows, width = x.shape
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // 128), dtype=torch.float32, device=x.device)
    _quantize[(rows, width // 128)](x, data, scales, rows, width)
    return data, scales


def linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Weight is checkpoint [N,K] FP8; scale is checkpoint [N/128,K/128]."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    if x.ndim != 2 or weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("FP8 inputs, weights and scales must be matrices")
    if weight.dtype != torch.float8_e4m3fn or weight.shape[1] != x.shape[1]:
        raise ValueError("FP8 checkpoint weight dtype or input width mismatch")
    if weight.shape[0] % 128 or weight_scale.shape != (
        weight.shape[0] // 128,
        weight.shape[1] // 128,
    ):
        raise ValueError("FP8 checkpoint requires 128x128 weight scaling")
    if weight_scale.dtype != torch.float32 or any(
        not t.is_cuda or not t.is_contiguous() or t.device != x.device
        for t in (weight, weight_scale)
    ):
        raise ValueError(
            "FP8 weights/scales must be contiguous on the input GPU, scales FP32"
        )
    data, scale = quantize(x)
    return gemm_fp8_nt_groupwise(
        data,
        weight,
        scale,
        weight_scale,
        scale_major_mode="K",
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
