"""Block-scaled FP8 linear operations using independent FlashInfer kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(
    X, Q, Scales, Rows: tl.constexpr, Width: tl.constexpr, Column: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * 128 + tl.arange(0, 128)
    values = tl.load(X + row * Width + columns).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(values), 0), 1e-10), 448.0)
    quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale), -448.0), 448.0)
    tl.store(Q + row * Width + columns, quantized)
    # CUTLASS K-major uses contiguous K groups, despite FlashInfer 0.6.18
    # describing this argument as column-major. Multi-row FP64 tests cover it.
    offset = group * Rows + row if Column else row * (Width // 128) + group
    tl.store(Scales + offset, scale)


@triton.jit
def _quantize_prefill(
    X, Q, Scales, Rows: tl.constexpr, Width: tl.constexpr, BlockRows: tl.constexpr
):
    rows = tl.program_id(0) * BlockRows + tl.arange(0, BlockRows)
    group = tl.program_id(1)
    columns = group * 128 + tl.arange(0, 128)
    values = tl.load(
        X + rows[:, None] * Width + columns[None, :], rows[:, None] < Rows, 0
    ).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(values), 1), 1e-10), 448.0)
    quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale[:, None]), -448.0), 448.0)
    tl.store(
        Q + rows[:, None] * Width + columns[None, :], quantized, rows[:, None] < Rows
    )
    tl.store(Scales + rows * (Width // 128) + group, scale, rows < Rows)


def quantize(
    x: torch.Tensor, *, column_major: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
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
    shape = (width // 128, rows) if column_major else (rows, width // 128)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    if column_major:
        scales = scales.T
    if rows >= 128 and not column_major:
        # Amortize CTA scheduling across rows for bandwidth-bound long prefills.
        _quantize_prefill[(triton.cdiv(rows, 16), width // 128)](
            x, data, scales, rows, width, 16
        )
    else:
        _quantize[(rows, width // 128)](x, data, scales, rows, width, column_major)
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
    # TRT-LLM's independent FlashInfer backend is faster for decode and avoids
    # CUTLASS SM100's nondeterministic 17..32-row low-latency path. Its activation
    # scales are column-major; checkpoint weight scales remain row-major.
    small = x.shape[0] <= 32
    data, scale = quantize(x, column_major=small)
    return gemm_fp8_nt_groupwise(
        data,
        weight,
        scale,
        weight_scale,
        scale_major_mode="K",
        mma_sm=2 if x.shape[0] >= 256 and weight.shape[0] >= 32768 else 1,
        out_dtype=torch.bfloat16,
        backend="trtllm" if small else "cutlass",
    )
