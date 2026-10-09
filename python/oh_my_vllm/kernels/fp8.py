"""Block-scaled FP8 linear operations using independent FlashInfer kernels."""

import torch

from .backend import NAME, kernel

_quantize = kernel("fp8", "_quantize")


def quantize(
    x: torch.Tensor, *, column_major: bool = False, silu_gate: bool = False
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
    # Packed fused SiLU input already contains both gate and up halves.
    # Compare by division so even a foreign shape scalar cannot overflow here.
    # The largest accessed signed-int32 offset is total_elements - 1.
    if rows > 2**31 // width:
        raise ValueError(
            f"tensor size {x.shape[0]} * {x.shape[1]} exceeds int32 flat offset range"
        )
    if silu_gate:
        if x.dtype != torch.bfloat16 or width % 256:
            raise ValueError("fused SiLU quantization requires packed BF16 gate/up")
        width //= 2
    if width > 2**31 - 1:
        raise ValueError("quantization width exceeds signed int32 kernel dimension")
    data = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    shape = (width // 128, rows) if column_major else (rows, width // 128)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    if column_major:
        scales = scales.T
    if NAME == "cuda":
        _quantize(column_major, silu_gate)(x, data, scales)
    else:
        tile = 16 if rows >= 128 and (silu_gate or not column_major) else 1
        _quantize(
            width, str(x.dtype).removeprefix("torch."), column_major, silu_gate, tile
        )(x, data, scales)
    return data, scales


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    silu_gate: bool = False,
) -> torch.Tensor:
    """Weight is checkpoint [N,K] FP8; scale is checkpoint [N/128,K/128]."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    if x.ndim != 2 or weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("FP8 inputs, weights and scales must be matrices")
    input_width = x.shape[1] // 2 if silu_gate else x.shape[1]
    if weight.dtype != torch.float8_e4m3fn or weight.shape[1] != input_width:
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
    data, scale = quantize(x, column_major=small, silu_gate=silu_gate)
    if NAME == "cuda" and not small:
        from .cuda_backend.groupwise import gemm

        return gemm(
            data,
            weight,
            scale,
            weight_scale,
            mma_sm=2 if x.shape[0] >= 4096 and weight.shape[0] >= 32768 else 1,
        )
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
