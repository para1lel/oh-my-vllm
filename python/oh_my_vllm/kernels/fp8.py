"""Block-scaled FP8 linear operations using independent FlashInfer kernels."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _quantize(width: int, dtype: str, column: bool, silu: bool, tile: int):
    rows = T.dynamic("rows")
    input_width = width * 2 if silu else width
    scale_strides = (1, rows) if column else (width // 128, 1)

    @T.prim_func
    def kernel(
        x: T.Tensor((rows, input_width), dtype),
        out: T.Tensor((rows, width), "float8_e4m3"),
        scales: T.StridedTensor((rows, width // 128), scale_strides, "float32"),
    ):
        with T.Kernel(
            T.ceildiv(rows, tile), width // 128, threads=32 if tile == 1 else 128
        ) as (br, group):
            values = T.alloc_fragment((tile, 128), "float32")
            absolute = T.alloc_fragment((tile, 128), "float32")
            maximum = T.alloc_fragment((tile,), "float32")
            for i, j in T.Parallel(tile, 128):
                row, col = br * tile + i, group * 128 + j
                values[i, j] = 0
                if row < rows:
                    value = x[row, col].astype("float32")
                    if silu:
                        activated = (value / (1 + T.exp(-value))).astype("bfloat16")
                        value = (
                            (
                                activated.astype("float32")
                                * x[row, width + col].astype("float32")
                            )
                            .astype("bfloat16")
                            .astype("float32")
                        )
                    values[i, j] = value
                absolute[i, j] = T.abs(values[i, j])
            T.reduce_max(absolute, maximum, dim=1)
            for i in T.Parallel(tile):
                maximum[i] = T.call_extern(
                    "float32", "__fdiv_rn", T.max(maximum[i], 1e-10), T.float32(448)
                )
                if br * tile + i < rows:
                    scales[br * tile + i, group] = maximum[i]
            for i, j in T.Parallel(tile, 128):
                if br * tile + i < rows:
                    value = T.call_extern(
                        "float32", "__fdiv_rn", values[i, j], maximum[i]
                    )
                    out[br * tile + i, group * 128 + j] = T.min(T.max(value, -448), 448)

    return kernel


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
    if silu_gate:
        if x.dtype != torch.bfloat16 or width % 256:
            raise ValueError("fused SiLU quantization requires packed BF16 gate/up")
        width //= 2
    data = torch.empty((rows, width), device=x.device, dtype=torch.float8_e4m3fn)
    shape = (width // 128, rows) if column_major else (rows, width // 128)
    scales = torch.empty(shape, dtype=torch.float32, device=x.device)
    if column_major:
        scales = scales.T
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
