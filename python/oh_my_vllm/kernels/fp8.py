"""Block-scaled FP8 projections with owned CUDA and FlashInfer providers."""

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
    if x.ndim != 2 or weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("FP8 inputs, weights and scales must be matrices")
    input_width = x.shape[1] // 2 if silu_gate else x.shape[1]
    _validate_weight(x, weight, weight_scale, input_width)
    # Small TRT-LLM GEMM uses column-major activation scales.
    column = x.shape[0] <= 32 or _large_mn_projection(x.shape[0], weight)
    data, scale = quantize(x, column_major=column, silu_gate=silu_gate)
    return _project_quantized(data, scale, weight, weight_scale)


def _validate_weight(x, weight, weight_scale, input_width):
    if weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("FP8 checkpoint weights and scales must be matrices")
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


def _large_mn_projection(rows, weight):
    """Select the measured full gate/up shape, with four-row scale alignment."""
    return NAME == "cuda" and rows >= 32144 and weight.shape == (34816, 5120)


def _project_quantized(data, scale, weight, weight_scale, *, logical_rows=None):
    """Project validated row-major FP8 data without a second quantization."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    # TRT-LLM's independent FlashInfer backend is faster for decode and avoids
    # CUTLASS SM100's nondeterministic 17..32-row low-latency path. Its activation
    # scales are column-major; checkpoint weight scales remain row-major.
    rows = data.shape[0] if logical_rows is None else logical_rows
    small = rows <= 32
    if NAME == "cuda" and not small:
        from .cuda_backend.groupwise import gemm

        mn = _large_mn_projection(rows, weight)
        if mn:
            padded = (rows + 3) // 4 * 4
            if data.shape[0] < padded:
                # The unfused public projection has compact quantization output.
                # The fused model path writes into padded storage directly.
                data = torch.nn.functional.pad(data, (0, 0, 0, padded - rows))
                scale = torch.nn.functional.pad(scale.T, (0, padded - rows)).T
            # Read the supplied scales on every call and graph replay. A hidden
            # cached transpose would miss subsequent changes to this tensor.
            weight_scale = weight_scale.T.contiguous().T
        output = gemm(
            data,
            weight,
            scale,
            weight_scale,
            mma_sm=2 if data.shape[0] >= 4096 and weight.shape[0] >= 32768 else 1,
            scale_major_k=not mn,
        )
        return output[:rows] if mn else output
    return gemm_fp8_nt_groupwise(
        data,
        weight,
        scale,
        weight_scale,
        scale_major_mode="K",
        mma_sm=2 if data.shape[0] >= 256 and weight.shape[0] >= 32768 else 1,
        out_dtype=torch.bfloat16,
        backend="trtllm" if small else "cutlass",
    )


def add_norm_linear(x, residual, gamma, weight, weight_scale):
    """Return the BF16 residual sum and its normalized FP8 projection.

    Model-width CUDA inputs fuse addition, RMS normalization and quantization.
    The residual sum and normalized values keep their BF16 rounding points.
    """
    if (
        x.ndim != 2
        or x.shape[1] != 5120
        or x.numel() == 0
        or residual.shape != x.shape
        or gamma.shape != (5120,)
    ):
        raise ValueError("normalized FP8 projection requires nonempty 5120-wide rows")
    if x.dtype != torch.bfloat16 or residual.dtype != x.dtype:
        raise ValueError("normalized FP8 projection inputs must be BF16")
    if gamma.dtype != torch.float32:
        raise ValueError("normalized FP8 projection gamma must be FP32")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != x.device
        for t in (x, residual, gamma)
    ):
        raise ValueError("normalized FP8 inputs must be contiguous on one CUDA device")
    if x.shape[0] > 2**31 // 5120:
        raise ValueError("normalized FP8 input exceeds int32 flat offset range")
    _validate_weight(x, weight, weight_scale, 5120)
    # At up to 32 rows, the previous CUDA chain is faster than fusion.
    # The explicit TileLang provider also uses its original complete chain.
    if NAME != "cuda" or x.shape[0] <= 32:
        from .normalization import add_rms_norm

        summed, normalized = add_rms_norm(x, residual, gamma)
        return summed, linear(normalized, weight, weight_scale)
    from .cuda_backend import compiled

    rows = x.shape[0]
    column = _large_mn_projection(rows, weight)
    physical = (rows + 3) // 4 * 4 if column else rows
    data = torch.empty((physical, 5120), device=x.device, dtype=torch.float8_e4m3fn)
    shape = (40, physical) if column else (physical, 40)
    scales = torch.empty(shape, device=x.device, dtype=torch.float32)
    if column:
        scales = scales.T
    summed = torch.empty_like(x)
    compiled().rms_quantize(
        x, residual, gamma, data[:rows], scales[:rows], summed, column
    )
    if physical != rows:
        data[rows:].zero_()
        scales[rows:].zero_()
    return summed, _project_quantized(
        data, scales, weight, weight_scale, logical_rows=rows
    )


def gated_norm_linear(x, gamma, gate, weight, weight_scale):
    """Project gated RMS features with the original BF16 rounding boundary.

    x and gate are BF16 [rows,48,128] with packed token strides allowed.
    gamma is the FP32 head multiplier. The FP8 checkpoint weight is [N,6144]
    with FP32 [N/128,48] scales. Outputs are independent BF16 [rows,N] storage.
    """
    if (
        x.ndim != 3
        or x.shape[1:] != (48, 128)
        or x.numel() == 0
        or gate.shape != x.shape
        or gamma.shape != (128,)
    ):
        raise ValueError("gated FP8 projection requires nonempty [rows,48,128] heads")
    if x.dtype != torch.bfloat16 or gate.dtype != x.dtype:
        raise ValueError("gated FP8 projection input and gate must be BF16")
    if gamma.dtype != torch.float32:
        raise ValueError("gated FP8 projection gamma must be FP32")
    if any(not t.is_cuda or t.device != x.device for t in (x, gate, gamma)):
        raise ValueError("gated FP8 inputs must share one CUDA device")
    if (
        any(
            t.stride(0) < 6144 or t.stride(1) != 128 or t.stride(2) != 1
            for t in (x, gate)
        )
        or not gamma.is_contiguous()
    ):
        raise ValueError("gated FP8 inputs require contiguous heads and gamma")
    if x.shape[0] > 2**31 // 6144:
        raise ValueError("gated FP8 input exceeds int32 flat offset range")
    _validate_weight(x, weight, weight_scale, 6144)
    if NAME != "cuda":
        from .normalization import rms_norm

        normalized = rms_norm(x, gamma, gate=gate)
        return linear(normalized.flatten(1), weight, weight_scale)
    from .cuda_backend import compiled

    rows = x.shape[0]
    column = rows <= 32
    data = torch.empty((rows, 6144), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (48, rows) if column else (rows, 48), device=x.device, dtype=torch.float32
    )
    if column:
        scales = scales.T
    compiled().gated_quantize(x, gamma, gate, data, scales, column)
    return _project_quantized(data, scales, weight, weight_scale)
