"""Released block quantization with nearest-even value and scale rounding."""

import torch

from ...operators.runtime.native import load
from ...operators.transform.api import FUNCTIONS


def _groups(values, block_size):
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("quantization block size must be positive")
    if values.ndim == 0 or values.size(-1) == 0 or values.size(-1) % block_size:
        raise ValueError("quantization width must be divisible by the block size")
    return values.float().reshape(
        *values.shape[:-1], values.size(-1) // block_size, block_size
    )


def _power_scale(values):
    # log2 can discard the fraction before ceil at large exponents.
    bits = values.contiguous().view(torch.int32)
    exponent = bits & 0x7F800000
    rounded = exponent + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    return torch.where(torch.isfinite(values), rounded.view(torch.float32), values)


def _native(values, bits, block_size, scale_dtype, round_scale, inplace):
    if (
        not values.is_cuda
        or values.dtype != torch.bfloat16
        or block_size not in (16, 32, 128)
    ):
        return None
    if values.ndim == 0 or values.size(-1) == 0 or values.size(-1) % block_size:
        raise ValueError("quantization width must be divisible by the block size")
    inputs = values.contiguous()
    if inputs.data_ptr() % 16:
        inputs = inputs.clone()
    width = inputs.size(-1)
    output = (
        inputs
        if inplace
        else torch.empty(
            (*inputs.shape[:-1], width * bits // 8),
            dtype=torch.uint8,
            device=inputs.device,
        )
    )
    scales = torch.empty(
        (*inputs.shape[:-1], width // block_size),
        dtype=scale_dtype,
        device=inputs.device,
    )
    scale_format = {
        torch.float32: 0,
        torch.float8_e8m0fnu: 1,
        torch.float8_e4m3fn: 2,
    }[scale_dtype]
    rows = inputs.numel() // width
    load("transform", FUNCTIONS)[0].quantize(
        inputs.view(rows, width),
        output.view(rows, width if inplace else width * bits // 8),
        scales.view(rows, width // block_size).view(
            torch.float32 if scale_dtype == torch.float32 else torch.uint8
        ),
        bits,
        block_size,
        scale_format,
        round_scale,
        inplace,
    )
    if inplace:
        if inputs is not values:
            values.copy_(output)
        return values
    return output.view(
        torch.float8_e4m3fn if bits == 8 else torch.float4_e2m1fn_x2
    ), scales


def act_quant(
    values, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    if scale_dtype not in (torch.float32, torch.float8_e8m0fnu):
        raise ValueError("FP8 scales must use FP32 or UE8M0")
    native = _native(values, 8, block_size, scale_dtype, scale_fmt is not None, inplace)
    if native is not None:
        return native
    groups = _groups(values, block_size)
    scales = groups.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448)
    if scale_fmt is not None:
        scales = _power_scale(scales)
    encoded = (groups / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        values.copy_((encoded.float() * scales.unsqueeze(-1)).reshape_as(values))
        return values
    return encoded.reshape_as(values), scales.to(scale_dtype)


def fp4_values(encoded):
    raw = encoded.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    levels = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=raw.device)
    return levels[(codes & 7).long()] * torch.where((codes & 8).bool(), -1.0, 1.0)


def _fp4_codes(values):
    absolute = values.abs()
    thresholds = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=values.device
    )
    indices = torch.bucketize(absolute.contiguous(), thresholds)
    ties = absolute == thresholds[indices.clamp_max(6)]
    indices += (ties & (indices.remainder(2) == 1)).long()
    return indices.to(torch.uint8) | (values.signbit().to(torch.uint8) << 3)


def fp4_act_quant(
    values, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu
):
    if scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise ValueError("FP4 scales must use UE8M0 or E4M3")
    native = _native(values, 4, block_size, scale_dtype, False, inplace)
    if native is not None:
        return native
    groups = _groups(values, block_size)
    maximum = groups.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        scales = (maximum.clamp_min(6 * 2.0**-9) / 6).to(scale_dtype)
    else:
        scales = _power_scale(maximum.clamp_min(6 * 2.0**-126) * (1.0 / 6)).to(
            scale_dtype
        )
    normalized = (groups / scales.float().unsqueeze(-1)).clamp(-6, 6)
    codes = _fp4_codes(normalized.reshape_as(values))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    encoded = packed.view(torch.float4_e2m1fn_x2)
    if inplace:
        restored = fp4_values(encoded).reshape_as(groups) * scales.float().unsqueeze(-1)
        values.copy_(restored.reshape_as(values))
        return values
    return encoded, scales
