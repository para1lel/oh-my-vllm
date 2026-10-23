"""Independent NumPy source-precision activation and cache quantization.

LOCAL-V41:inference/kernel.py:fast_round_scale, act_quant_kernel and
fp4_quant_kernel define the arithmetic. model.py selects the three layouts
below. No runtime/native kernel is imported. This does not replace the FP64
mathematical reference or establish compound numerical acceptance limits.
"""

from typing import NamedTuple

import numpy as np

from .quantization import decode_tensor

SOURCE_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
_E4M3 = decode_tensor(np.arange(127, dtype=np.uint8), "F8_E4M3").astype(np.float32)
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


class Quantized(NamedTuple):
    encoded: np.ndarray
    scales: np.ndarray
    dequantized: np.ndarray


def _bf16(values):
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    result = rounded.view(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("quantization exceeds finite BF16 range")
    return result


def _input(values, group_size):
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float32):
        raise ValueError("expected a float32 NumPy array")
    if values.ndim == 0 or not values.size or values.shape[-1] % group_size:
        raise ValueError(f"expected nonempty rows with width divisible by {group_size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("quantization requires finite input")
    return _bf16(values).reshape(*values.shape[:-1], -1, group_size)


def _encode(values, alphabet, sign_bit):
    magnitude = np.minimum(np.abs(values), alphabet[-1])
    upper = np.searchsorted(alphabet, magnitude)
    lower = np.maximum(upper - 1, 0)
    below, above = magnitude - alphabet[lower], alphabet[upper] - magnitude
    indices = np.where(
        (above < below) | ((above == below) & (upper % 2 == 0)), upper, lower
    )
    codes = indices.astype(np.uint8) | (np.signbit(values).astype(np.uint8) * sign_bit)
    return codes, np.copysign(alphabet[indices], values)


def _power_scale(amax, maximum):
    # fast_round_scale multiplies by an FP32 reciprocal before exponent rounding.
    scaled = amax * np.float32(1 / maximum)
    bits = scaled.view(np.uint32)
    exponent = ((bits >> 23) & 255).astype(np.int32)
    exponent += (bits & 0x7FFFFF) != 0
    if np.any((exponent < 1) | (exponent > 254)):
        raise ValueError("quantization exceeds finite normal UE8M0 scale range")
    return exponent.astype(np.uint8), np.ldexp(
        np.ones(amax.shape, dtype=np.float32), exponent - 127
    )


def _quantize(values, group_size, maximum, e4m3_scale=False):
    grouped = _input(values, group_size)
    amax = np.max(np.abs(grouped), axis=-1)
    if e4m3_scale:
        amax = np.maximum(amax, np.float32(6 * 2**-9))
        # This branch divides in FP32, then rounds the scale itself to E4M3.
        scales, scale_values = _encode(amax / np.float32(6), _E4M3, 128)
    else:
        floor = 1e-4 if maximum == 448 else 6 * 2**-126
        scales, scale_values = _power_scale(
            np.maximum(amax, np.float32(floor)), maximum
        )
    normalized = np.clip(grouped / scale_values[..., None], -maximum, maximum)
    alphabet, sign_bit = (_E4M3, 128) if maximum == 448 else (_E2M1, 8)
    encoded, decoded = _encode(normalized, alphabet, sign_bit)
    with np.errstate(over="ignore"):
        restored = decoded * scale_values[..., None]
    if not np.all(np.isfinite(restored)):
        raise ValueError("quantization exceeds finite FP32 output range")
    dequantized = _bf16(restored).reshape(values.shape)
    encoded = encoded.reshape(values.shape)
    if maximum == 6:
        encoded = encoded[..., ::2] | (encoded[..., 1::2] << 4)
    return Quantized(encoded, scales, dequantized)


def activation_fp8(values):
    """Quantize dense activations or window KV in groups of 32 with UE8M0 scales.

    Input is a finite float32 NumPy array, rounded to BF16 first. Returns uint8
    E4M3 codes, uint8 scale codes, and the BF16 cache roundtrip as float32 values.
    Leading dimensions are preserved. The input is not changed.
    """
    return _quantize(values, 32, 448)


def indexer_fp4(values):
    """Quantize indexer Q/K in groups of 32 with UE8M0 scales.

    The input and return rules match activation_fp8. Codes are packed uint8
    bytes: channel 2j occupies the low nibble, channel 2j+1 the high nibble.
    """
    return _quantize(values, 32, 6)


def compressed_kv_fp4(values):
    """Quantize compressed main KV in groups of 16 with E4M3 scale codes.

    The input, packed nibble order and BF16 return rules match indexer_fp4.
    Finite overflow saturates when converted to E4M3 or E2M1; nonfinite input
    and nonfinite BF16 input/output conversions raise ValueError.
    """
    return _quantize(values, 16, 6, e4m3_scale=True)
