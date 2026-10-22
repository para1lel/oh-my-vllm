"""Independent FP64 decoding of original checkpoint storage with NumPy only.

Mathematical sources: released inference/convert.py:cast_e2m1fn_to_e4m3fn,
model.py:Linear and ParallelEngramEmbedding, and kernel.py:fp8_gemm_kernel
and fp4_gemm_kernel. This module does not emulate runtime rounding.
"""

import numpy as np

_STORAGE = {
    "BF16": ("u", 2),
    "F32": ("f", 4),
    "F8_E4M3": ("u", 1),
    "F8_E8M0": ("u", 1),
    "I8": ("i", 1),
}


def _require_storage(raw, kind, size, name):
    if not isinstance(raw, np.ndarray):
        raise ValueError(f"{name}: expected a NumPy array")
    if raw.dtype.kind != kind or raw.dtype.itemsize != size:
        raise ValueError(f"{name}: unsupported storage dtype {raw.dtype}")


def _bf16(raw):
    exponent = ((raw >> 7) & 255).astype(np.int16)
    if np.any(exponent == 255):
        raise ValueError("BF16: nonfinite encoding")
    mantissa = raw & 127
    value = np.ldexp(
        np.where(exponent == 0, mantissa, mantissa + 128).astype(np.float64),
        np.where(exponent == 0, -133, exponent - 134),
    )
    return np.copysign(value, np.where(raw & 0x8000, -1.0, 1.0))


def _e4m3(raw):
    if np.any((raw & 127) == 127):
        raise ValueError("F8_E4M3: nonfinite encoding")
    exponent = ((raw >> 3) & 15).astype(np.int16)
    mantissa = raw & 7
    value = np.ldexp(
        np.where(exponent == 0, mantissa, mantissa + 8).astype(np.float64),
        np.where(exponent == 0, -9, exponent - 10),
    )
    return np.copysign(value, np.where(raw & 128, -1.0, 1.0))


def _e8m0(raw):
    if np.any(raw == 255):
        raise ValueError("F8_E8M0: nonfinite encoding")
    # Byte zero is 2**-127, not zero. There is no sign bit or zero encoding.
    return np.ldexp(np.ones(raw.shape, dtype=np.float64), raw.astype(np.int16) - 127)


def _e2m1(raw):
    bits = raw.view(np.uint8)
    nibbles = np.stack((bits & 15, bits >> 4), axis=-1).reshape(
        raw.shape[0], raw.shape[1] * 2
    )
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    return np.copysign(magnitudes[nibbles & 7], np.where(nibbles & 8, -1.0, 1.0))


def _scaled_shape(raw, dtype, scales, scale_layout):
    if dtype not in ("F8_E4M3", "I8"):
        raise ValueError(f"{dtype}: scales are not supported")
    if scale_layout not in ("dense32", "row32"):
        raise ValueError("Expected scale_layout dense32 or row32")
    if dtype == "I8" and scale_layout != "row32":
        raise ValueError("Packed I8 requires row32 scales")
    if raw.ndim != 2 or any(d == 0 for d in raw.shape):
        raise ValueError("Scaled weights must be a nonempty matrix")
    rows, columns = raw.shape
    if dtype == "I8":
        columns *= 2
    if scale_layout == "row32" and columns % 32:
        raise ValueError("row32 requires a logical column count divisible by 32")
    _require_storage(scales, "u", 1, "UE8M0 scales")
    row_stride = 32 if scale_layout == "dense32" else 1
    expected = ((rows + row_stride - 1) // row_stride, (columns + 31) // 32)
    if scales.shape != expected:
        raise ValueError(f"Expected scale shape {expected}; received {scales.shape}")
    return row_stride


def _unscaled(raw, dtype):
    if dtype == "BF16":
        return _bf16(raw)
    if dtype == "F8_E4M3":
        return _e4m3(raw)
    if dtype == "F8_E8M0":
        return _e8m0(raw)
    if dtype == "I8":
        return _e2m1(raw)
    if not np.all(np.isfinite(raw)):
        raise ValueError("F32: nonfinite encoding")
    return raw.astype(np.float64)


def decode_tensor(
    raw: np.ndarray,
    dtype: str,
    *,
    scales: np.ndarray | None = None,
    scale_layout: str | None = None,
) -> np.ndarray:
    """Decode checkpoint storage into a new float64 array without input changes.

    BF16 uses uint16 bit patterns; F32 uses float32 values. FP8 formats use
    uint8 bit patterns. I8 uses signed int8 bytes with the low FP4 nibble first.
    Byte order is carried by the NumPy dtype for BF16 and F32.

    Scales must be raw uint8 UE8M0 encodings, not decoded floating-point values.
    dense32 supplies one scale per 32-by-32 block, including partial edge blocks.
    row32 supplies one scale per row and 32 columns; width must be a multiple of 32.
    I8 requires row32 and doubles the last dimension. Only F8_E4M3 and I8 accept
    scales. Without scales, all other supported formats keep their input shape.
    Invalid formats, nonfinite encodings, and bad scale contracts raise ValueError.
    """
    if not isinstance(dtype, str) or dtype not in _STORAGE:
        raise ValueError(f"Unsupported checkpoint format: {dtype!r}")
    _require_storage(raw, *_STORAGE[dtype], dtype)
    if scales is None and scale_layout is None:
        if dtype == "I8":
            raise ValueError("Packed I8 requires row32 scales")
        return np.asarray(_unscaled(raw, dtype), dtype=np.float64)
    if scales is None or not isinstance(scale_layout, str):
        raise ValueError("scales and scale_layout must be supplied together")
    row_stride = _scaled_shape(raw, dtype, scales, scale_layout)
    decoded_scales = _e8m0(scales)
    values = _unscaled(raw, dtype)
    row_ids = np.arange(values.shape[0]) // row_stride
    column_ids = np.arange(values.shape[1]) // 32
    return values * decoded_scales[row_ids[:, None], column_ids[None, :]]
