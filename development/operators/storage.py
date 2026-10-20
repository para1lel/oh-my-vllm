import numpy as np


def equal_bits(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )
    )


def storage_dtype(dtype):
    return {"bfloat16": "uint16", "float8_e4m3fn": "uint8"}.get(dtype, dtype)


def decode(value, dtype):
    if dtype == "bfloat16":
        return (value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "float8_e4m3fn":
        bits = value.astype(np.uint8)
        exponent, fraction = (bits >> 3) & 15, bits & 7
        magnitude = np.where(
            exponent == 0,
            fraction.astype(np.float64) * 2.0**-9,
            (1 + fraction.astype(np.float64) / 8) * np.exp2(exponent.astype(int) - 7),
        )
        magnitude[(exponent == 15) & (fraction == 7)] = np.nan
        return np.where(bits & 128, -magnitude, magnitude)
    return value.astype(np.float64) if value.dtype.kind == "f" else value


def state_rows(value, layout):
    if layout == "kv":
        return value.transpose(0, 2, 1, 3, 4).reshape(-1, 2, 4, 256)
    if layout == "rows":
        return value
    raise ValueError(f"unknown state layout: {layout}")
