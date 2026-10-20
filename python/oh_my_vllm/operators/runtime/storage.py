from math import prod

WIDTHS = {
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float32": 4,
    "float64": 8,
    "int32": 4,
    "int64": 8,
    "uint64": 8,
    "uint8": 1,
}


def storage_bytes(fields):
    return sum(
        ((max(prod(shape) * WIDTHS[dtype], 1) + 255) // 256) * 256
        for shape, dtype in fields
    )
