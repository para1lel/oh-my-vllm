import math

VERSION = "http-reference-v4"
RATES = dict(
    hbm_bytes=7.7e12,
    fp8_flops=4.5e15,
    bf16_flops=2.25e15,
    tf32_flops=1.1e15,
    fp32_flops=75e12,
    fp64_flops=37e12,
)


def add(*items):
    return {name: sum(item.get(name, 0) for item in items) for name in RATES}


def scale(work, count):
    return {name: value * count for name, value in work.items()}


def linear(rows, columns, width, *, fp8=False):
    weights = columns * width * (1 if fp8 else 2)
    scales = 4 * ((columns + 127) // 128) * ((width + 127) // 128) if fp8 else 0
    return dict(
        hbm_bytes=weights + scales,
        **{"fp8_flops" if fp8 else "bf16_flops": 2 * rows * columns * width},
    )


def norm(rows, width):
    return dict(hbm_bytes=4 * width, fp32_flops=rows * (4 * width + 2))


def attention(queries, *, first=0, cached=False):
    pairs = sum(
        count * (start - first) + count * (count + 1) // 2 for start, count in queries
    )
    history = sum(
        max(0, start + (count if cached else 0) - first) for start, count in queries
    )
    return dict(
        hbm_bytes=history * 4096,
        bf16_flops=4 * pairs * 24 * 256,
        fp32_flops=3 * pairs * 24,
    )


def reference_seconds(work: dict[str, float]) -> float:
    if set(work) != set(RATES) or any(
        not math.isfinite(value) or value < 0 for value in work.values()
    ):
        raise ValueError("reference work must contain six finite nonnegative resources")
    return max(
        work["hbm_bytes"] / RATES["hbm_bytes"],
        sum(
            work[name] / RATES[name]
            for name in ("fp8_flops", "bf16_flops", "tf32_flops")
        ),
        work["fp32_flops"] / RATES["fp32_flops"],
        work["fp64_flops"] / RATES["fp64_flops"],
    )
