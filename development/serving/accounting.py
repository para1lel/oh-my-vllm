import math

VERSION = "http-reference-v1"
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


def ffn(rows):
    return add(
        linear(rows, 34816, 5120, fp8=True),
        linear(rows, 5120, 17408, fp8=True),
        norm(rows, 5120),
        dict(fp32_flops=rows * (5120 + 2 * 17408)),
    )


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


def gdn_math(queries, verify):
    if verify:
        return dict(
            fp32_flops=sum(count for _, count in queries)
            * 48
            * (7 * 128 * 128 + 3 * 128)
        )
    bf16 = fp32 = 0
    for start, count in queries:
        while count:
            rows = min(count, 64, 784 - start % 784)
            diagonal = sum(
                size * (size - 1) * (size - 2) // 3
                for size in (min(8, rows - first) for first in range(0, rows, 8))
            )
            inverse = rows * (rows - 1) * (rows - 2) // 3 - diagonal
            bf16 += (
                6 * 48 * 128 * 128 * rows
                + 48 * 128 * rows * (3 * rows - 1)
                + 2 * 16 * 128 * rows * rows
                + 48 * inverse
            )
            fp32 += 48 * (
                diagonal
                + 2 * 128 * 128
                + 8 * 128 * rows
                + rows * (rows - 1)
                + rows * (rows + 1) // 2
                + 2 * rows
            )
            start += rows
            count -= rows
    return dict(bf16_flops=bf16, fp32_flops=fp32)


def target_work(rows, verify):
    queries = [(row["start"], row["count"]) for row in rows]
    tokens = sum(row["count"] for row in rows)
    states = sum(
        int(row["start"] > 0)
        + row["end"] // 784
        - row["start"] // 784
        + int(row["end"] % 784 != 0)
        for row in rows
    )
    gdn = add(
        linear(tokens, 16384, 5120, fp8=True),
        linear(tokens, 96, 5120),
        linear(tokens, 5120, 6144, fp8=True),
        norm(tokens, 5120),
        gdn_math(queries, verify),
        dict(
            hbm_bytes=states * (48 * 128 * 128 + 10240 * 3) * 2
            + 10240 * 4 * 2
            + 4 * (128 + 96),
            fp32_flops=tokens * (5120 + 7 * 10240),
        ),
    )
    fa = add(
        linear(tokens, 14336, 5120, fp8=True),
        linear(tokens, 5120, 6144, fp8=True),
        norm(tokens, 5120),
        attention(queries),
        dict(
            hbm_bytes=sum(row["end"] - row["start"] for row in rows) * 4096 + 4 * 512,
            fp32_flops=tokens * (5120 + 6144 + 28 * (4 * 256 + 2) + 28 * 64 * 3),
        ),
    )
    return add(scale(gdn, 48), scale(fa, 16), scale(ffn(tokens), 64))


def sample_work(rows):
    return (
        add(linear(rows, 248320, 5120), norm(rows, 5120), dict(fp32_flops=rows * 5120))
        if rows
        else add()
    )


def context_work(rows):
    return (
        add(
            linear(rows, 5120, 10240),
            linear(rows, 2048, 5120, fp8=True),
            scale(norm(rows, 5120), 3),
            norm(rows * 4, 256),
            dict(hbm_bytes=rows * 4096, fp32_flops=rows * 4 * 64 * 3),
        )
        if rows
        else add()
    )


def draft_work(rows, stage):
    count = len(rows)
    full = stage != 0
    heads = 28 if full else 24
    queries = [(row["end"] + stage, 1) for row in rows]
    return add(
        linear(count, 5120, 10240) if full else add(),
        linear(count, 14336 if full else 12288, 5120, fp8=True),
        linear(count, 5120, 6144, fp8=True),
        ffn(count),
        sample_work(count),
        attention(queries, first=1, cached=not full),
        scale(norm(count, 5120), 3 if full else 1),
        dict(
            hbm_bytes=(count * 4096 if full else 0) + 4 * (512 if full else 256),
            fp32_flops=count * (5120 + 6144 + heads * (4 * 256 + 2) + heads * 64 * 3),
        ),
    )


def reference_work(passes: list[dict]) -> dict[str, float]:
    total = add()
    for logical in passes:
        rows = logical["requests"]
        samples = sum(
            row["count"] if logical["verify"] else int(row["tokens"] > 0)
            for row in rows
        )
        total = add(total, target_work(rows, logical["verify"]), sample_work(samples))
        if logical["mtp"]:
            total = add(total, context_work(sum(row["context_count"] for row in rows)))
            for stage in range(4):
                drafts = [row for row in rows if row["proposed"] > stage]
                if drafts:
                    total = add(total, draft_work(drafts, stage))
    return total


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
