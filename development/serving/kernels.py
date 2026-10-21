import math
from dataclasses import dataclass, replace

from .accounting import RATES, add, attention, linear, norm, reference_seconds, scale


@dataclass(frozen=True)
class Group:
    name: str
    work: dict
    repetitions: int = 1


def group(name, *items):
    return Group(name, add(*items))


def repeated(groups, count):
    return (
        [replace(item, repetitions=item.repetitions * count) for item in groups]
        if count
        else []
    )


def input_credit(groups, size):
    first, *remaining = groups
    credited = replace(first, work=add(first.work, dict(hbm_bytes=size)), repetitions=1)
    return [
        credited,
        *(
            [replace(first, repetitions=first.repetitions - 1)]
            if first.repetitions > 1
            else []
        ),
        *remaining,
    ]


def resources(groups):
    return add(*(scale(item.work, item.repetitions) for item in groups))


def serial_seconds(groups, *, cache_bytes=0):
    if not math.isfinite(cache_bytes) or cache_bytes < 0:
        raise ValueError("cache allowance must be finite and nonnegative")
    if any(
        type(item.repetitions) is not int or item.repetitions < 1 for item in groups
    ):
        raise ValueError("kernel repetitions must be positive integers")
    raw = math.fsum(item.repetitions * reference_seconds(item.work) for item in groups)
    compute = math.fsum(
        item.repetitions * reference_seconds(dict(item.work, hbm_bytes=0))
        for item in groups
    )
    # One cache budget can hide only the memory time above each group's compute floor.
    return max(compute, raw - cache_bytes / RATES["hbm_bytes"])


def normalization(rows):
    return add(norm(rows, 5120), dict(fp32_flops=rows * 5120))


def ffn_groups(rows):
    normalize = normalization(rows)
    upper = linear(rows, 34816, 5120, fp8=True)
    activation = dict(fp32_flops=rows * 2 * 17408)
    down = linear(rows, 5120, 17408, fp8=True)
    if rows in (1, 4, 5):
        return [
            group("ffn.input-pdl", normalize, upper, activation),
            group("ffn.down", down),
        ]
    if rows <= 32:
        return [
            group("ffn.input-pdl", normalize, upper),
            *(
                [group("ffn.activation", activation), group("ffn.down", down)]
                if rows <= 8
                else [group("ffn.output-pdl", activation, down)]
            ),
        ]
    if 65 <= rows <= 1024:
        return [group("ffn.prefill-pdl", normalize, upper, activation, down)]
    return [
        group("ffn.norm", normalize),
        *(
            [group("ffn.upper", upper, activation)]
            if rows > 64
            else [group("ffn.upper", upper), group("ffn.activation", activation)]
        ),
        group("ffn.down", down),
    ]


def fa_groups(queries, *, variant="full", first=0, writes=0):
    rows = sum(count for _, count in queries)
    heads = {"full": 28, "query_only": 24, "kv_only": 4}[variant]
    columns = {"full": 14336, "query_only": 12288, "kv_only": 2048}[variant]
    normalize = normalization(rows)
    projection = linear(rows, columns, 5120, fp8=True)
    result = (
        [group("fa.input-fused", normalize, projection)]
        if rows == 1 and variant != "kv_only"
        else [group("fa.norm", normalize), group("fa.input", projection)]
    )
    result.append(
        group(
            "fa.heads",
            dict(
                hbm_bytes=writes * 4096 + 4 * (512 if variant == "full" else 256),
                fp32_flops=rows * heads * (4 * 256 + 2 + 64 * 3),
            ),
        )
    )
    if variant != "kv_only":
        result.extend(
            [
                group(
                    "fa.attention",
                    attention(queries, first=first, cached=variant == "query_only"),
                ),
                group("fa.gate", dict(fp32_flops=rows * 6144)),
                group("fa.output", linear(rows, 5120, 6144, fp8=True)),
            ]
        )
    return result


def sample_groups(rows):
    return (
        [
            group("output.norm", normalization(rows)),
            group("output.projection", linear(rows, 248320, 5120)),
        ]
        if rows
        else []
    )


def input_groups(rows, *, embedding_bytes=0):
    return [
        group("mtp.norm", scale(norm(rows, 5120), 2), dict(hbm_bytes=embedding_bytes)),
        group("mtp.fc", linear(rows, 5120, 10240)),
    ]


def context_groups(rows, *, writes=True, embedding_bytes=0):
    if not rows:
        return []
    result = input_groups(rows, embedding_bytes=embedding_bytes)
    result.extend(
        [
            group("mtp.context.norm", norm(rows, 5120)),
            group("mtp.context.kv", linear(rows, 2048, 5120, fp8=True)),
            group(
                "mtp.context.heads",
                norm(rows * 4, 256),
                dict(
                    hbm_bytes=rows * 4096 if writes else 0, fp32_flops=rows * 4 * 64 * 3
                ),
            ),
        ]
    )
    return result


def draft_groups(queries, stage, *, writes=True, embedding_bytes=0):
    rows = len(queries)
    full = stage != 0
    return [
        *(input_groups(rows, embedding_bytes=embedding_bytes) if full else []),
        *fa_groups(
            queries,
            variant="full" if full else "query_only",
            first=1,
            writes=rows if full and writes else 0,
        ),
        *ffn_groups(rows),
        *sample_groups(rows),
    ]


def wy_groups(queries, state_bytes):
    prepare, carry, output = add(), add(), add()
    carry["hbm_bytes"] = state_bytes
    for start, count in queries:
        while count:
            n = min(count, 64, 784 - start % 784)
            diagonal = sum(
                b * (b - 1) * (b - 2) // 3
                for b in (min(8, n - first) for first in range(0, n, 8))
            )
            inverse = n * (n - 1) * (n - 2) // 3 - diagonal
            causal = n * (n + 1) // 2
            prepare = add(
                prepare,
                dict(
                    bf16_flops=2 * 16 * 128 * n * n
                    + 2 * 48 * 128 * n * (n + 1)
                    + 48 * inverse,
                    fp32_flops=48 * (diagonal + n * (n - 1) + 3 * causal + 128 * n),
                ),
            )
            carry = add(
                carry,
                dict(
                    bf16_flops=4 * 48 * 128 * 128 * n,
                    fp32_flops=48 * (2 * 128 * 128 + 2 * 128 * n),
                ),
            )
            output = add(
                output,
                dict(
                    bf16_flops=2 * 48 * 128 * 128 * n + 48 * 128 * n * (n + 1),
                    fp32_flops=48 * (2 * 128 * n + n),
                ),
            )
            start += n
            count -= n
    return [
        group("gdn.wy.prepare", prepare),
        group("gdn.wy.carry", carry),
        group("gdn.wy.output", output),
    ]


def gdn_groups(queries, verify, *, states=None):
    rows = sum(count for _, count in queries)
    maximum = max(count for _, count in queries)
    states = sum(start > 0 for start, _ in queries) if states is None else states
    small = rows <= 32
    matrix_ba = not small or (verify and maximum == 1)
    chunked = not verify and maximum > 128
    split_values = not small or verify
    normalize = normalization(rows)
    projected = linear(rows, 16384, 5120, fp8=True)
    ba = linear(rows, 96, 5120)
    if not matrix_ba:
        ba["fp32_flops"] = ba.pop("bf16_flops")
    conv = dict(
        hbm_bytes=states * 10240 * 3 * 2 + 10240 * 4 * 2, fp32_flops=rows * 7 * 10240
    )
    if rows == 1 and verify:
        projected = add(projected, conv)
    if not matrix_ba and rows > 1:
        conv = add(conv, ba)
    result = (
        [group("gdn.input-pdl", normalize, projected)]
        if small
        else [group("gdn.norm", normalize), group("gdn.input", projected)]
    )
    if matrix_ba:
        result.append(group("gdn.ba", ba))
    gates = dict(hbm_bytes=4 * 96)
    if not small and not verify:
        result.append(group("gdn.gates", gates))
    if not (rows == 1 and verify):
        result.append(group("gdn.convolution", conv))
    state_bytes = states * 48 * 128 * 128 * 4
    if chunked:
        result.extend(wy_groups(queries, state_bytes))
    else:
        recurrence = add(
            dict(
                hbm_bytes=state_bytes,
                fp32_flops=rows * (7 * 48 * 128 * 128 + 2 * 48 * 128 + 16 * 128),
            )
        )
        if not matrix_ba and rows == 1:
            recurrence = add(recurrence, ba)
        if small or verify:
            recurrence = add(recurrence, gates)
        if not split_values:
            recurrence = add(recurrence, dict(hbm_bytes=4 * 128))
        result.append(group("gdn.recurrent", recurrence))
    projected = linear(rows, 5120, 6144, fp8=True)
    if split_values:
        quantize = dict(hbm_bytes=4 * 128)
        if 8 < rows <= 32:
            projected = add(projected, quantize)
        else:
            result.append(group("gdn.quant", quantize))
    elif 8 < rows <= 32:
        result[-1] = group("gdn.output-pdl", result[-1].work, projected)
        return result
    result.append(
        group("gdn.output-pdl" if 8 < rows <= 32 else "gdn.output", projected)
    )
    return result


def target_groups(rows, verify):
    queries = [(row["start"], row["count"]) for row in rows]
    tokens = sum(count for _, count in queries)
    states = sum(
        int(row["start"] > 0)
        + row["end"] // 784
        - row["start"] // 784
        + int(row["end"] % 784 != 0)
        for row in rows
    )
    return [
        *repeated(gdn_groups(queries, verify, states=states), 48),
        *repeated(
            fa_groups(queries, writes=sum(row["end"] - row["start"] for row in rows)),
            16,
        ),
        *repeated(ffn_groups(tokens), 64),
    ]


def pass_groups(logical):
    rows = logical["requests"]
    samples = sum(
        row["count"] if logical["verify"] else int(row["tokens"] > 0) for row in rows
    )
    result = target_groups(rows, logical["verify"]) + sample_groups(samples)
    if logical["mtp"]:
        result.extend(context_groups(sum(row["context_count"] for row in rows)))
        for stage in range(4):
            queries = [
                (row["end"] + stage, 1) for row in rows if row["proposed"] > stage
            ]
            if queries:
                result.extend(draft_groups(queries, stage))
    return result


def reference(passes):
    totals = {}
    for logical in passes:
        phase = "decode" if logical["verify"] else "prefill"
        for item in pass_groups(logical):
            name = f"{phase}.{item.name}"
            entry = totals.setdefault(name, dict(groups=0, seconds=0.0, work=add()))
            entry["groups"] += item.repetitions
            entry["seconds"] += item.repetitions * reference_seconds(item.work)
            entry["work"] = add(entry["work"], scale(item.work, item.repetitions))
    return dict(
        seconds=math.fsum(item["seconds"] for item in totals.values()),
        work=add(*(item["work"] for item in totals.values())),
        groups=totals,
    )
