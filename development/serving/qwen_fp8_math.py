"""Useful Qwen arithmetic. Mandatory rows exclude speculative and replay work."""

from .accounting import add, linear, reference_seconds

FAST_MEMORY_BYTES = 1 << 30
CAPACITY_SOURCES = (
    "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy",
    "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#increased-l2-capacity",
    "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory",
)
EXCLUSIONS = (
    "absmax, datatype conversion, rounding, SFU and integer instruction time",
    "sampling, penalties, grammar, sorting, RNG and acceptance instruction time",
    "proposal and rejected work, padding, replay, copies, COW and rollback",
    "unproved history reloads, writebacks, temporary buffers and launch overhead",
)
TARGET_MATRICES = (
    (48, 16384, 5120),
    (48, 5120, 6144),
    (16, 14336, 5120),
    (16, 5120, 6144),
    (64, 34816, 5120),
    (64, 5120, 17408),
)
SOURCES = {
    "matrices": "models/qwen38/weights.py:81-164",
    "ba": "operators/gdn_radix/cuda/gdn_radix.cu:244-296",
    "wy_inverse": "operators/gdn/cuda/chunk/gdn_wy_bf16_inverse.cuh:54-162",
    "wy_products": "operators/gdn/cuda/chunk/gdn_wy_prepare_stages.cuh:3-43",
    "wy_carry": "operators/gdn/cuda/chunk/gdn_wy_register.cuh:85-173",
    "wy_output": "operators/gdn/cuda/chunk/gdn_wy_native_output.cuh:98-126",
    "recurrence": "operators/gdn/cuda/recurrent/gdn_recurrent_steps.cuh:142-175",
    "head": "operators/output/cuda/sample.cu:306-307",
}


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def ba_precision(rows, maximum, verify):
    integer(rows, "modeled rows", 1)
    integer(maximum, "modeled maximum", 1)
    if type(verify) is not bool or maximum > rows:
        raise ValueError("invalid modeled GDN dispatch")
    return "bf16" if rows > 32 or (verify and maximum == 1) else "fp32"


def triangular_inverse(n):
    integer(n, "chunk rows", 1)
    if n > 64:
        raise ValueError("GDN chunk exceeds 64 rows")
    scalar = sum(
        b * (b - 1) * (b - 2) // 3
        for b in (min(8, n - first) for first in range(0, n, 8))
    )
    tensor = 0
    for size in (16, 32, 64):
        half = size // 2
        for base in range(0, n, size):
            upper, lower = min(half, n - base), max(0, min(half, n - base - half))
            if lower:
                tensor += lower * upper * (lower - 1) + lower * upper * (upper - 1)
    return dict(bf16_flops=48 * tensor, fp32_flops=48 * scalar)


def wy_chunk(start, n):
    integer(start, "chunk start")
    causal = n * (n + 1) // 2
    nonzero = int(start > 0)
    inverse = triangular_inverse(n)
    return add(
        inverse,
        dict(
            bf16_flops=2 * 16 * 128 * n * n
            + (2 + 2 * nonzero) * 48 * 128 * causal
            + (2 + 4 * nonzero) * 48 * 128 * 128 * n
            + 2 * 48 * 128 * causal,
            fp32_flops=48
            * (
                n * (n - 1)
                + 128 * n * nonzero
                + 128 * n * nonzero
                + 128 * (n - 1)
                + 2 * nonzero * 128 * 128
                + (n + 2 * 128 * n) * nonzero
            ),
        ),
    )


def gdn_arithmetic(queries, verify):
    chunked = not verify and max(count for _, count in queries) > 128
    work = add()
    for start, count in queries:
        if not chunked:
            nonzero = count - int(start == 0)
            work = add(
                work,
                dict(
                    fp32_flops=48
                    * (
                        nonzero * (7 * 48 * 128 * 128 + 2 * 48 * 128)
                        + int(start == 0) * (3 * 48 * 128 * 128 + 48 * 128)
                    )
                ),
            )
            continue
        end = start + count
        while start < end:
            stop = min(start + 64, end, (start // 8192 + 1) * 8192)
            work = add(
                work,
                {
                    name: value * 48
                    for name, value in wy_chunk(start, stop - start).items()
                },
            )
            start = stop
    return work


def weight_objects(head, mtp):
    objects = {}
    for index, (layers, columns, width) in enumerate(TARGET_MATRICES):
        for layer in range(layers):
            objects[f"target.{index}.{layer}"] = linear(1, columns, width, fp8=True)[
                "hbm_bytes"
            ]
    for layer in range(48):
        objects[f"target.ba.{layer}"] = 2 * 96 * 5120
        objects[f"target.conv.{layer}"] = 2 * 10240 * 4
    objects["target.input_and_ffn_norms"] = 64 * 2 * 4 * 5120
    objects["target.gdn_gates_and_norms"] = 48 * 4 * (96 + 128)
    objects["target.fa_head_norms"] = 16 * 4 * (256 + 256)
    objects["target.final_norm"] = 4 * 5120
    if head:
        objects["vocabulary"] = 2 * 248320 * 5120
    if mtp:
        objects["mtp.fc"] = 2 * 5120 * 10240
        objects["mtp.kv"] = linear(1, 2048, 5120, fp8=True)["hbm_bytes"]
        objects["mtp.norms"] = 4 * (3 * 5120 + 256)
    return objects


def pass_resources(logical, component_function):
    rows = logical["requests"]
    queries = [(row["start"], row["end"] - row["start"]) for row in rows]
    useful = sum(count for _, count in queries)
    modeled = sum(row["dispatch_count"] for row in rows)
    if logical.get("requires_forward_evidence"):
        arithmetic = []
        for row in rows:
            cursor = row["start"]
            for start, end, precision in row["ba_spans"]:
                if (
                    start != cursor
                    or not start < end <= row["end"]
                    or precision not in ("bf16", "fp32")
                ):
                    raise ValueError("invalid recorded forward BA precision span")
                arithmetic.append((start, end, precision))
                cursor = end
            if cursor != row["end"]:
                raise ValueError("missing recorded forward BA precision")
    else:
        precision = ba_precision(
            modeled, max(row["dispatch_count"] for row in rows), logical["verify"]
        )
        arithmetic = [(start, start + count, precision) for start, count in queries]
    work = add()
    for start, end, precision in arithmetic:
        from .qwen_fp8_roofline import Span

        component = component_function(Span(start, end), ba_precision=precision)
        work = add(
            work,
            {
                name: getattr(component, name)
                for name in ("fp8_flops", "bf16_flops", "fp32_flops")
            },
        )
    work = add(work, gdn_arithmetic(queries, logical["verify"]))
    zero_taps = sum(
        2 * (3 - position)
        for start, count in queries
        for position in range(start, min(start + count, 3))
    )
    work["fp32_flops"] -= zero_taps * 10240 * 48
    head = sum(row["head_count"] for row in rows)
    context = sum(row["context_count"] for row in rows)
    normalized = (
        useful if logical["mtp"] else len({(row["id"], row["end"] - 1) for row in rows})
    )
    work = add(
        work,
        dict(
            bf16_flops=2 * head * 248320 * 5120 + 2 * context * 5120 * 10240,
            fp8_flops=2 * context * 2048 * 5120,
            fp32_flops=useful * 64 * (5 * 5120 + 1 + 2 * 17408)
            + useful * 64 * (5 * 5120 + 1)
            + useful * 16 * (6144 + 28 * (4 * 256 + 1 + 3 * 64))
            + useful * 48 * (32 * 3 * 128 + 48 * (6 * 128 + 1) + 10240)
            - useful * (5120 + 16 * 24)
            + normalized * (5 * 5120 + 1)
            + context * (2 * (4 * 5120 + 1) + 4 * 5120 + 1),
        ),
    )
    work["hbm_bytes"] = 0
    precisions = {value[2] for value in arithmetic}
    precision = next(iter(precisions)) if len(precisions) == 1 else "mixed"
    return work, weight_objects(head > 0, context > 0), precision


def resident_allowance(device):
    sms = integer(device.get("multiprocessor_count"), "device multiprocessor count", 1)
    l2 = integer(device.get("l2_cache_bytes"), "device L2 bytes", 1)
    if "B200" not in device.get("name", "") or sms > 160 or l2 > 128 * 1024**2:
        raise ValueError("device exceeds the documented B200 resident profile")
    return FAST_MEMORY_BYTES


def resource_reference(passes, component_function, device):
    allowance = resident_allowance(device)
    total, union, details, memory = add(), {}, [], 0
    for logical in passes:
        work, objects, precision = pass_resources(logical, component_function)
        reads = max(0, sum(objects.values()) - allowance)
        memory += reads
        total = add(total, work)
        union.update(objects)
        details.append(
            dict(
                phase=logical["phase"],
                work=work,
                ba_precision=precision,
                read_union_bytes=sum(objects.values()),
                hbm_lower_bytes=reads,
            )
        )
    total["hbm_bytes"] = max(memory, max(0, sum(union.values()) - allowance))
    return dict(
        seconds=reference_seconds(total),
        work=total,
        passes=details,
        resident_allowance_bytes=allowance,
        capacity_sources=list(CAPACITY_SOURCES),
        exclusions=list(EXCLUSIONS),
        nccl_bytes=0,
        scope="known-resource-conservative-floor-no-unknown-instruction-credit",
    )
