from dataclasses import dataclass

from development.serving.accounting import (
    RATES,
    add,
    ffn,
    gdn_math,
    linear,
    norm,
    reference_seconds,
    scale,
)

VERSION = "operator-reference-v1"
CACHE_CAPACITY = 128 * 1024 * 1024
VOCABULARY = 248320


@dataclass(frozen=True)
class ResourceWork:
    unique_read_bytes: int
    arithmetic: dict

    @property
    def resources(self):
        return {
            **self.arithmetic,
            "hbm_bytes": max(0, self.unique_read_bytes - CACHE_CAPACITY),
        }


def attention(rows, batch, queries, context, *, first=0, cached=False):
    start = context - queries
    pairs = batch * (queries * (start - first) + queries * (queries + 1) // 2)
    history = batch * max(0, start - first + (queries if cached else 0))
    product = 2 * pairs * 24 * 256
    products, cache_bytes = {"bf16_flops": 2 * product}, 4096
    return add(
        {"hbm_bytes": history * cache_bytes},
        products,
        {"fp32_flops": 3 * pairs * 24},
    )


def fa_work(shape, *, variant="full", first=0):
    rows, batch, queries, context = (
        shape[name] for name in ("rows", "batch", "queries", "context")
    )
    heads = 28 if variant == "full" else 24
    return add(
        linear(rows, 14336 if variant == "full" else 12288, 5120, fp8=True),
        linear(rows, 5120, 6144, fp8=True),
        norm(rows, 5120),
        attention(
            rows, batch, queries, context, first=first, cached=variant == "query_only"
        ),
        dict(
            hbm_bytes=4 * (512 if variant == "full" else 256),
            fp32_flops=rows * (5120 + 6144 + heads * (4 * 256 + 2) + heads * 64 * 3),
        ),
    )


def gdn_work(shape, verify):
    rows, batch, queries, context = (
        shape[name] for name in ("rows", "batch", "queries", "context")
    )
    start = context - queries
    return add(
        linear(rows, 16384, 5120, fp8=True),
        linear(rows, 96, 5120),
        linear(rows, 5120, 6144, fp8=True),
        norm(rows, 5120),
        gdn_math([(start, queries)] * batch, verify),
        dict(
            hbm_bytes=(batch if start else 0) * (48 * 128 * 128 * 4 + 10240 * 3 * 2)
            + 10240 * 4 * 2
            + 4 * (128 + 96),
            fp32_flops=rows * (5120 + 7 * 10240),
        ),
    )


def output_work(rows):
    return add(
        linear(rows, VOCABULARY, 5120), norm(rows, 5120), dict(fp32_flops=rows * 5120)
    )


def context_work(rows):
    return add(
        linear(rows, 5120, 10240),
        linear(rows, 2048, 5120, fp8=True),
        scale(norm(rows, 5120), 3),
        norm(rows * 4, 256),
        dict(hbm_bytes=min(rows, VOCABULARY) * 5120 * 2, fp32_flops=rows * 4 * 64 * 3),
    )


def draft_work(shape, full):
    rows = shape["rows"]
    return add(
        fa_work(shape, variant="full" if full else "query_only", first=1),
        ffn(rows),
        output_work(rows),
        add(
            linear(rows, 5120, 10240),
            scale(norm(rows, 5120), 2),
            dict(hbm_bytes=rows * 5120 * 2),
        )
        if full
        else add(),
    )


def chain_work(shape):
    stages = [
        draft_work({**shape, "context": shape["context"] + index}, index != 0)
        for index in range(4)
    ]
    total = add(*stages)
    total["hbm_bytes"] = stages[1]["hbm_bytes"] - shape["rows"] * 5120 * 2
    return total


def acceptance_work(case):
    batch = case.shape["batch"]
    category = case.options["acceptance"]
    accepted = {
        "all": [4] * batch,
        "first-rejection": [0] * batch,
        "partial": [request + 1 for request in range(batch)],
        "mixed": list(range(batch)),
        "inactive": list(range(batch)),
    }[category]
    attempts = sum(
        value + int(value < 4 and category != "inactive") for value in accepted
    )
    residuals = sum(value < 4 for value in accepted) if category != "inactive" else 0
    greedy = case.options["target"] == "greedy"
    return add(
        dict(
            fp32_flops=attempts + (0 if greedy else (batch + residuals) * VOCABULARY),
            fp64_flops=0 if greedy else batch,
            hbm_bytes=0 if greedy else 4 * (batch + residuals) * VOCABULARY,
        )
    )


def vision_block(rows, tokens):
    return add(
        linear(rows, 3456, 1152),
        linear(rows, 1152, 1152),
        linear(rows, 4304, 1152),
        linear(rows, 1152, 4304),
        dict(
            hbm_bytes=2 * (4 * 1152 + 3456 + 1152 + 4304 + 1152),
            bf16_flops=4 * rows * tokens * 1152,
            fp32_flops=rows
            * (2 * (7 * 1152 + 2) + 2 * 1152 + 2 * 1152 * 3 + 3 * tokens * 16),
        ),
    )


def vision_work(case):
    shape = case.shape
    rows, tokens = shape["rows"], shape["height"] * shape["width"]
    total = scale(vision_block(rows, tokens), len(case.weights))
    if case.boundary == "vision.input":
        total = add(
            total,
            linear(rows, 1152, 1536),
            dict(
                hbm_bytes=2304 * 1152 * 2
                + 1152 * 2
                + shape["frames"] * tokens * 16 * 16 * 3,
                fp32_flops=rows * (1536 * 2 + 1152),
            ),
        )
    else:
        total = add(total, dict(hbm_bytes=rows * 1152 * 2))
    if case.boundary == "vision.merge":
        total = add(
            total,
            linear(rows // 4, 4608, 4608),
            linear(rows // 4, 5120, 4608),
            dict(
                hbm_bytes=2 * (2 * 1152 + 4608 + 5120), fp32_flops=rows * (7 * 1152 + 2)
            ),
        )
    return total


def work(case):
    boundary, shape = case.boundary, case.shape
    rows = shape.get("rows", 0)
    if boundary == "fa":
        total = scale(fa_work(shape), len(case.weights))
    elif boundary == "gdn":
        total = scale(gdn_work(shape, case.options["verify"]), len(case.weights))
        if 0 in case.weights:
            total = add(total, dict(hbm_bytes=min(rows, VOCABULARY) * 5120 * 2))
    elif boundary == "ffn":
        total = scale(ffn(rows), len(case.weights))
    elif boundary == "output.hidden":
        total = add(norm(rows, 5120), dict(fp32_flops=rows * 5120))
    elif boundary in {"output.distribution", "output.sample"}:
        total = output_work(rows)
    elif boundary == "mtp.context":
        total = context_work(rows)
    elif boundary == "mtp.draft":
        total = draft_work(shape, case.options["variant"] == "full")
    elif boundary == "mtp.chain":
        total = chain_work(shape)
    elif boundary == "mtp.accept":
        total = acceptance_work(case)
    elif boundary.startswith("vision."):
        total = vision_work(case)
    else:
        raise ValueError("unknown operator boundary")
    if boundary in {
        "fa",
        "gdn",
        "ffn",
        "output.hidden",
        "output.distribution",
        "output.sample",
    }:
        total = add(
            total, dict(hbm_bytes=rows * 5120 * (4 if boundary == "ffn" else 6))
        )
    elif boundary in {"mtp.context", "mtp.draft", "mtp.chain"}:
        total = add(total, dict(hbm_bytes=rows * 5120 * 2))
    return ResourceWork(
        int(total["hbm_bytes"]),
        {key: total[key] for key in RATES if key != "hbm_bytes"},
    )


def lower_bound(case, device_spec):
    if (
        "B200" not in device_spec["name"]
        or not 0 < device_spec["l2_bytes"] <= CACHE_CAPACITY
    ):
        raise ValueError("device differs from the fixed cache policy")
    result = reference_seconds(work(case).resources)
    if result <= 0:
        raise ValueError("zero operator theoretical lower bound")
    return result
