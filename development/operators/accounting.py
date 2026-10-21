from dataclasses import dataclass, replace

import numpy as np

from development.serving.accounting import RATES, linear
from development.serving.kernels import (
    Group,
    context_groups,
    draft_groups,
    fa_groups,
    ffn_groups,
    gdn_groups,
    group,
    input_credit,
    normalization,
    repeated,
    resources,
    sample_groups,
    serial_seconds,
)

VERSION = "operator-reference-v3"
CACHE_CAPACITY = 128 * 1024 * 1024
VOCABULARY = 248320


@dataclass(frozen=True)
class ResourceWork:
    unique_read_bytes: int
    arithmetic: dict
    groups: list[Group]

    @property
    def resources(self):
        return {
            **self.arithmetic,
            "hbm_bytes": max(0, self.unique_read_bytes - CACHE_CAPACITY),
        }


def queries(shape):
    return [(shape["context"] - shape["queries"], shape["queries"])] * shape["batch"]


def chain_groups(shape):
    result = []
    credited = {}
    for stage in range(4):
        current = queries({**shape, "context": shape["context"] + stage})
        for item in draft_groups(current, stage, writes=False):
            # Later drafts reuse matrices and history; new KV has no read credit.
            key = item.name.replace("fa.input-fused", "fa.input")
            memory = item.work["hbm_bytes"]
            if key == "fa.attention":
                memory = max(0, shape["context"] - 1) * shape["batch"] * 4096
            new_bytes = max(0, memory - credited.get(key, 0))
            credited[key] = max(memory, credited.get(key, 0))
            result.append(replace(item, work=dict(item.work, hbm_bytes=new_bytes)))
    return result


def acceptance_groups(case):
    batch, category = case.shape["batch"], case.options["acceptance"]
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
    result = [group("mtp.accept.ratio", dict(fp32_flops=attempts))]
    if case.options["target"] != "greedy":
        residuals = (
            sum(value < 4 for value in accepted) if category != "inactive" else 0
        )
        result.extend(
            [
                group(
                    "mtp.accept.mass",
                    dict(
                        hbm_bytes=4 * (batch + residuals) * VOCABULARY,
                        fp32_flops=(batch + residuals) * VOCABULARY,
                    ),
                ),
                group("mtp.accept.select", dict(fp64_flops=batch)),
            ]
        )
    return result


def position_bytes(height, width):
    counts = []
    for size in (height, width):
        first = (
            np.arange(size, dtype=np.float32) * (np.float32(47) / np.float32(size - 1))
        ).astype(np.int32)
        counts.append(len(set(first) | set(np.minimum(first + 1, 47))))
    return counts[0] * counts[1] * 1152 * 2


def vision_groups(case):
    shape = case.shape
    rows, tokens = shape["rows"], shape["height"] * shape["width"]
    normalized = dict(hbm_bytes=4 * 1152, fp32_flops=rows * (7 * 1152 + 2))
    block = [
        group("vision.norm", normalized),
        group("vision.qkv", linear(rows, 3456, 1152)),
        group(
            "vision.rotary", dict(hbm_bytes=2 * 3456, fp32_flops=rows * 2 * 1152 * 3)
        ),
        group(
            "vision.attention",
            dict(
                bf16_flops=4 * rows * tokens * 1152, fp32_flops=rows * 3 * tokens * 16
            ),
        ),
        group("vision.output", linear(rows, 1152, 1152)),
        group(
            "vision.residual-norm",
            normalized,
            dict(hbm_bytes=2 * 1152, fp32_flops=rows * 1152),
        ),
        group("vision.upper", linear(rows, 4304, 1152), dict(hbm_bytes=2 * 4304)),
        group("vision.down", linear(rows, 1152, 4304)),
        group("vision.residual", dict(hbm_bytes=2 * 1152, fp32_flops=rows * 1152)),
    ]
    result = repeated(block, len(case.weights))
    if case.boundary == "vision.input":
        result = [
            group(
                "vision.pixels",
                dict(
                    hbm_bytes=shape["frames"] * tokens * 16 * 16 * 3,
                    fp32_flops=rows * 1536 * 2,
                ),
            ),
            group("vision.patch", linear(rows, 1152, 1536)),
            group(
                "vision.position-norm",
                normalized,
                dict(
                    hbm_bytes=position_bytes(shape["height"], shape["width"])
                    + 1152 * 2,
                    fp32_flops=rows * 1152,
                ),
            ),
            *block[1:],
            *repeated(block, len(case.weights) - 1),
        ]
    else:
        result = input_credit(result, rows * 1152 * 2)
    if case.boundary == "vision.merge":
        result.extend(
            [
                group("vision.merge.norm", normalized),
                group("vision.merge.upper", linear(rows // 4, 4608, 4608)),
                group("vision.merge.activation", dict(hbm_bytes=2 * 4608)),
                group("vision.merge.down", linear(rows // 4, 5120, 4608)),
                group("vision.merge.bias", dict(hbm_bytes=2 * 5120)),
            ]
        )
    return result


def kernel_groups(case):
    boundary, shape = case.boundary, case.shape
    rows = shape.get("rows", 0)
    if boundary == "fa":
        result = repeated(fa_groups(queries(shape)), len(case.weights))
    elif boundary == "gdn":
        result = repeated(
            gdn_groups(queries(shape), case.options["verify"]), len(case.weights)
        )
        if 0 in case.weights:
            result.insert(
                0,
                group(
                    "gdn.embedding", dict(hbm_bytes=min(rows, VOCABULARY) * 5120 * 2)
                ),
            )
    elif boundary == "ffn":
        result = repeated(ffn_groups(rows), len(case.weights))
    elif boundary == "output.hidden":
        result = [group("output.norm", normalization(rows))]
    elif boundary in {"output.distribution", "output.sample"}:
        result = sample_groups(rows)
    elif boundary == "mtp.context":
        result = context_groups(
            rows, writes=False, embedding_bytes=min(rows, VOCABULARY) * 5120 * 2
        )
    elif boundary == "mtp.draft":
        full = case.options["variant"] == "full"
        result = draft_groups(
            queries(shape),
            int(full),
            writes=False,
            embedding_bytes=rows * 5120 * 2 if full else 0,
        )
    elif boundary == "mtp.chain":
        result = chain_groups(shape)
    elif boundary == "mtp.accept":
        return acceptance_groups(case)
    elif boundary.startswith("vision."):
        return vision_groups(case)
    else:
        raise ValueError("unknown operator boundary")
    input_bytes = (
        rows
        * 5120
        * (4 if boundary == "ffn" else 2 if boundary.startswith("mtp.") else 6)
    )
    return input_credit(result, input_bytes)


def work(case):
    groups = kernel_groups(case)
    total = resources(groups)
    return ResourceWork(
        int(total["hbm_bytes"]),
        {key: total[key] for key in RATES if key != "hbm_bytes"},
        groups,
    )


def lower_bound(case, device_spec):
    if (
        "B200" not in device_spec["name"]
        or not 0 < device_spec["l2_bytes"] <= CACHE_CAPACITY
    ):
        raise ValueError("device differs from the fixed cache policy")
    result = serial_seconds(kernel_groups(case), cache_bytes=CACHE_CAPACITY)
    if result <= 0:
        raise ValueError("zero operator theoretical lower bound")
    return result
