"""Check local MoE composition, not model acceptance or collective execution."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.operators.check_deepseek_exact import (
    caller_stream,
    native_identity,
    rejected,
    require,
)
from development.operators.performance_inputs import bf16
from development.operators.storage import decode
from development.serving.device import device_identity, selected_uuid


def source_identity():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), root / "python/oh_my_vllm/models/deepseek_v41/moe.py"]
    for family in (
        "activation",
        "linear",
        "projection",
        "quantization",
        "routing",
        "runtime",
    ):
        paths.extend((root / "python/oh_my_vllm/operators" / family).rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def weight(device, bits, outputs, width, code, scale, sparse):
    from oh_my_vllm.operators.projection.api import Weights
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    raw = np.zeros((outputs, width * bits // 8), np.uint8)
    if sparse:
        raw[:, 0] = code
    else:
        raw.fill(code | (code << 4) if bits == 4 else code)
    scales = np.full(
        (outputs if bits == 4 else outputs // 32, width // 32), scale, np.uint8
    )
    return prepare_weights(
        device, Weights(device.upload(raw), device.upload(scales)), bits=bits
    )


def owner_weights(device, strategy, rank, draft):
    from oh_my_vllm.models.deepseek_v41.placement import RankPlan
    from oh_my_vllm.models.deepseek_v41.weights import RankWeights

    experts = 128 if draft else 384
    name = "mtp.0" if draft else "layers.0"
    prefix = f"{name}.ffn"
    width = 576 if strategy == "TP4" else 2304
    shared_width = 2304 if strategy == "DEP4" else 576
    plan = RankPlan(MappingProxyType({}), MappingProxyType({}), 0, 0, strategy, rank)
    owner = RankWeights(device, plan, 1 << 30, 0)
    owner.tensors[f"{prefix}.gate.weight"] = device.upload(
        bf16(np.zeros((experts, 5120), np.float32)), "bfloat16"
    )
    selected = (
        ([0, 31, 127], [32, 63, 64])
        if draft
        else ([0, 1, 95, 96, 191, 383], [2, 3, 94, 97, 190, 382])
    )
    for suffix, ids in zip(("bias", "bias_vl"), selected, strict=True):
        bias = np.zeros(experts, np.float32)
        bias[ids] = np.arange(len(ids), 0, -1, dtype=np.float32)
        owner.tensors[f"{prefix}.gate.{suffix}"] = device.upload(bias)
    count = experts if strategy == "TP4" else experts // 4
    start = 0 if strategy == "TP4" else rank * count
    # Exact dyadic weights: w1*x=16*x[0], w3*x=x[0], w2 sums channels.
    routed = {
        "w1": weight(device, 4, width, 5120, 6, 129, True),
        "w3": weight(device, 4, width, 5120, 2, 127, True),
        "w2": weight(device, 4, 5120, width, 2, 127, False),
    }
    odd_down = weight(device, 4, 5120, width, 4, 127, False)
    for expert in range(start, start + count):
        for projection, value in routed.items():
            owner.packed[f"{prefix}.experts.{expert}.{projection}.weight"] = (
                odd_down if projection == "w2" and expert % 2 else value
            )
    for projection, outputs, columns, code, sparse in (
        ("w1", shared_width, 5120, 0x58, True),
        ("w3", shared_width, 5120, 0x38, True),
        ("w2", 5120, shared_width, 0x38, False),
    ):
        owner.packed[f"{prefix}.shared_experts.{projection}.weight"] = weight(
            device, 8, outputs, columns, code, 127, sparse
        )
    return owner, name, selected, start, count, width, shared_width


def check_case(device, strategy, rank, draft, rows):
    from oh_my_vllm.models.deepseek_v41.moe import prepare

    owner, name, selected, start, count, width, shared_width = owner_weights(
        device, strategy, rank, draft
    )
    workspace = None
    try:
        workspace = prepare(owner, name, rows=rows)
        host = np.zeros((rows, 5120), np.float32)
        host[:, 0] = 1 + np.arange(rows) % 2
        mask = (np.arange(rows) % 2).astype(np.uint8)
        value = device.upload(bf16(host), "bfloat16")
        image_mask = device.upload(mask)
        local_weight = np.array(
            [
                sum(
                    1 + expert % 2
                    for expert in selected[m]
                    if start <= expert < start + count
                )
                for m in mask
            ]
        )
        # The clamped gate result rounds to exactly ten in BF16.
        # Top-k probabilities are equal; routes are 1.5/6 or 1.5/3.
        route = 0.5 if draft else 0.25
        routed = (10 * route * host[:, 0] * width * local_weight)[:, None]
        shared = (10 * host[:, 0] * shared_width)[:, None]

        def check(invalid=False):
            flags = [
                device.download(status, np.uint8) for status in workspace.outputs.status
            ]
            if invalid:
                require(
                    flags[0][0, 0] == 1 and flags[1][0] == 1,
                    "invalid MoE input accepted",
                )
                require(
                    not flags[0][1:].any() and not flags[1][1:].any(),
                    "invalid MoE row affected other rows",
                )
            else:
                require(
                    not any(flag.any() for flag in flags), "valid MoE status differs"
                )
            for output, expected, dtype in (
                (workspace.outputs.routed, routed, np.float32),
                (workspace.outputs.shared, shared, np.uint16),
            ):
                actual = device.download(output, dtype)
                if dtype == np.uint16:
                    actual = decode(actual, "bfloat16")
                if invalid:
                    actual, expected = actual[1:], expected[1:]
                require(
                    np.array_equal(actual, np.broadcast_to(expected, actual.shape)),
                    "MoE output differs",
                )

        def operation():
            return workspace.run(value, image_mask)

        operation()
        check()
        if (strategy, draft, rows) == ("TP4", False, 2):
            rejected(lambda: prepare(owner, name, rows=rows, byte_limit=1))
            rejected(lambda: prepare(owner, "layers.40", rows=rows))
            rejected(lambda: workspace.run(workspace.outputs.shared, image_mask))

            def overlapping(shape, dtype):
                if tuple(shape) == (rows, 5120) and dtype == "bfloat16":
                    return value
                return device.empty(shape, dtype)

            rejected(lambda: prepare(owner, name, rows=rows, allocate=overlapping))
            operation()
            check()
        with workspace.graph_lease(), device.graph(operation) as replay:
            replay()
            check()
            if rows:
                invalid = bf16(host)
                invalid[0, 0] = 0x7FC0
                device.module.copy_bytes(value, device.upload(invalid, "bfloat16"))
                replay()
                check(invalid=True)
                device.module.copy_bytes(value, device.upload(bf16(host), "bfloat16"))
                replay()
                check()
            workspace.close()
            owner.close()
            replay()
            check()
        rejected(operation)
        rejected(lambda: prepare(owner, name, rows=rows))
        return dict(
            strategy=strategy,
            rank=rank,
            draft=draft,
            rows=rows,
            workspace_bytes=workspace.allocated_bytes,
            native=[native_identity(part) for part in workspace.parts],
        )
    finally:
        device.synchronize_compute()
        if workspace is not None:
            workspace.close()
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from oh_my_vllm.models.deepseek_v41.moe import prepare  # noqa: F401
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    device = Device()
    try:
        with caller_stream(device):
            checks = [
                check_case(device, strategy, rank, draft, rows)
                for strategy, rank, draft, rows in (
                    ("TP4", 0, False, 2),
                    ("TEP4", 1, False, 2),
                    ("DEP4", 3, False, 2),
                    ("TP4", 0, True, 1),
                    ("TEP4", 1, True, 2),
                    ("DEP4", 2, True, 2),
                    ("TP4", 0, False, 0),
                    ("TEP4", 0, False, 384),
                )
            ]
        require(source_identity() == sources, "MoE sources changed during execution")
        report = dict(
            passed=True,
            device=device_identity(selected_uuid()),
            checks=checks,
            sources=sources,
        )
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    finally:
        device.synchronize_compute()
        device.close()


if __name__ == "__main__":
    main()
