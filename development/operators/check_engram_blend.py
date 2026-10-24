"""Check frozen Engram blending and the model's pinned-lookup/projection chain."""

import argparse
import hashlib
import importlib.util
import json
import weakref
from contextlib import closing
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.model.deepseek_v41 import fixtures
from development.serving.device import device_identity, selected_uuid

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
)
from .check_deepseek_exact import (
    source_identity as operator_sources,
)
from .numerics import compare
from .performance_inputs import bf16
from .storage import decode, equal_bits


def source_identity():
    result = operator_sources()
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "python/oh_my_vllm/models/deepseek_v41").glob("*.py"))
    for family in ("projection", "quantization", "engram", "runtime"):
        paths.extend(
            path
            for path in (root / "python/oh_my_vllm/operators" / family).rglob("*")
            if path.suffix in (".py", ".cu", ".cuh", ".cpp", ".hpp")
        )
    result.update(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )
    return result


def composition_weights(device):
    from oh_my_vllm.models.deepseek_v41.placement import RankPlan
    from oh_my_vllm.models.deepseek_v41.weights import RankWeights
    from oh_my_vllm.operators.projection.api import Weights
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    plan = RankPlan(MappingProxyType({}), MappingProxyType({}), 0, 0, "TP4", 0)
    owner = RankWeights(device, plan, 1 << 30, 0)
    raw = np.zeros((25600, 6144), np.uint8)
    for copy in range(4):
        raw[copy * 5120 : (copy + 1) * 5120, copy * 256] = (
            0x38 if copy % 2 == 0 else 0xB8
        )
    for head in range(24):
        raw[20480:, head * 256] = 0x38 + 8 * (head % 4)
    scales = np.full((800, 192), 127, np.uint8)
    packed = prepare_weights(
        device, Weights(device.upload(raw), device.upload(scales)), bits=8
    )
    for layer in (1, 14):
        prefix = f"layers.{layer}.engram"
        owner.packed[prefix + ".wkv.weight"] = packed
        for name, weights in (
            ("q_weight", [1, 1, 0.25, -0.5]),
            ("k_weight", [1, 4, -1, 2]),
        ):
            owner.tensors[prefix + "." + name] = device.upload(
                bf16(np.broadcast_to(np.array(weights)[:, None], (4, 5120)).copy()),
                "bfloat16",
            )
    return owner


def composition_table():
    from oh_my_vllm.operators.engram import lookup
    from oh_my_vllm.operators.engram.shared import SharedWeightFile

    # Integer rows and binary weights give exact sums, then the BF16 boundary rounds.
    codes = np.array([0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E, 0x50, 0x51], np.uint8)
    payload = np.repeat(codes[:, None], 256, axis=1).tobytes() + bytes([127]) * (9 * 8)
    with SharedWeightFile(len(payload), limit=4096) as shared:
        shared.append(payload)
        shared.seal()
        return lookup.register(shared, limit=4096)


def check_composition(device, owner, rows, layer, table_owner):
    from oh_my_vllm.models.deepseek_v41.engram import Table
    from oh_my_vllm.models.deepseek_v41.engram_runtime import prepare

    table = table_owner[0]
    layout = Table(9, 0, 9 * 256)
    workspace = prepare(owner, layer, table=table, layout=layout, rows=rows)
    ids = (np.arange(rows * 24, dtype=np.int64).reshape(rows, 24) * 5 + 3) % 9
    residual_host = np.broadcast_to(
        np.array([1, 2, 4, 8], np.float32)[None, :, None], (rows, 4, 5120)
    ).copy()
    mask_host = (np.arange(rows) % 3 != 1).astype(np.uint8)
    values = (ids + 1).astype(np.float64)
    sums = np.sum(values * (2 ** (np.arange(24) % 4))[None, :], axis=1)
    require(
        np.array_equal(sums, np.array([435, 444, 471])[np.arange(rows) % 3]),
        "independent Engram projection sums differ",
    )
    projected = np.array([436, 444, 472], np.float64)[np.arange(rows) % 3]
    # Constant nonzero keys/residuals cancel their magnitudes under per-copy RMS.
    dot = np.sqrt(5120) * np.array([1, -4, -0.25, 1], np.float64)
    gate = 1 / (1 + np.exp(-np.copysign(np.sqrt(np.abs(dot)), dot)))
    expected = bf16(
        residual_host
        + mask_host[:, None, None] * gate[None, :, None] * projected[:, None, None]
    )
    indices, mask = device.upload(ids), device.upload(mask_host)
    residual = device.upload(bf16(residual_host), "bfloat16")

    def operation():
        return workspace.run(residual, indices, mask)

    def check(invalid=False):
        flags = [device.download(value, np.uint8) for value in workspace.outputs.status]
        if invalid:
            require(
                flags[0][0, 0] == 1,
                "lookup failure was not exposed by model composition",
            )
            return
        require(
            all(not value.any() for value in flags),
            "Engram composition returned status",
        )
        actual = device.download(workspace.outputs.value, np.uint16)
        require(
            equal_bits(actual, expected),
            "Engram lookup/projection/blend endpoint differs",
        )
        looked_up = device.download(workspace.lookup.outputs.value, np.uint16)
        require(
            equal_bits(
                looked_up, bf16(np.broadcast_to(values[..., None], (rows, 24, 256)))
            ),
            "lookup head ordering differs",
        )
        projected_actual = decode(
            device.download(workspace.projection.outputs.value, np.uint16), "bfloat16"
        )
        require(
            np.array_equal(
                projected_actual[:, :20480].reshape(rows, 4, 5120),
                np.broadcast_to(
                    (values[:, :4] * np.array([1, -1, 1, -1]))[..., None],
                    (rows, 4, 5120),
                ),
            )
            and np.array_equal(
                projected_actual[:, 20480:],
                np.broadcast_to(projected[:, None], (rows, 5120)),
            ),
            "Engram flattened lookup projection differs",
        )
        require(
            equal_bits(device.download(residual, np.uint16), bf16(residual_host)),
            "Engram changed caller residual",
        )

    rejected(lambda: prepare(owner, 0, table=table, layout=layout, rows=rows))
    if rows:
        rejected(
            lambda: prepare(
                owner,
                layer,
                table=table,
                layout=layout,
                rows=rows,
                byte_limit=workspace.allocated_bytes - 1,
            )
        )
        rejected(lambda: workspace.run(workspace.outputs.value, indices, mask))
    operation()
    check()
    restore = device.upload(ids)
    bad = ids.copy()
    if rows:
        bad[0, 0] = 9
    invalid = device.upload(bad)
    with workspace.graph_lease(), device.graph(operation) as replay:
        replay()
        check()
        if rows:
            device.module.copy_bytes(indices, invalid)
            replay()
            check(invalid=True)
            device.module.copy_bytes(indices, restore)
            replay()
            check()
        workspace.close()
        if layer == 14:
            owner.close()
            rejected(
                lambda: prepare(owner, layer, table=table, layout=layout, rows=rows)
            )
            table_owner.clear()
        table = None
        replay()
        check()
    rejected(operation)
    require(workspace._allocate is None, "composition retained its allocator")
    return dict(
        rows=rows,
        layer=layer,
        workspace_bytes=workspace.allocated_bytes,
        native=[
            native_identity(value)
            for value in (
                workspace.lookup,
                workspace.projection.quantization,
                workspace.projection.projection,
                workspace.blend,
            )
        ],
    ), weakref.ref(workspace)


def check_case(device, case, limits):
    from oh_my_vllm.operators.engram.blend import Inputs, make_plan, prepare

    uploaded = {
        name: device.upload(value, case.dtype("inputs", name))
        for name, value in case.inputs.items()
    }
    inputs = Inputs(**uploaded)
    workspace = prepare(make_plan(rows=case.manifest["case"]["shape"]["rows"]), device)
    result = {}

    def check():
        bits = device.download(workspace.outputs.value, np.uint16)
        gate = device.download(workspace.outputs.gate, np.float32)
        measured = compare(
            {"gate": gate, "value": decode(bits, "bfloat16")}, case, limits
        )
        require(measured.passed, f"Engram blend differs: {measured.failures}")
        masked = case.inputs["mask"] == 0
        require(not gate[masked].any(), "masked Engram gate is nonzero")
        require(
            equal_bits(bits[masked], case.inputs["residual"][masked]),
            "masked Engram residual changed",
        )
        for name, original in case.inputs.items():
            require(
                equal_bits(device.download(uploaded[name], original.dtype), original),
                "Engram blend changed an input",
            )
        result["metrics"] = measured.metrics

    try:
        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check)
        return {
            "case": case.manifest["case"]["id"],
            **result,
            "native": native_identity(workspace),
            "modes": ["eager", "graph"],
        }
    finally:
        device.synchronize()
        workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.engram.blend import make_plan

    for rows in (-1, True, 385, 1.5):
        rejected(lambda rows=rows: make_plan(rows=rows))
    for rows in (0, 1):
        check_contract(device, rows)
    return {"empty_eager_graph": True, "invalid_inputs": True, "closed_workspace": True}


def check_contract(device, rows):
    from oh_my_vllm.operators.engram.blend import Inputs, make_plan, prepare

    inputs = Inputs(
        device.zeros((rows, 4, 5120), "bfloat16"),
        device.zeros((rows, 5, 5120), "bfloat16"),
        device.zeros((4, 5120), "bfloat16"),
        device.zeros((4, 5120), "bfloat16"),
        device.zeros((rows,), "uint8"),
    )
    workspace = prepare(make_plan(rows=rows), device)
    try:
        if rows:
            for field, shape, dtype in (
                ("residual", (1, 4, 5120), "float32"),
                ("projected", (1, 4, 5120), "bfloat16"),
                ("q_weight", (5120,), "bfloat16"),
                ("k_weight", (4, 5120), "float32"),
                ("mask", (1,), "int32"),
            ):
                invalid = inputs._replace(**{field: device.empty(shape, dtype)})
                rejected(lambda invalid=invalid: workspace.run(invalid))
            rejected(
                lambda: workspace.module.engram_blend(
                    *inputs, workspace.outputs.gate, inputs.residual
                )
            )
        execute_modes(
            device,
            (workspace,),
            lambda: workspace.run(inputs),
            lambda: require(
                not device.download(workspace.outputs.gate, np.float32).any()
                and not device.download(workspace.outputs.value, np.uint16).any(),
                "empty or masked Engram blend differs",
            ),
        )
        with (
            workspace.graph_lease(),
            device.graph(lambda: workspace.run(inputs)) as replay,
        ):
            workspace.close()
            replay()
            device.synchronize()
    finally:
        device.synchronize()
        workspace.close()
    rejected(lambda: workspace.run(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.engram.blend") is not None,
        "native Engram blend is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    metadata, cases = fixtures.load_engram_blend()
    sources = source_identity()
    report = {
        "fixture_sha256": fixtures.ENGRAM_BLEND_SHA256,
        "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
        "scope": (
            "Frozen Engram blending and synthetic lookup/projection composition; "
            "no model or performance acceptance."
        ),
        "device": device_identity(selected_uuid()),
        "sources": sources,
    }
    with closing(Device()) as device, caller_stream(device):
        from oh_my_vllm.operators.engram.lookup import pinned_bytes

        report["device_native"] = native_identity(device)
        before = pinned_bytes()
        table_owner = [composition_table()]
        owner = None
        report["composition"] = []
        try:
            report["contracts"] = check_contracts(device)
            report["cases"] = [
                check_case(device, case, metadata["cases"][name]["limits"])
                for name, case in cases.items()
            ]
            owner = composition_weights(device)
            for rows, layer in ((0, 1), (2, 1), (384, 14)):
                result, reference = check_composition(
                    device, owner, rows, layer, table_owner
                )
                require(
                    reference() is None, "Engram model workspace requires cyclic GC"
                )
                require(
                    pinned_bytes() == before + (4096 if table_owner else 0),
                    "Engram composition shared-table charge differs",
                )
                report["composition"].append(result)
        finally:
            if owner is not None:
                owner.close()
            table_owner.clear()
    require(source_identity() == sources, "Engram blend sources changed")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
