"""Check the isolated Engram gate and residual against frozen FP64 fields."""

import argparse
import importlib.util
import json
from contextlib import closing
from pathlib import Path

import numpy as np

from development.model.deepseek_v41 import fixtures
from development.serving.device import device_identity, selected_uuid

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
    source_identity,
)
from .numerics import compare
from .storage import decode, equal_bits


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
            "Post-projection Engram gate and residual; "
            "no model or performance acceptance."
        ),
        "device": device_identity(selected_uuid()),
        "sources": sources,
    }
    with closing(Device()) as device, caller_stream(device):
        report["device_native"] = native_identity(device)
        report["contracts"] = check_contracts(device)
        report["cases"] = [
            check_case(device, case, metadata["cases"][name]["limits"])
            for name, case in cases.items()
        ]
    require(source_identity() == sources, "Engram blend sources changed")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
