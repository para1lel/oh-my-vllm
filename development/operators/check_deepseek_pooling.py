"""Check isolated ratio2 FP32 pooling before normalization, with BF16 output."""

import argparse
import hashlib
import importlib.util
import json
from contextlib import closing
from pathlib import Path

import numpy as np

from development.model.deepseek_v41 import fixtures
from development.serving.device import device_identity, selected_uuid
from tests.test_collectives import metadata_tensor

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
)
from .check_deepseek_exact import source_identity as common_sources
from .numerics import compare
from .storage import decode, equal_bits


def source_identity():
    root = Path(__file__).resolve().parents[2]
    sources = common_sources()
    paths = [
        *(root / "python/oh_my_vllm/operators/compression").rglob("*.py"),
        root / "tests/test_collectives.py",
    ]
    for path in paths:
        sources[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_case(device, case, limits):
    from oh_my_vllm.operators.compression.api import Inputs
    from oh_my_vllm.operators.compression.plan import make_plan
    from oh_my_vllm.operators.compression.workspace import prepare

    rows = case.manifest["case"]["shape"]["rows"]
    uploaded = {
        key: device.upload(value, "float32") for key, value in case.inputs.items()
    }
    inputs, workspace = Inputs(**uploaded), prepare(make_plan(rows=rows), device)
    result = {}

    def check():
        output = decode(device.download(workspace.outputs.value, np.uint16), "bfloat16")
        measured = compare({"value": output}, case, limits)
        require(measured.passed, f"ratio2 pooling differs: {measured.failures}")
        for key, original in case.inputs.items():
            require(
                equal_bits(device.download(uploaded[key], np.float32), original),
                "ratio2 pooling changed an input",
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
        try:
            device.synchronize()
        finally:
            workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.compression.plan import make_plan

    for rows in (True, -1, 385, 1.0):
        rejected(lambda rows=rows: make_plan(rows=rows))
    for rows in (0, 1):
        check_contract(device, rows)
    return {
        "invalid_arguments": True,
        "empty_eager_graph": True,
        "input_output_overlap": True,
        "graph_after_workspace_close": True,
        "closed_workspace": True,
    }


def check_contract(device, rows):
    from oh_my_vllm.operators.compression.api import Inputs
    from oh_my_vllm.operators.compression.plan import make_plan
    from oh_my_vllm.operators.compression.workspace import prepare

    workspace = prepare(make_plan(rows=rows), device)
    kv, gates = (device.zeros((rows, 2, 512), "float32") for _ in range(2))
    inputs = Inputs(kv, gates)
    try:
        if rows:
            for invalid in (
                Inputs(device.empty((rows, 2, 512), "bfloat16"), gates),
                Inputs(kv, device.empty((rows, 2, 512), "bfloat16")),
                Inputs(kv, device.empty((rows, 1, 512), "float32")),
                Inputs(device.empty((rows, 2, 256), "float32"), gates),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            for value in (kv, gates):
                # Valid BF16 metadata isolates the overlap check.
                alias, guards = metadata_tensor(
                    rows * 512, pointer=value.data_ptr(), code=4, bits=16
                )
                alias = device.view(alias, (rows, 512))
                require(str(alias.dtype) == "bfloat16", "overlap alias dtype differs")
                rejected(
                    lambda alias=alias: workspace.module.pool_ratio2(kv, gates, alias)
                )
                del alias, guards
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(inputs)) as replay,
            ):
                workspace.close()
                replay()
                require(
                    not device.download(workspace.outputs.value, np.uint16).any(),
                    "recorded pooling graph failed after workspace close",
                )
        else:
            execute_modes(
                device,
                (workspace,),
                lambda: workspace.run(inputs),
                lambda: require(
                    workspace.outputs.value.shape == (0, 512),
                    "empty ratio2 pooling changed shape",
                ),
            )
    finally:
        try:
            device.synchronize()
        finally:
            workspace.close()
    rejected(lambda: workspace.run(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.compression") is not None,
        "native ratio2 pooling boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    metadata, cases = fixtures.load_pooling()
    sources = source_identity()
    report = {
        "fixture_sha256": fixtures.POOLING_SHA256,
        "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
        "scope": (
            "Complete FP32 ratio2 KV/gate pairs, final BF16 pooling; "
            "no projection, RMS, RoPE, state assembly, performance or full model."
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
    require(source_identity() == sources, "pooling sources changed during validation")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
