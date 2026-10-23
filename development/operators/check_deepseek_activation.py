"""Check clipped and unclipped SwiGLU with fixed per-request numerical limits."""

import argparse
import hashlib
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
)
from .check_deepseek_exact import source_identity as common_sources
from .numerics import compare
from .storage import decode, equal_bits


def source_identity():
    root = Path(__file__).resolve().parents[2]
    sources = common_sources()
    for path in (root / "python/oh_my_vllm/operators/activation").rglob("*.py"):
        sources[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_case(device, case, limits):
    from oh_my_vllm.operators.activation.api import Inputs
    from oh_my_vllm.operators.activation.plan import make_plan
    from oh_my_vllm.operators.activation.workspace import prepare

    shape, options = (case.manifest["case"][key] for key in ("shape", "options"))
    plan = make_plan(
        rows=shape["rows"],
        width=shape["width"],
        limit=options["limit"],
        routed=options["route"],
    )
    uploaded = {
        key: device.upload(value, "float32" if key == "route" else "bfloat16")
        for key, value in case.inputs.items()
    }
    inputs, workspace = Inputs(**uploaded), prepare(plan, device)
    result = {}

    def check():
        value = decode(device.download(workspace.outputs.value, np.uint16), "bfloat16")
        measured = compare({"value": value}, case, limits)
        require(measured.passed, f"SwiGLU differs: {measured.failures}")
        for key, original in case.inputs.items():
            require(
                equal_bits(device.download(uploaded[key], original.dtype), original),
                "SwiGLU changed its inputs",
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
    from oh_my_vllm.operators.activation.plan import make_plan

    for options in (
        dict(rows=True, width=576),
        dict(rows=2305, width=576),
        dict(rows=1, width=512),
        dict(rows=1, width=2304, limit=0),
        dict(rows=1, width=2816, limit=0, routed=True),
        dict(rows=1, width=576, routed=1),
    ):
        rejected(lambda options=options: make_plan(**options))
    for rows, routed in ((0, False), (0, True), (1, False), (1, True)):
        check_contract(device, rows, routed)
    return {
        "invalid_arguments": True,
        "empty_eager_graph": True,
        "closed_workspace": True,
    }


def check_contract(device, rows, routed):
    from oh_my_vllm.operators.activation.api import Inputs
    from oh_my_vllm.operators.activation.plan import make_plan
    from oh_my_vllm.operators.activation.workspace import prepare

    workspace = prepare(make_plan(rows=rows, width=576, routed=routed), device)
    values = device.zeros((rows, 576), "bfloat16")
    route = device.zeros((rows,), "float32") if routed else None
    inputs = Inputs(values, values, route)
    try:
        if rows:
            for invalid in (
                Inputs(device.empty((rows, 576), "float32"), values, route),
                Inputs(values, device.empty((rows, 2304), "bfloat16"), route),
                Inputs(
                    values,
                    values,
                    None if routed else device.empty((rows,), "float32"),
                ),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            native_route = route if routed else workspace.empty_route
            rejected(
                lambda: workspace.module.swiglu(
                    values, values, native_route, values, 10, routed
                )
            )
            rejected(
                lambda: workspace.module.swiglu(
                    values, values, native_route, workspace.outputs.value, 5, routed
                )
            )
            if routed:
                rejected(
                    lambda: workspace.run(
                        Inputs(values, values, device.empty((rows,), "bfloat16"))
                    )
                )
        else:
            execute_modes(
                device,
                (workspace,),
                lambda: workspace.run(inputs),
                lambda: require(
                    workspace.outputs.value.shape == (0, 576),
                    "empty SwiGLU changed shape",
                ),
            )
    finally:
        device.synchronize()
        workspace.close()
    rejected(lambda: workspace.run(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.activation") is not None,
        "native SwiGLU boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    metadata, cases = fixtures.load_swiglu()
    sources = source_identity()
    report = {
        "fixture_sha256": fixtures.SWIGLU_SHA256,
        "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
        "scope": (
            "BF16 gate/up, optional row route, final BF16 SwiGLU; "
            "no projections or full model."
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
    require(
        source_identity() == sources, "activation sources changed during validation"
    )
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
