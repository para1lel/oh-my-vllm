"""Check the frozen HC boundary and one channel-addressing transformation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    HYPERCONNECTION_SHA256,
    load_hyperconnection,
)
from development.model.deepseek_v41.layout import tensor_layout
from development.model.deepseek_v41.weights import Weights as CheckpointWeights
from development.serving.device import device_identity, selected_uuid

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
)
from .check_deepseek_exact import (
    source_identity as common_sources,
)
from .fixtures import Fixture
from .numerics import compare
from .storage import decode, equal_bits, storage_dtype


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    paths = list(
        (repository / "python/oh_my_vllm/operators/hyperconnection").rglob("*.py")
    )
    paths.extend(
        repository / "development/model/deepseek_v41" / name
        for name in ("weights.py", "layout.py", "identity.py", "quantization.py")
    )
    for path in paths:
        sources[str(path.relative_to(repository))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_contracts(device, parameters):
    from oh_my_vllm.operators.hyperconnection.api import Inputs, Weights
    from oh_my_vllm.operators.hyperconnection.plan import make_plan
    from oh_my_vllm.operators.hyperconnection.workspace import prepare

    weights = Weights(*(device.upload(value) for value in parameters))
    workspace = prepare(make_plan(rows=1), device, weights)
    residual = device.zeros((1, 4, 5120), "bfloat16")
    incoming = device.zeros((1, 4), "float32")
    branch = device.zeros((1, 5120), "bfloat16")
    try:
        for invalid in (
            Inputs(device.empty((1, 4, 5120), "float32"), incoming, branch),
            Inputs(residual, device.empty((1, 3), "float32"), branch),
            Inputs(residual, incoming, device.empty((1, 5120), "float32")),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        rejected(
            lambda: workspace.module.hc_expand(
                residual,
                branch,
                workspace.storage.post,
                workspace.storage.comb,
                residual,
            )
        )
    finally:
        device.synchronize()
        workspace.close()
    for count in (-1, True, 385):
        rejected(lambda count=count: make_plan(rows=count))
    empty = prepare(make_plan(rows=0), device, weights)
    inputs = Inputs(
        device.empty((0, 4, 5120), "bfloat16"),
        device.empty((0, 4), "float32"),
        device.empty((0, 5120), "bfloat16"),
    )
    try:
        execute_modes(
            device,
            (empty,),
            lambda: empty.run(inputs),
            lambda: require(
                all(tensor.shape[0] == 0 for tensor in empty.outputs),
                "empty HC changed shape",
            ),
        )
    finally:
        device.synchronize()
        empty.close()
    rejected(lambda: empty.run(inputs))
    return {"invalid_arguments": 7, "empty_eager_graph": True, "closed_workspace": True}


def check_case(device, case, limits, parameters, *, permute=False):
    from oh_my_vllm.operators.hyperconnection.api import Inputs, Weights
    from oh_my_vllm.operators.hyperconnection.plan import make_plan
    from oh_my_vllm.operators.hyperconnection.workspace import prepare

    inputs, expected = dict(case.inputs), dict(case.expected)
    parameters = [value.copy() for value in parameters]
    if permute:
        permutation = np.arange(5120) * 73 % 5120
        inputs["residual"] = inputs["residual"][:, :, permutation]
        inputs["branch"] = inputs["branch"][:, permutation]
        expected["collapsed"] = expected["collapsed"][:, permutation]
        expected["expanded"] = expected["expanded"][:, :, permutation]
        parameters[0] = (
            parameters[0].reshape(24, 4, 5120)[:, :, permutation].reshape(24, 20480)
        )
    fixture = Fixture(case.manifest, inputs, {}, expected)
    weights = Weights(*(device.upload(value) for value in parameters))
    values = Inputs(
        device.upload(inputs["residual"], "bfloat16"),
        device.upload(inputs["incoming_pre"]),
        device.upload(inputs["branch"], "bfloat16"),
    )
    workspace = prepare(make_plan(rows=len(inputs["residual"])), device, weights)
    originals = [
        (tensor, value)
        for tensor, value in zip(
            (*values, *weights),
            (inputs["residual"], inputs["incoming_pre"], inputs["branch"], *parameters),
            strict=True,
        )
    ]
    result = {}
    try:

        def operation():
            workspace.run(values)

        def check():
            actual = {
                name: decode(
                    device.download(tensor, storage_dtype(str(tensor.dtype))),
                    str(tensor.dtype),
                )
                for name, tensor in workspace.outputs._asdict().items()
            }
            compared = compare(actual, fixture, limits)
            require(compared.passed, f"HC boundary differs: {compared.failures}")
            for tensor, original in originals:
                require(
                    equal_bits(device.download(tensor, original.dtype), original),
                    "HC changed an input or weight tensor",
                )
            result.update(metrics=compared.metrics)

        execute_modes(device, (workspace,), operation, check)
        result.update(
            case=case.manifest["case"]["id"],
            channel_permutation={"multiplier": 73, "modulus": 5120}
            if permute
            else None,
            native=native_identity(workspace),
            modes=["eager", "graph"],
        )
    finally:
        device.synchronize()
        workspace.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.hyperconnection") is not None,
        "native HC boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    manifest = json.loads(raw)
    config = json.loads((args.model / "config.json").read_text())
    weights = CheckpointWeights(args.model, manifest, tensor_layout(config))
    metadata, cases = load_hyperconnection()
    names = cases["decode"].manifest["case"]["weights"]
    parameters = [weights.tensor(name).astype(np.float32) for name in names]
    identity = device_identity(selected_uuid())
    device = Device()
    try:
        with caller_stream(device):
            checks = [
                check_case(device, case, metadata["cases"][name]["limits"], parameters)
                for name, case in cases.items()
            ]
            checks.append(
                check_case(
                    device,
                    cases["decode"],
                    metadata["cases"]["decode"]["limits"],
                    parameters,
                    permute=True,
                )
            )
            contracts = check_contracts(device, parameters)
        weights._verify_states()
        require(
            source_identity() == sources, "HC checker sources changed during execution"
        )
    finally:
        device.synchronize()
        device.close()
    report = {
        "passed": True,
        "device": identity,
        "fixture_sha256": HYPERCONNECTION_SHA256,
        "checks": checks,
        "contracts": contracts,
        "sources": sources,
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
