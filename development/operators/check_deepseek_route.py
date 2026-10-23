"""Check frozen post-logit routing without a projection or dispatch oracle."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    ROUTING_SHA256,
    load_routing,
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
from .check_deepseek_hc import source_identity as common_sources
from .numerics import compare
from .storage import decode, equal_bits, storage_dtype


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    for path in (repository / "python/oh_my_vllm/operators/routing").rglob("*.py"):
        sources[str(path.relative_to(repository))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_contracts(device):
    from oh_my_vllm.operators.routing.api import Inputs, Weights
    from oh_my_vllm.operators.routing.plan import make_plan
    from oh_my_vllm.operators.routing.workspace import prepare

    weights = Weights(*(device.zeros((384,), "float32") for _ in range(2)))
    workspace = prepare(make_plan(rows=1, experts=384), device, weights)
    logits = device.zeros((1, 384), "float32")
    mask = device.zeros((1,), "uint8")
    try:
        for invalid in (
            Inputs(device.empty((1, 384), "float64"), mask),
            Inputs(device.empty((1, 128), "float32"), mask),
            Inputs(logits, device.empty((1,), "int32")),
            Inputs(logits, device.empty((2,), "uint8")),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        rejected(
            lambda: workspace.module.route(
                logits,
                *weights,
                mask,
                workspace.outputs.expert_ids,
                device.view(logits, (1, 6)),
            )
        )
        rejected(
            lambda: workspace.module.route(
                logits,
                device.empty((384,), "float64"),
                weights.bias_vl,
                mask,
                *workspace.outputs,
            )
        )
    finally:
        device.synchronize()
        workspace.close()
    for rows, experts in ((-1, 384), (True, 384), (385, 384), (1, 256), (1, True)):
        rejected(
            lambda rows=rows, experts=experts: make_plan(rows=rows, experts=experts)
        )
    host_logits = np.zeros((4, 384), np.float32)
    host_logits[0, 17] = np.nan
    image_bias = np.zeros(384, np.float32)
    image_bias[37] = np.nan
    mixed = prepare(
        make_plan(rows=4, experts=384),
        device,
        Weights(device.zeros((384,), "float32"), device.upload(image_bias)),
    )
    mixed_inputs = Inputs(
        device.upload(host_logits),
        device.upload(np.array([0, 2, 1, 0], np.uint8)),
    )
    try:

        def check_invalid_rows():
            ids = device.download(mixed.outputs.expert_ids, np.int32)
            coefficients = device.download(mixed.outputs.coefficients, np.float32)
            require(np.all(ids[:3] == -1), "invalid routing row has expert IDs")
            require(np.isnan(coefficients[:3]).all(), "invalid routing row is finite")
            require(
                np.array_equal(ids[3], np.arange(6))
                and np.allclose(coefficients[3], 0.25, rtol=0, atol=1e-7),
                "invalid routing row affected a valid text row",
            )

        execute_modes(
            device, (mixed,), lambda: mixed.run(mixed_inputs), check_invalid_rows
        )
    finally:
        device.synchronize()
        mixed.close()
    for experts in (128, 384):
        weights = Weights(*(device.empty((experts,), "float32") for _ in range(2)))
        empty = prepare(make_plan(rows=0, experts=experts), device, weights)
        inputs = Inputs(
            device.empty((0, experts), "float32"), device.empty((0,), "uint8")
        )
        try:
            execute_modes(
                device,
                (empty,),
                lambda empty=empty, inputs=inputs: empty.run(inputs),
                lambda empty=empty: require(
                    all(tensor.shape[0] == 0 for tensor in empty.outputs),
                    "empty routing changed shape",
                ),
            )
        finally:
            device.synchronize()
            empty.close()
        rejected(lambda empty=empty, inputs=inputs: empty.run(inputs))
    return {
        "invalid_arguments": 11,
        "invalid_value_isolation": True,
        "empty_eager_graph": True,
        "closed_workspace": True,
    }


def check_case(device, case, limits, parameters):
    from oh_my_vllm.operators.routing.api import Inputs, Weights
    from oh_my_vllm.operators.routing.plan import make_plan
    from oh_my_vllm.operators.routing.workspace import prepare

    groups = case.manifest["case"]["options"].get("groups", {"": None})
    workspaces, inputs, originals, outputs = [], [], [], {}
    try:
        for group in groups:
            prefix = group + "_" if group else ""
            logits, mask = (
                case.inputs[prefix + field] for field in ("logits", "image_mask")
            )
            biases = (
                [case.inputs[prefix + field] for field in ("bias", "bias_vl")]
                if group
                else parameters
            )
            weights = Weights(*(device.upload(value) for value in biases))
            values = Inputs(device.upload(logits), device.upload(mask))
            workspace = prepare(
                make_plan(rows=len(logits), experts=logits.shape[1]), device, weights
            )
            workspaces.append(workspace)
            inputs.append(values)
            originals.extend(
                zip((*values, *weights), (logits, mask, *biases), strict=True)
            )
            outputs.update(
                (prefix + name, tensor)
                for name, tensor in workspace.outputs._asdict().items()
            )
        result = {}

        def operation():
            for workspace, values in zip(workspaces, inputs, strict=True):
                workspace.run(values)

        def check():
            actual = {
                name: decode(
                    device.download(tensor, storage_dtype(str(tensor.dtype))),
                    str(tensor.dtype),
                )
                for name, tensor in outputs.items()
            }
            compared = compare(actual, case, limits)
            require(compared.passed, f"routing differs: {compared.failures}")
            for tensor, value in originals:
                require(
                    equal_bits(device.download(tensor, value.dtype), value),
                    "routing changed its inputs or weights",
                )
            result.update(metrics=compared.metrics)

        execute_modes(device, workspaces, operation, check)
        result.update(
            case=case.manifest["case"]["id"],
            native=native_identity(workspaces[0]),
            modes=["eager", "graph"],
        )
        return result
    finally:
        device.synchronize()
        for workspace in workspaces:
            workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.routing") is not None,
        "native routing boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    config = json.loads((args.model / "config.json").read_text())
    reader = CheckpointWeights(args.model, json.loads(raw), tensor_layout(config))
    metadata, cases = load_routing()
    identity = device_identity(selected_uuid())
    device = Device()
    try:
        with caller_stream(device):
            checks = []
            for name, case in cases.items():
                parameters = [
                    reader.tensor(weight).astype(np.float32)
                    for weight in case.manifest["case"]["weights"][1:]
                ]
                checks.append(
                    check_case(
                        device, case, metadata["cases"][name]["limits"], parameters
                    )
                )
            contracts = check_contracts(device)
        reader._verify_states()
        require(
            source_identity() == sources, "routing sources changed during execution"
        )
    finally:
        device.synchronize()
        device.close()
    report = dict(
        passed=True,
        device=identity,
        fixture_sha256=ROUTING_SHA256,
        checks=checks,
        contracts=contracts,
        sources=sources,
    )
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
