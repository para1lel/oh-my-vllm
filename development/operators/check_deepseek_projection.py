"""Check encoded A8W8/A8W4 projection against complete frozen FP64 outputs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    PROJECTION_SHA256,
    load_projection,
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
from .storage import decode, equal_bits


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    for path in (repository / "python/oh_my_vllm/operators/projection").rglob("*.py"):
        sources[str(path.relative_to(repository))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def weight_values(reader, case):
    definition = case.manifest["case"]
    if not definition["weights"]:
        return [case.inputs[name] for name in ("weight", "weight_scales")]
    options = definition["options"]
    values = [reader._raw(name) for name in definition["weights"]]
    axis, rank = options["shard_axis"], options["rank"]
    if axis is not None:
        selected = []
        for value in values:
            slices = [slice(None)] * 2
            width = value.shape[axis] // 4
            slices[axis] = slice(rank * width, (rank + 1) * width)
            selected.append(np.ascontiguousarray(value[tuple(slices)]))
        values = selected
    return values


def packed_scales(scales, rows, width, bits):
    groups = ((width + 127) // 128) * 4
    padded_rows = ((rows + 127) // 128) * 128
    result = np.full(padded_rows * groups, 127, np.uint8)
    n, g = np.indices((rows, width // 32))
    # Native scale atoms contain 128 rows and four groups, with K tiles first.
    offsets = (
        (n // 128) * (groups // 4) * 512
        + (g // 4) * 512
        + (n % 32) * 16
        + ((n % 128) // 32) * 4
        + g % 4
    )
    result[offsets] = scales[n if bits == 4 else n // 32, g]
    return result


def check_case(device, case, limits, values):
    from oh_my_vllm.operators.projection.api import Inputs, Weights
    from oh_my_vllm.operators.projection.plan import make_plan
    from oh_my_vllm.operators.projection.workspace import prepare, prepare_weights

    shape = case.manifest["case"]["shape"]
    rows, width, outputs = (shape[key] for key in ("rows", "width", "outputs"))
    bits = 4 if case.manifest["case"]["options"]["weight_dtype"] == "I8" else 8
    host_weight, host_scales = values[0].view(np.uint8), values[1]
    raw = Weights(device.upload(host_weight), device.upload(host_scales))
    weights = prepare_weights(device, raw, bits=bits)
    plan = make_plan(rows=rows, outputs=outputs, width=width, bits=bits)
    workspace = prepare(plan, device, weights)
    inputs = Inputs(
        *(device.upload(case.inputs[name]) for name in ("encoded", "scales"))
    )
    result = {}
    try:
        wanted_scale = packed_scales(host_scales, outputs, width, bits)
        actual_scale = device.download(weights.scales, np.uint8)
        require(equal_bits(actual_scale, wanted_scale), "weight scale layout differs")
        padding = ((width + 127) // 128 * 128 - width) // 2 if bits == 4 else 0
        wanted_weight = np.pad(host_weight, ((0, 0), (0, padding)))
        require(
            equal_bits(device.download(weights.encoded, np.uint8), wanted_weight),
            "prepared encoded weights differ",
        )
        if not padding:
            require(weights.encoded is raw.encoded, "aligned weights were duplicated")
        originals = (
            (raw.encoded, host_weight),
            (raw.scales, host_scales),
            (weights.encoded, wanted_weight),
            (weights.scales, wanted_scale),
            (inputs.encoded, case.inputs["encoded"]),
            (inputs.scales, case.inputs["scales"]),
        )

        def check():
            actual = decode(
                device.download(workspace.outputs.value, np.uint16), "bfloat16"
            )
            compared = compare({"value": actual}, case, limits)
            require(compared.passed, f"projection differs: {compared.failures}")
            for first, last, column in case.manifest["case"]["options"]["zero_columns"]:
                zero = actual[first:last] if column < 0 else actual[first:last, column]
                require(np.max(np.abs(zero)) <= 1e-6, "cancellation endpoint differs")
            for tensor, value in originals:
                require(
                    equal_bits(device.download(tensor, value.dtype), value),
                    "projection changed its inputs or weights",
                )
            result.update(metrics=compared.metrics)

        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check)
        result.update(
            case=case.manifest["case"]["id"],
            native=native_identity(workspace),
            modes=["eager", "graph"],
            prepared_bytes=wanted_weight.nbytes + wanted_scale.nbytes,
        )
        return result
    finally:
        device.synchronize()
        workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.projection.api import Inputs, Weights
    from oh_my_vllm.operators.projection.plan import make_plan
    from oh_my_vllm.operators.projection.workspace import prepare, prepare_weights

    base = dict(rows=1, outputs=32, width=32, bits=8)
    invalids = (
        dict(rows=-1),
        dict(rows=385),
        dict(rows=True),
        dict(outputs=33),
        dict(outputs=65536),
        dict(width=31),
        dict(width=32768),
        dict(bits=16),
        dict(bits=True),
    )
    for change in invalids:
        rejected(lambda change=change: make_plan(**(base | change)))

    def check_format(bits):
        raw = Weights(
            device.zeros((32, 32 * bits // 8), "uint8"),
            device.upload(np.full((32 if bits == 4 else 1, 1), 127, np.uint8)),
        )
        weights = prepare_weights(device, raw, bits=bits)
        rejected(
            lambda: prepare_weights(
                device, Weights(raw.encoded, device.empty((1,), "float32")), bits=bits
            )
        )
        rejected(
            lambda: prepare(
                make_plan(**(base | dict(width=64, bits=bits))), device, weights
            )
        )
        workspace = prepare(make_plan(**(base | dict(bits=bits))), device, weights)
        inputs = Inputs(
            device.zeros((1, 32), "uint8"),
            device.upload(np.full((1, 1), 127, np.uint8)),
        )
        try:
            overlap = device.empty((4096,), "uint8")
            rejected(
                lambda: workspace.module.pack_weights(
                    device.view(overlap, raw.encoded.shape),
                    raw.scales,
                    device.view(overlap, weights.encoded.shape, 16),
                    weights.scales,
                    bits,
                )
            )
            for invalid in (
                Inputs(device.empty((1, 32), "float32"), inputs.scales),
                Inputs(inputs.encoded, device.empty((1, 1), "int32")),
                Inputs(device.empty((2, 32), "uint8"), inputs.scales),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            rejected(
                lambda: workspace.module.pack_weights(
                    *raw, weights.encoded, device.view(raw.encoded, (512,)), bits
                )
            )
            rejected(
                lambda: workspace.module.project(
                    inputs.encoded,
                    device.view(workspace.packed_input_scales, (1, 1)),
                    weights.encoded,
                    weights.scales,
                    workspace.outputs.value,
                    workspace.padded_input,
                    workspace.packed_input_scales,
                    workspace.scratch,
                    32,
                    bits,
                )
            )
        finally:
            device.synchronize()
            workspace.close()
        rejected(lambda: workspace.run(inputs))
        empty = prepare(make_plan(**(base | dict(rows=0, bits=bits))), device, weights)
        values = Inputs(device.empty((0, 32), "uint8"), device.empty((0, 1), "uint8"))
        try:
            execute_modes(
                device,
                (empty,),
                lambda: empty.run(values),
                lambda: require(
                    empty.outputs.value.shape[0] == 0, "empty rows changed"
                ),
            )
        finally:
            device.synchronize()
            empty.close()

    for bits in (4, 8):
        check_format(bits)
    return dict(
        invalid_plans=len(invalids),
        invalid_contracts=16,
        empty_eager_graph=True,
        closed_workspace=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.projection") is not None,
        "native projection boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    reader = CheckpointWeights(
        args.model,
        json.loads(raw),
        tensor_layout(json.loads((args.model / "config.json").read_text())),
    )
    metadata, cases = load_projection()
    identity = device_identity(selected_uuid())
    device = Device()
    try:
        with caller_stream(device):
            checks = [
                check_case(
                    device,
                    case,
                    metadata["cases"][name]["limits"],
                    weight_values(reader, case),
                )
                for name, case in cases.items()
            ]
            contracts = check_contracts(device)
        reader._verify_states()
        require(
            source_identity() == sources, "projection sources changed during execution"
        )
    finally:
        device.synchronize()
        device.close()
    with args.output.open("x") as output:
        json.dump(
            dict(
                passed=True,
                device=identity,
                fixture_sha256=PROJECTION_SHA256,
                checks=checks,
                contracts=contracts,
                sources=sources,
            ),
            output,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
