"""Check unquantized projections against complete frozen FP64 outputs."""

import argparse
import hashlib
import json
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import tvm_ffi
from oh_my_vllm.models.deepseek_v41.checkpoint import (
    EncodedReader,
    TensorSlice,
)
from oh_my_vllm.models.deepseek_v41.weights import promote_bf16, promote_fp8
from oh_my_vllm.operators.linear.api import Inputs, Weights
from oh_my_vllm.operators.linear.plan import make_plan
from oh_my_vllm.operators.linear.workspace import prepare
from oh_my_vllm.operators.runtime.device import Device
from oh_my_vllm.operators.runtime.native import source_files

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
from .numerics import compare
from .storage import decode, equal_bits


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and name.endswith(".py") and Path(name).resolve().is_relative_to(root):
            paths.add(Path(name).resolve())
    paths.update(
        source_files(root / "python/oh_my_vllm/operators/linear/cuda/linear.cu")
    )
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def selections(cases):
    result = {}
    for case in cases.values():
        for record in case.manifest["case"]["options"]["weight_selections"].values():
            shape = tuple(record["original_shape"])
            result[record["name"]] = TensorSlice(
                record["original_dtype"], shape, (0, 0), shape
            )
            if "scale" in record:
                scale_shape = tuple((size + 31) // 32 for size in shape)
                result[record["scale"]["name"]] = TensorSlice(
                    "F8_E8M0", scale_shape, (0, 0), scale_shape
                )
    return result


def upload_weight(device, reader, record, output, group=0):
    first, _ = record["selection"]["start"]
    last, width = record["selection"]["stop"]
    count = last - first
    step = min(1024, (64 << 20) // (8 * width))
    raw_hash, prepared_hash, scale_hash = (hashlib.sha256() for _ in range(3))
    for offset in range(0, count, step):
        rows = min(step, count - offset)
        kind = np.uint8 if record["original_dtype"] == "F8_E4M3" else np.uint16
        raw = np.empty((rows, width), kind)
        require(
            reader.read_into(record["name"], first + offset, raw) == rows,
            "incomplete original weight tile",
        )
        raw_hash.update(raw.tobytes())
        if kind == np.uint8:
            start = first + offset
            scale = np.empty(((start % 32 + rows + 31) // 32, width // 32), np.uint8)
            require(
                reader.read_into(record["scale"]["name"], start // 32, scale)
                == len(scale),
                "incomplete original weight scale tile",
            )
            scale_hash.update(scale.tobytes())
            prepared = promote_fp8(raw, scale, row_offset=start % 32)
        else:
            prepared = raw
        if record["prepared_dtype"] == "F32":
            prepared = promote_bf16(prepared)
        prepared_hash.update(prepared.tobytes())
        destination = device.view(
            output, prepared.shape, (group * count + offset) * width
        )
        source = tvm_ffi.from_dlpack(prepared)
        device.module.copy_bytes(destination, source)
        device.synchronize_compute()
    require(raw_hash.hexdigest() == record["raw_sha256"], "raw selected weight differs")
    require(
        prepared_hash.hexdigest() == record["prepared_sha256"],
        "prepared weight differs",
    )
    if "scale" in record:
        require(
            scale_hash.hexdigest() == record["scale"]["sha256"], "weight scales differ"
        )


def weight_digest(device, value, group, rows, width, dtype):
    result = hashlib.sha256()
    step = min(1024, (64 << 20) // (8 * width))
    for first in range(0, rows, step):
        count = min(step, rows - first)
        part = device.view(value, (count, width), (group * rows + first) * width)
        result.update(device.download(part, dtype).tobytes())
    return result.hexdigest()


def bindings(device, reader, case):
    definition, inputs = case.manifest["case"], case.inputs
    shape, selected = definition["shape"], definition["options"]["weight_selections"]
    if selected:
        grouped = shape["groups"] != 1
        if grouped:
            records = list(selected.values())
            records.sort(key=lambda r: r["selection"]["start"][0])
            weight = device.empty((shape["groups"], 1024, 4096), "bfloat16")
            for group, record in enumerate(records):
                upload_weight(device, reader, record, weight, group)
            return [
                ("grouped", "value", weight, "bfloat16", "bfloat16", shape["groups"])
            ]
        result = []
        for field, record in selected.items():
            first, last = record["selection"]["start"], record["selection"]["stop"]
            dtype = "float32" if record["prepared_dtype"] == "F32" else "bfloat16"
            weight = device.empty((last[0] - first[0], last[1] - first[1]), dtype)
            upload_weight(device, reader, record, weight)
            input_field = field if field in inputs else "value"
            output_dtype = (
                "bfloat16"
                if definition["boundary"] == "deepseek.bf16-projection"
                else "float32"
            )
            result.append((field, input_field, weight, dtype, output_dtype, 1))
        return result
    weight = device.upload(inputs["weight"], "bfloat16")
    return [
        ("bf16_value", "value", weight, "bfloat16", "bfloat16", 1),
        ("router_value", "value", weight, "bfloat16", "float32", 1),
        (
            "low_input_value",
            "low_input",
            device.upload(inputs["unit_weight"]),
            "float32",
            "float32",
            1,
        ),
        (
            "low_weight_value",
            "unit_input",
            device.upload(inputs["low_weight"]),
            "float32",
            "float32",
            1,
        ),
    ]


def exact_fields(actual, case):
    options, expected = case.manifest["case"]["options"], case.expected
    queries = case.manifest["case"]["shape"]["queries"]
    for name in expected:
        if name.endswith("_zero_nonzero"):
            field = name.removesuffix("_zero_nonzero")
            guard = options["zero_guards"].get(field)
            rows = np.array([], np.int64)
            if guard:
                rows = np.arange(guard["first"], guard["last"])
                rows = rows[rows != guard["near_row"]]
            actual[name] = np.concatenate(
                (
                    [np.count_nonzero(actual[field][:queries])],
                    (actual[field][rows, 0] != 0),
                )
            ).astype(np.uint64)
    if "router_bits" in expected:
        actual["router_bits"] = (
            actual["router_value"]
            .astype(np.float32)
            .view(np.uint32)[:, [0, 16, 32, 96]]
        )
        actual["bf16_bits"] = (
            actual["bf16_value"].astype(np.float32).view(np.uint32)[:, [0, 16, 32, 96]]
            >> 16
        ).astype(np.uint16)
        actual["low_bits"] = (
            np.concatenate(
                (actual["low_input_value"], actual["low_weight_value"]), axis=1
            )
            .astype(np.float32)
            .view(np.uint32)
        )
        actual["zero_nonzero"] = np.concatenate(
            (
                actual["bf16_value"][:, 64:96] != 0,
                actual["router_value"][:, 64:96] != 0,
            ),
            axis=1,
        ).astype(np.uint8)


def check_case(device, reader, case, limits):
    shape = case.manifest["case"]["shape"]
    uploaded = {
        key: device.upload(value, "bfloat16" if value.dtype == np.uint16 else "float32")
        for key, value in case.inputs.items()
        if key not in ("weight", "unit_weight", "low_weight")
    }
    bound = bindings(device, reader, case)
    workspaces = []
    result = {}
    try:
        for _field, _input_name, weight, input_dtype, output_dtype, groups in bound:
            plan = make_plan(
                rows=shape["rows"],
                outputs=weight.shape[-2],
                width=weight.shape[-1],
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                groups=groups,
            )
            workspaces.append(prepare(plan, device, Weights(weight)))

        def operation():
            for workspace, binding in zip(workspaces, bound, strict=True):
                workspace.run(Inputs(uploaded[binding[1]]))

        def check():
            actual = {}
            for workspace, binding in zip(workspaces, bound, strict=True):
                field = binding[0]
                if workspace.plan.output_dtype == "bfloat16":
                    value = decode(
                        device.download(workspace.outputs.value, np.uint16), "bfloat16"
                    )
                else:
                    value = device.download(workspace.outputs.value, np.float32).astype(
                        np.float64
                    )
                if field == "grouped":
                    actual.update(
                        {f"value_g{g}": value[:, g] for g in range(shape["groups"])}
                    )
                else:
                    actual[field] = value
            exact_fields(actual, case)
            compared = compare(actual, case, limits)
            require(
                compared.passed, f"unquantized projection differs: {compared.failures}"
            )
            for key, value in uploaded.items():
                require(
                    equal_bits(
                        device.download(value, case.inputs[key].dtype), case.inputs[key]
                    ),
                    "linear input changed",
                )
            result["metrics"] = compared.metrics

        execute_modes(device, workspaces, operation, check)
        records = case.manifest["case"]["options"]["weight_selections"]
        for field, record in records.items():
            group = int(field.removeprefix("value_g")) if shape["groups"] != 1 else 0
            tensor = (
                workspaces[0].weights.value
                if shape["groups"] != 1
                else workspaces[list(records).index(field)].weights.value
            )
            first, last = record["selection"]["start"], record["selection"]["stop"]
            dtype = np.float32 if record["prepared_dtype"] == "F32" else np.uint16
            require(
                weight_digest(device, tensor, group, last[0] - first[0], last[1], dtype)
                == record["prepared_sha256"],
                "linear changed a prepared weight",
            )
        if not records:
            for binding, workspace in zip(bound, workspaces, strict=True):
                name = (
                    "weight"
                    if binding[0] in ("bf16_value", "router_value")
                    else (
                        "unit_weight"
                        if binding[0] == "low_input_value"
                        else "low_weight"
                    )
                )
                require(
                    equal_bits(
                        device.download(
                            workspace.weights.value, case.inputs[name].dtype
                        ),
                        case.inputs[name],
                    ),
                    "linear changed a synthetic weight",
                )
        return {
            "case": case.manifest["case"]["id"],
            **result,
            "native": native_identity(workspaces[0]),
            "modes": ["eager", "graph"],
        }
    finally:
        try:
            device.synchronize()
        finally:
            for workspace in workspaces:
                workspace.close()


def check_contract(device, rows, input_dtype, output_dtype):
    plan = make_plan(
        rows=rows,
        outputs=32,
        width=32,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    weight = device.zeros(plan.weight_shape, input_dtype)
    workspace = prepare(plan, device, Weights(weight))
    value = device.zeros(plan.input_shape, input_dtype)
    try:
        if rows:
            rejected(lambda: workspace.run(Inputs(device.empty((1, 32), "uint8"))))
            rejected(lambda: workspace.run(Inputs(device.empty((2, 32), input_dtype))))
            rejected(lambda: workspace.run(Inputs(device.view(weight, (1, 32)))))
            alias, guards = metadata_tensor(
                32,
                pointer=workspace.scratch.data_ptr(),
                code=4 if input_dtype == "bfloat16" else 2,
                bits=16 if input_dtype == "bfloat16" else 32,
            )
            alias = device.view(alias, plan.input_shape)
            rejected(lambda: workspace.run(Inputs(alias)))
            del alias, guards
            alias, guards = metadata_tensor(
                32,
                pointer=value.data_ptr(),
                code=4 if output_dtype == "bfloat16" else 2,
                bits=16 if output_dtype == "bfloat16" else 32,
            )
            alias = device.view(alias, plan.output_shape)
            rejected(
                lambda: workspace.module.linear_project(
                    workspace.owner, value, weight, alias, 1
                )
            )
            del alias, guards
            with caller_stream(device):
                rejected(lambda: workspace.run(Inputs(value)))
            workspace.run(Inputs(value))
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(Inputs(value))) as replay,
            ):
                workspace.close()
                replay()
                dtype = np.float32 if output_dtype == "float32" else np.uint16
                require(
                    not device.download(workspace.outputs.value, dtype).any(),
                    "recorded linear graph failed after workspace close",
                )
        else:
            execute_modes(
                device,
                (workspace,),
                lambda: workspace.run(Inputs(value)),
                lambda: require(
                    workspace.outputs.value.shape == (0, 32),
                    "empty linear output differs",
                ),
            )
    finally:
        device.synchronize()
        workspace.close()
    rejected(lambda: workspace.run(Inputs(value)))


def check_contracts(device):
    for input_dtype, output_dtype in (
        ("bfloat16", "bfloat16"),
        ("bfloat16", "float32"),
        ("float32", "float32"),
    ):
        for rows in (0, 1):
            check_contract(device, rows, input_dtype, output_dtype)
    return {
        "empty_eager_graph": True,
        "graph_after_close": True,
        "invalid_tensors": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == fixtures.CHECKPOINT_MANIFEST_SHA256,
        "checkpoint manifest identity differs",
    )
    manifest = json.loads(raw)
    require(manifest["verification"] == "full", "full checkpoint identity required")
    metadata, cases = fixtures.load_unquantized_projection()
    sources = source_identity()
    report = {
        "fixture_sha256": fixtures.UNQUANTIZED_PROJECTION_SHA256,
        "checkpoint_manifest_sha256": fixtures.CHECKPOINT_MANIFEST_SHA256,
        "device": device_identity(selected_uuid()),
        "sources": sources,
    }
    with (
        closing(Device()) as device,
        caller_stream(device),
        EncodedReader(args.model, manifest["files"], selections(cases)) as reader,
    ):
        report["device_native"] = native_identity(device)
        report["contracts"] = check_contracts(device)
        report["cases"] = [
            check_case(device, reader, case, metadata["cases"][name]["limits"])
            for name, case in cases.items()
        ]
        report["consumed_checkpoint_files"] = reader.finish()
    require(
        sha(args.manifest) == fixtures.CHECKPOINT_MANIFEST_SHA256, "manifest changed"
    )
    require(source_identity() == sources, "linear validation sources changed")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
