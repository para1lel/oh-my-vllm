"""Check packed sparse attention against five complete, fixed FP64 outputs."""

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
        Path(__file__),
        *(root / "python/oh_my_vllm/operators/sparse_attention").rglob("*.py"),
        root / "tests/test_collectives.py",
    ]
    for path in paths:
        sources[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def materialize(case):
    values = case.inputs
    defaults = case.manifest["case"]["options"]["defaults"]
    arrays = {name: values[name] for name in ("query", "sink", "indices")}
    for name, rows, width, group in (
        ("window", int(values["logical_rows"][0]), 512, 32),
        ("compressed", int(values["logical_rows"][1]), 256, 8),
    ):
        payload = np.full((rows, width), defaults[name + "_payload"], dtype=np.uint8)
        scales = np.full(
            (rows, width // group), defaults[name + "_scale"], dtype=np.uint8
        )
        ids = values[name + "_rows"]
        payload[ids], scales[ids] = values[name], values[name + "_scales"]
        arrays[name], arrays[name + "_scales"] = payload, scales
    return arrays


def upload(device, arrays):
    from oh_my_vllm.operators.sparse_attention.api import Inputs

    dtypes = {"query": "bfloat16", "sink": "float32", "indices": "int32"}
    return Inputs(
        **{
            name: device.upload(value, dtypes.get(name, "uint8"))
            for name, value in arrays.items()
        }
    )


def check_case(device, name, case, record):
    from oh_my_vllm.operators.sparse_attention.plan import make_plan
    from oh_my_vllm.operators.sparse_attention.workspace import prepare

    arrays = materialize(case)
    inputs = upload(device, arrays)
    rows, heads = case.inputs["query"].shape[:2]
    plan = make_plan(rows=rows, heads=heads, slots=case.inputs["indices"].shape[1])
    workspace, result = prepare(plan, device), {}

    def check():
        status = device.download(workspace.outputs.status, np.uint8)
        require(not status.any(), f"{name}: valid inputs rejected")
        value = decode(device.download(workspace.outputs.value, np.uint16), "bfloat16")
        measured = compare({"value": value}, case, record["limits"])
        require(measured.passed, f"{name}: {measured.failures}")
        for row in record["exact_guards"].get("zero_rows", []):
            require(not value[row].any(), "all-masked output is not zero")
        for row in record["exact_guards"].get("opposite_rows", []):
            require(
                np.max(np.abs(value[row]))
                <= record["exact_guards"]["opposite_max_abs"],
                "opposite-pair output differs",
            )
        result["metrics"] = measured.metrics

    try:
        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check)
        for field, original in arrays.items():
            require(
                equal_bits(
                    device.download(getattr(inputs, field), original.dtype), original
                ),
                f"{name}: input changed: {field}",
            )
        return dict(
            case=name,
            **result,
            native=native_identity(workspace),
            modes=["eager", "graph"],
        )
    finally:
        device.synchronize()
        workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.sparse_attention.plan import make_plan

    for settings in (
        dict(rows=True, heads=16, slots=1),
        dict(rows=385, heads=16, slots=1),
        dict(rows=1, heads=32, slots=1),
        dict(rows=1, heads=16, slots=641),
        dict(rows=-1, heads=16, slots=1),
    ):
        rejected(lambda settings=settings: make_plan(**settings))
    for rows, slots in ((0, 1), (1, 0), (1, 1)):
        check_contract(device, rows, slots)
    check_invalid_values(device)
    return dict(
        empty=True,
        invalid_ids=True,
        nonfinite_isolation=True,
        overlap=True,
        closed_workspace=True,
        graph_after_close=True,
    )


def check_contract(device, rows, slots):
    from oh_my_vllm.operators.sparse_attention.plan import make_plan
    from oh_my_vllm.operators.sparse_attention.workspace import prepare

    arrays = dict(
        query=np.zeros((rows, 16, 512), dtype=np.uint16),
        sink=np.zeros(16, dtype=np.float32),
        indices=np.full((rows, slots), -1, dtype=np.int32),
        window=np.zeros((1, 512), dtype=np.uint8),
        window_scales=np.full((1, 16), 127, dtype=np.uint8),
        compressed=np.zeros((1, 256), dtype=np.uint8),
        compressed_scales=np.full((1, 32), 56, dtype=np.uint8),
    )
    inputs = upload(device, arrays)
    workspace = prepare(make_plan(rows=rows, heads=16, slots=slots), device)

    def check_empty():
        require(
            not device.download(workspace.outputs.status, np.uint8).any(),
            "empty status",
        )
        require(
            not device.download(workspace.outputs.value, np.uint16).any(),
            "empty value",
        )

    try:
        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check_empty)
        if rows and slots:
            rejected(
                lambda: workspace.run(
                    inputs._replace(query=device.empty((1, 16, 512), "float32"))
                )
            )
            rejected(
                lambda: workspace.run(
                    inputs._replace(indices=device.empty((1, 2), "int32"))
                )
            )
            rejected(
                lambda: workspace.module.attend(
                    *inputs, inputs.query, workspace.outputs.status
                )
            )
            alias, owners = metadata_tensor(
                1, pointer=inputs.indices.data_ptr(), code=1, bits=8
            )
            alias = device.view(alias, (1, 1))
            rejected(
                lambda: workspace.module.attend(*inputs, workspace.outputs.value, alias)
            )
            del alias, owners
            for invalid in (-2, 2, np.iinfo(np.int32).max):
                device.module.copy_bytes(
                    inputs.indices, device.upload(np.array([[invalid]], np.int32))
                )

                def check_invalid():
                    require(
                        device.download(workspace.outputs.status, np.uint8).all(),
                        "invalid ID accepted",
                    )
                    require(
                        not device.download(workspace.outputs.value, np.uint16).any(),
                        "invalid ID output",
                    )

                execute_modes(
                    device,
                    (workspace,),
                    lambda: workspace.run(inputs),
                    check_invalid,
                )
            device.module.copy_bytes(
                inputs.indices, device.upload(np.array([[-1]], np.int32))
            )
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(inputs)) as replay,
            ):
                workspace.close()
                replay()
                check_empty()
    finally:
        device.synchronize()
        workspace.close()
    rejected(lambda: workspace.run(inputs))


def invalid_values(kind):
    arrays = dict(
        query=np.zeros((3, 64, 512), dtype=np.uint16),
        sink=np.zeros(64, dtype=np.float32),
        indices=np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int32),
        window=np.full((3, 512), 0x38, dtype=np.uint8),
        window_scales=np.full((3, 16), 127, dtype=np.uint8),
        compressed=np.full((3, 256), 0x22, dtype=np.uint8),
        compressed_scales=np.full((3, 32), 0x38, dtype=np.uint8),
    )
    status = np.zeros((3, 4), dtype=np.uint8)
    if kind == "query":
        arrays["query"][0, 0, 0] = 0x7FC0
        arrays["query"][1, 16:32] = 0x7F7F
        status[0, 0] = status[1, 1] = 1
    elif kind == "sink":
        arrays["sink"][32] = np.nan
        status[:, 2] = 1
    elif kind == "window":
        arrays["window"][0, 0] = 127
        arrays["window_scales"][1, 0] = 255
        status[:2] = 1
    elif kind == "compressed":
        arrays["compressed_scales"][0, 0] = 127
        arrays["compressed_scales"][1, 0] = 0
        status[:2] = 1
    elif kind == "roundtrip":
        arrays["window"][0, 0] = 126
        arrays["window_scales"][0, 0] = 254
        status[0] = 1
    elif kind == "accumulation":
        arrays["window"][0] = 0x3F
        arrays["window_scales"][0] = 254
        arrays["indices"][0] = 0
        status[0] = 1
    else:
        raise ValueError("unknown invalid-value probe")
    return arrays, status


def check_invalid_values(device):
    from oh_my_vllm.operators.sparse_attention.plan import make_plan
    from oh_my_vllm.operators.sparse_attention.workspace import prepare

    workspace = prepare(make_plan(rows=3, heads=64, slots=2), device)
    try:
        for kind in (
            "query",
            "sink",
            "window",
            "compressed",
            "roundtrip",
            "accumulation",
        ):
            arrays, expected = invalid_values(kind)
            inputs = upload(device, arrays)

            def check(expected=expected):
                status = device.download(workspace.outputs.status, np.uint8)
                require(np.array_equal(status, expected), "rejection escaped its tile")
                value = device.download(workspace.outputs.value, np.uint16).reshape(
                    3, 4, 16, 512
                )
                require(
                    not value[expected.astype(bool)].any(), "rejected tile is not zero"
                )
                # Two unit KV entries and a zero sink give BF16_RN(2/3) = 0x3f2b.
                require(
                    np.all(value[~expected.astype(bool)] == 0x3F2B),
                    "rejection changed a valid tile",
                )

            execute_modes(
                device, (workspace,), lambda inputs=inputs: workspace.run(inputs), check
            )
    finally:
        device.synchronize()
        workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.sparse_attention") is not None,
        "native packed sparse-attention boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    metadata, cases = fixtures.load_sparse_attention()
    sources = source_identity()
    report = dict(
        fixture_sha256=fixtures.SPARSE_ATTENTION_SHA256,
        checkpoint_manifest_sha256=fixtures.CHECKPOINT_MANIFEST_SHA256,
        scope=(
            "Selected packed KV, original sink and BF16 queries; "
            "no projections, index selection, state or performance."
        ),
        sources=sources,
        device=device_identity(selected_uuid()),
    )
    with closing(Device()) as device, caller_stream(device):
        report["device_native"] = native_identity(device)
        report["contracts"] = check_contracts(device)
        report["cases"] = [
            check_case(device, name, case, metadata["cases"][name])
            for name, case in cases.items()
        ]
    require(source_identity() == sources, "sources changed during the check")
    report["passed"] = True
    with args.output.open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
