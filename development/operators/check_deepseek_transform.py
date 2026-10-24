"""Check fixed RMS, position-derived RoPE, and exact BF16 widening."""

import argparse
import hashlib
import importlib.util
import json
from contextlib import closing, nullcontext
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
from .check_deepseek_exact import (
    source_identity as common_sources,
)
from .numerics import compare
from .storage import decode, equal_bits


def source_identity():
    root = Path(__file__).resolve().parents[2]
    sources = common_sources()
    for path in (root / "python/oh_my_vllm/operators/transform").rglob("*.py"):
        sources[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name in (
        "tests/test_collectives.py",
        "python/oh_my_vllm/models/deepseek_v41/rotary.py",
    ):
        path = root / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def check_rotary_tables(device, metadata, cases):
    from oh_my_vllm.models.deepseek_v41.rotary import prepare

    results = []
    with closing(prepare(device, length=1048576)) as tables:
        positions = device.upload(np.array([0], np.int32))
        for table in (tables.window, tables.compressed):
            alias = device.view(table, (1, 32, 2))
            rejected(
                lambda alias=alias: tables.gather(positions, alias, compressed=False)
            )
        for name in ("window", "compressed", "indexdecode"):
            case = cases[name]
            expected = case.inputs["coefficients"]
            positions = device.upload(case.inputs["positions"].astype(np.int32))
            coefficients = device.empty(expected.shape, "float32")

            def operation(positions=positions, coefficients=coefficients, name=name):
                tables.gather(positions, coefficients, compressed=name != "window")

            operation()
            rotation = check_case(
                device,
                case,
                metadata["cases"][name]["limits"],
                heads={"window": 16, "compressed": 64, "indexdecode": 32}[name],
                coefficients=coefficients,
            )
            results.append(rotation)
            for mode in ("eager", "graph"):
                with (
                    tables.graph_lease(),
                    device.graph(operation)
                    if mode == "graph"
                    else nullcontext(operation) as invoke,
                ):
                    for order in (slice(None), slice(None, None, -1)):
                        reordered = device.upload(
                            case.inputs["positions"][order].astype(np.int32)
                        )
                        device.module.copy_bytes(positions, reordered)
                        device.module.zero(coefficients)
                        invoke()
                        actual = device.download(coefficients, np.float32)
                        error = float(np.max(np.abs(actual - expected[order])))
                        # The frozen norm/RoPE policy allows 16 FP32 unit roundoffs.
                        require(
                            np.isfinite(actual).all() and error <= 2**-20,
                            f"generated {name} coefficients differ: {error}",
                        )
                results.append(dict(case=name, mode=mode, coefficient_max_abs=error))
        # A recorded gather retains the two startup tables after owner closure.
        with tables.graph_lease(), device.graph(operation) as replay:
            tables.close()
            rejected(operation)
            device.module.zero(coefficients)
            replay()
            require(
                np.allclose(
                    device.download(coefficients, np.float32),
                    expected,
                    rtol=0,
                    atol=2**-20,
                ),
                "recorded coefficient gather failed after owner close",
            )
    for length in (0, -1, True, 1048577):
        rejected(lambda length=length: prepare(device, length=length))
    return results


def check_table_contracts(device):
    from oh_my_vllm.models.deepseek_v41.rotary import prepare
    from oh_my_vllm.operators.runtime.native import load
    from oh_my_vllm.operators.transform.api import FUNCTIONS

    module, _, _ = load("transform", FUNCTIONS)
    frequencies = device.upload(np.ones(32, np.float64))
    output = device.empty((1, 32, 2), "float32")
    for invalid in (
        device.empty((32,), "float32"),
        device.empty((1, 32), "float64"),
    ):
        rejected(lambda invalid=invalid: module.rope_table(invalid, output))
    for invalid in (
        device.empty((0, 32, 2), "float32"),
        device.empty((1, 64), "float32"),
        device.empty((1, 32, 2), "bfloat16"),
    ):
        rejected(lambda invalid=invalid: module.rope_table(frequencies, invalid))
    alias, guards = metadata_tensor(32, pointer=output.data_ptr(), code=2, bits=64)
    rejected(lambda alias=alias: module.rope_table(alias, output))
    del alias, guards
    for options in (
        dict(stride=2),
        dict(device=1),
        dict(pointer=frequencies.data_ptr() + 8),
    ):
        invalid, guards = metadata_tensor(
            32, code=2, bits=64, **(dict(pointer=frequencies.data_ptr()) | options)
        )
        rejected(lambda invalid=invalid: module.rope_table(invalid, output))
        del invalid, guards
    for operation in (
        lambda: module.rope_table(frequencies, output),
        lambda: prepare(device, length=1),
    ):
        rejected(lambda operation=operation: device.capture(operation))
    module.rope_table(frequencies, output)
    expected = np.tile(np.array([1, 0], np.float32), (1, 32, 1))
    require(
        equal_bits(device.download(output, np.float32), expected),
        "position zero differs",
    )
    return dict(invalid_contracts=True, startup_only=True, position_zero_exact=True)


def check_widen_plans():
    from oh_my_vllm.operators.transform.plan import make_plan

    base = dict(kind="widen", rows=1, width=5120)
    for rows in (0, 1, 128, 384):
        plan = make_plan(**(base | dict(rows=rows)))
        require(
            plan.storage == (((rows, 5120), "float32"), ((rows,), "uint8")),
            "widening storage shape or type differs",
        )
    for change in (
        dict(rows=-1),
        dict(rows=385),
        dict(rows=True),
        dict(width=512),
        dict(width=True),
        dict(heads=2),
        dict(heads=True),
        dict(epsilon=1e-20),
        dict(inverse=True),
    ):
        rejected(lambda change=change: make_plan(**(base | change)))


def widen_values(fixture, rows, *, invalid=False):
    shape = (rows, 5120)
    values = np.resize(fixture["bf16_words"], shape)
    expected = np.resize(fixture["bf16_expected_f32_bits"], shape)
    status = np.zeros(rows, np.uint8)
    if invalid:
        require(rows == 128, "invalid widening fixture requires 128 rows")
        values[0, :128] = fixture["bf16_invalid_words"][:128]
        values[-1, -128:] = fixture["bf16_invalid_words"][128:]
        expected[0, :128], expected[-1, -128:] = 0, 0
        status[[0, 127]] = 1
    return values, expected, status


def check_widen_case(device, fixture, rows):
    from oh_my_vllm.operators.transform.api import Inputs
    from oh_my_vllm.operators.transform.plan import make_plan
    from oh_my_vllm.operators.transform.workspace import prepare

    states = [widen_values(fixture, rows)]
    if rows == 128:
        states.extend((widen_values(fixture, rows, invalid=True), states[0]))
    uploads = [device.upload(state[0], "bfloat16") for state in states]
    values = device.empty((rows, 5120), "bfloat16")
    workspace = prepare(make_plan(kind="widen", rows=rows, width=5120), device)
    inputs = Inputs(values, None)
    poisons = [
        device.upload(np.full((rows, 5120), 90, np.float32)),
        device.upload(np.full(rows, 0x5A, np.uint8)),
    ]

    def operation():
        return workspace.run(inputs)

    try:
        for mode in ("eager", "graph"):
            with (
                workspace.graph_lease(),
                device.graph(operation)
                if mode == "graph"
                else nullcontext(operation) as invoke,
            ):
                for source, (host, expected, status) in zip(
                    uploads, states, strict=True
                ):
                    device.module.copy_bytes(values, source)
                    for _ in range(2):
                        for output, poison in zip(
                            workspace.outputs, poisons, strict=True
                        ):
                            device.module.copy_bytes(output, poison)
                        invoke()
                        require(
                            equal_bits(
                                device.download(workspace.outputs.value, np.uint32),
                                expected,
                            )
                            and equal_bits(
                                device.download(workspace.outputs.status, np.uint8),
                                status,
                            ),
                            "widening output bits or row status differ",
                        )
                        require(
                            equal_bits(device.download(values, np.uint16), host),
                            "widening changed its input",
                        )
        return dict(
            rows=rows,
            exact=True,
            invalid_then_valid=rows == 128,
            modes=["eager", "graph"],
            native=native_identity(workspace),
        )
    finally:
        device.synchronize()
        workspace.close()


def check_widen_contracts(device, fixture):
    from oh_my_vllm.operators.transform.api import Inputs
    from oh_my_vllm.operators.transform.plan import make_plan
    from oh_my_vllm.operators.transform.workspace import prepare

    host, expected, _ = widen_values(fixture, 2)
    values = device.upload(host, "bfloat16")
    inputs = Inputs(values, None)
    workspace = prepare(make_plan(kind="widen", rows=2, width=5120), device)
    output, status = workspace.outputs
    try:
        for invalid in (
            Inputs(values, values),
            Inputs(device.empty((2, 5120), "float32"), None),
            Inputs(device.empty((2, 512), "bfloat16"), None),
            Inputs(device.empty((10240,), "bfloat16"), None),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        for invalid_output, invalid_status in (
            (device.empty((2, 5120), "bfloat16"), status),
            (output, device.empty((2,), "int32")),
            (output, device.empty((1,), "uint8")),
        ):
            rejected(
                lambda invalid_output=invalid_output, invalid_status=invalid_status: (
                    workspace.module.widen(values, invalid_output, invalid_status)
                )
            )
        alias, guards = metadata_tensor(
            10240, pointer=output.data_ptr(), code=4, bits=16
        )
        alias = device.view(alias, (2, 5120))
        rejected(lambda alias=alias: workspace.module.widen(alias, output, status))
        del alias, guards
        for pointer in (values.data_ptr(), output.data_ptr()):
            alias, guards = metadata_tensor(2, pointer=pointer, code=1, bits=8)
            rejected(lambda alias=alias: workspace.module.widen(values, output, alias))
            del alias, guards
        for options in (dict(stride=2), dict(device=1)):
            invalid, guards = metadata_tensor(
                2, pointer=status.data_ptr(), code=1, bits=8, **options
            )
            rejected(
                lambda invalid=invalid: workspace.module.widen(values, output, invalid)
            )
            del invalid, guards
        poison = device.upload(np.full(2, 0x5A, np.uint8))
        with (
            workspace.graph_lease(),
            device.graph(lambda: workspace.run(inputs)) as replay,
        ):
            workspace.close()
            rejected(lambda: workspace.run(inputs))
            rejected(lambda: workspace.graph_lease().__enter__())
            device.module.zero(output)
            device.module.copy_bytes(status, poison)
            replay()
            require(
                equal_bits(device.download(output, np.uint32), expected)
                and not device.download(status, np.uint8).any(),
                "recorded widening graph failed after workspace close",
            )
    finally:
        device.synchronize()
        workspace.close()
    return dict(
        invalid_contracts=True, overlap=True, stride=True, closed_owner_graph=True
    )


def check_case(device, case, limits, heads=1, coefficients=None):
    from oh_my_vllm.operators.transform.api import Inputs
    from oh_my_vllm.operators.transform.plan import make_plan
    from oh_my_vllm.operators.transform.workspace import prepare

    shape, options = (case.manifest["case"][key] for key in ("shape", "options"))
    rotation = "coefficients" in case.inputs
    parameter = case.inputs["coefficients" if rotation else "weight"]
    original = case.inputs["values"]
    if rotation:
        original = np.repeat(original[:, None, :], heads, axis=1)
        original[:, 1::2] ^= np.uint16(0x8000)
    values = device.upload(original, "bfloat16")
    if coefficients is None:
        parameters = device.upload(parameter, "float32" if rotation else "bfloat16")
    else:
        parameters = coefficients
        parameter = device.download(coefficients, np.float32)
    inputs = Inputs(values, parameters)
    workspaces, outputs, result = [], {}, {}
    try:
        for field in case.expected:
            plan = make_plan(
                kind="rope" if rotation else "rms",
                rows=shape["rows"],
                width=shape["width"],
                heads=heads,
                epsilon=None if rotation else options["eps"],
                inverse=field == "inverse",
            )
            workspace = prepare(plan, device)
            workspaces.append(workspace)
            outputs[field] = workspace.outputs.value

        def operation():
            for workspace in workspaces:
                workspace.run(inputs)

        def check():
            raw = {
                field: device.download(output, np.uint16)
                for field, output in outputs.items()
            }
            if rotation:
                require(
                    all(
                        equal_bits(value[..., :-64], original[..., :-64])
                        for value in raw.values()
                    ),
                    "RoPE changed prefix bits",
                )
            metrics = []
            for head in range(heads):
                actual = {
                    field: decode(
                        value[:, head, -64:] if rotation else value, "bfloat16"
                    )
                    * (-1 if head % 2 else 1)
                    for field, value in raw.items()
                }
                compared = compare(actual, case, limits)
                require(compared.passed, f"transform differs: {compared.failures}")
                metrics.append(compared.metrics)
            require(
                equal_bits(device.download(values, np.uint16), original)
                and equal_bits(device.download(parameters, parameter.dtype), parameter),
                "transform changed its inputs",
            )
            result.update(metrics=metrics, heads=heads)

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


def check_contract(device, kind, rows):
    from oh_my_vllm.operators.transform.api import Inputs
    from oh_my_vllm.operators.transform.plan import make_plan
    from oh_my_vllm.operators.transform.workspace import prepare

    options = dict(kind=kind, width=512, epsilon=1e-20 if kind == "rms" else None)
    workspace = prepare(make_plan(rows=rows, **options), device)
    shape = (rows, 512) if kind == "rms" else (rows, 1, 512)
    values = device.zeros(shape, "bfloat16")
    parameter = device.zeros(
        (512,) if kind == "rms" else (rows, 32, 2),
        "bfloat16" if kind == "rms" else "float32",
    )
    inputs = Inputs(values, parameter)
    try:
        if rows:
            for invalid in (
                Inputs(device.empty(shape, "float32"), parameter),
                Inputs(device.empty((*shape[:-1], 256), "bfloat16"), parameter),
                Inputs(values, device.empty(parameter.shape, "float64")),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            native = workspace.module
            if kind == "rms":
                rejected(lambda: native.rms(values, parameter, values, 1e-20))
                rejected(
                    lambda: native.rms(
                        values, parameter, workspace.outputs.value, float("nan")
                    )
                )
            else:
                rejected(lambda: native.rope(values, parameter, values, False))
        else:
            execute_modes(
                device,
                (workspace,),
                lambda: workspace.run(inputs),
                lambda: require(
                    workspace.outputs.value.shape == shape,
                    "empty transform changed shape",
                ),
            )
    finally:
        device.synchronize()
        workspace.close()
    rejected(lambda: workspace.run(inputs))


def check_contracts(device):
    from oh_my_vllm.operators.transform.plan import make_plan

    for kind in ("rms", "rope"):
        for rows in (1, 0):
            check_contract(device, kind, rows)
    for options in (
        dict(kind="other", rows=1, width=512),
        dict(kind="rms", rows=True, width=512, epsilon=1e-20),
        dict(kind="rms", rows=385, width=512, epsilon=1e-20),
        dict(kind="rms", rows=1, width=128, epsilon=1e-20),
        dict(kind="rms", rows=1, width=512, epsilon=0),
        dict(kind="rms", rows=1, width=512, epsilon=1e-20, inverse=True),
        dict(kind="rope", rows=1, width=512, epsilon=1e-20),
        dict(kind="rope", rows=1, width=512, inverse=1),
        dict(kind="rope", rows=1, width=512, heads=3),
    ):
        rejected(lambda options=options: make_plan(**options))
    return {
        "invalid_arguments": True,
        "empty_eager_graph": True,
        "closed_workspace": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.transform") is not None,
        "native norm/RoPE boundary is not implemented",
    )
    require(
        importlib.util.find_spec("oh_my_vllm.models.deepseek_v41.rotary") is not None,
        "model RoPE coefficient generation is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    check_widen_plans()
    metadata, cases = fixtures.load_norm_rope()
    sources = source_identity()
    report = {
        "fixture_sha256": fixtures.NORM_ROPE_SHA256,
        "widen_fixture_sha256": fixtures.STARTUP_PROMOTION_SHA256,
        "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
        "scope": (
            "RMS, position-derived tail rotation coefficients, and exact BF16 "
            "widening; excludes full model and vision rotation."
        ),
        "device": device_identity(selected_uuid()),
        "sources": sources,
    }
    with closing(Device()) as device, caller_stream(device):
        report["device_native"] = native_identity(device)
        report["table_contracts"] = check_table_contracts(device)
        report["rotary_tables"] = check_rotary_tables(device, metadata, cases)
        report["contracts"] = check_contracts(device)
        report["cases"] = [
            check_case(
                device,
                case,
                metadata["cases"][name]["limits"],
            )
            for name, case in cases.items()
            if "coefficients" not in case.inputs
        ]
        report["cases"].extend(
            check_case(
                device,
                cases["indexdecode"],
                metadata["cases"]["indexdecode"]["limits"],
                heads=heads,
            )
            for heads in (1, 8)
        )
        promotion = fixtures.load_startup_promotion()
        report["widen"] = [
            check_widen_case(device, promotion, rows) for rows in (0, 1, 128, 384)
        ]
        report["widen_contracts"] = check_widen_contracts(device, promotion)
    require(source_identity() == sources, "transform sources changed during validation")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
