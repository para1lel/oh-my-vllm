"""Check frozen DeepSeek exact primitives in Eager and CUDA Graph execution."""

import argparse
import ctypes
import hashlib
import json
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import numpy as np
import tvm_ffi
from oh_my_vllm.operators.engram import api as engram
from oh_my_vllm.operators.engram.plan import make_plan as hash_plan
from oh_my_vllm.operators.engram.workspace import prepare as prepare_hash
from oh_my_vllm.operators.quantization import api as quantization
from oh_my_vllm.operators.quantization.plan import make_plan as quant_plan
from oh_my_vllm.operators.quantization.workspace import prepare as prepare_quant
from oh_my_vllm.operators.runtime.device import Device

from development.model.deepseek_v41.fixtures import (
    EXACT_OPERATORS_SHA256,
    load_exact_operators,
)
from development.serving.device import device_identity, selected_uuid

from .cases import digest
from .fixtures import Fixture
from .numerics import compare
from .performance_inputs import bf16
from .storage import equal_bits, storage_dtype

MODES = {
    "activation_fp8": "fp8_ue8m0",
    "indexer_fp4": "fp4_ue8m0",
    "compressed_kv_fp4": "fp4_e4m3",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("invalid tensor contract was accepted")


def native_identity(workspace):
    return {
        "source_id": workspace.source_id,
        "binary_sha256": hashlib.sha256(
            Path(workspace.library).read_bytes()
        ).hexdigest(),
    }


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    paths = [
        *(
            path
            for family in ("runtime", "quantization", "engram")
            for path in (repository / "python/oh_my_vllm/operators" / family).rglob(
                "*.py"
            )
        ),
        *Path(__file__).parent.glob("*.py"),
        repository / "development/model/deepseek_v41/fixtures.py",
        repository / "development/serving/device.py",
        repository / "development/serving/gpu.py",
    ]
    return {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


@contextmanager
def caller_stream(device):
    stream = ctypes.c_void_p()
    device.checked(device.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
    try:
        with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
            yield
    finally:
        try:
            device.checked(device.runtime.cudaStreamSynchronize(stream))
        finally:
            device.checked(device.runtime.cudaStreamDestroy(stream))


def execute_modes(device, workspaces, operation, check):
    poisons = []
    for workspace in workspaces:
        for output in workspace.outputs:
            host = np.full(output.shape, 0x5A, storage_dtype(str(output.dtype)))
            poisons.append((output, device.upload(host, str(output.dtype)), host))
    device.synchronize()
    for mode in ("eager", "graph"):
        with ExitStack() as resources:
            for workspace in workspaces:
                resources.enter_context(workspace.graph_lease())
            invoke = resources.enter_context(
                device.graph(operation) if mode == "graph" else nullcontext(operation)
            )
            for _ in range(2):
                for output, sentinel, _ in poisons:
                    device.module.copy_bytes(output, sentinel)
                invoke()
                check()


def repeated_case(fixture, rows, width):
    base_rows, base_width = fixture.inputs["values"].shape
    require(width % base_width == 0, "case expansion must keep complete groups")

    def repeat(value):
        return np.tile(
            value, ((rows + base_rows - 1) // base_rows, width // base_width)
        )[:rows]

    # Repetition keeps complete independent groups and their frozen exact outputs.
    case = dict(
        fixture.manifest["case"],
        id=f"{fixture.manifest['case']['id']}.{rows}x{width}",
        shape={"rows": rows, "channels": width},
    )
    limits = dict(fixture.manifest["limits"], case_sha256=digest(case))
    return Fixture(
        {"case": case, "limits": limits},
        {"values": repeat(fixture.inputs["values"])},
        {},
        {key: repeat(value) for key, value in fixture.expected.items()},
    )


def quantization_case(device, name, fixture, rows, width):
    base_shape = fixture.inputs["values"].shape
    case = repeated_case(fixture, rows, width)
    plan = quant_plan(rows=rows, width=width, mode=MODES[name])
    workspace = prepare_quant(plan, device)
    try:
        initial = bf16(case.inputs["values"])
        value = device.upload(initial, "bfloat16")
        inputs = quantization.Inputs(value)
        identity = native_identity(workspace)

        def operation():
            quantization.run(workspace, inputs)

        def check():
            outputs = workspace.outputs
            require(
                not np.any(device.download(outputs.status, np.uint8)),
                f"{name}: valid input produced an error status",
            )
            actual = {
                "encoded": device.download(outputs.encoded, np.uint8),
                "scales": device.download(outputs.scales, np.uint8),
                "dequantized_bf16_bits": device.download(
                    outputs.dequantized, np.uint16
                ),
            }
            result = compare(actual, case, case.manifest["limits"])
            require(result.passed, f"{name} {rows}x{width}: {result.failures}")
            require(
                equal_bits(device.download(value, np.uint16), initial),
                "quantization changed its input",
            )

        execute_modes(device, (workspace,), operation, check)
        if (rows, width) == base_shape:
            rejected(
                lambda: workspace.run(
                    quantization.Inputs(device.empty((rows, width), "float32"))
                )
            )
            rejected(
                lambda: workspace.run(
                    quantization.Inputs(device.empty((rows + 1, width), "bfloat16"))
                )
            )
            output = workspace.outputs
            rejected(
                lambda: workspace.module.quantize(
                    value,
                    output.encoded,
                    output.scales,
                    output.dequantized,
                    output.scales,
                    plan.native_mode,
                )
            )
            invalid = initial.copy()
            invalid[0, 0] = 0x7FC0
            device.module.copy_bytes(value, device.upload(invalid, "bfloat16"))
            operation()
            require(
                device.download(output.status, np.uint8)[0, 0] == 1,
                "nonfinite BF16 input was not reported",
            )
        return (
            identity,
            {"case": name, "rows": rows, "width": width, "modes": ["eager", "graph"]},
        )
    finally:
        try:
            device.synchronize()
        finally:
            workspace.close()


def check_quantization(device, name, fixture):
    base_shape = fixture.inputs["values"].shape
    records, identity = [], None
    for rows, width in (
        base_shape,
        (0, base_shape[1]),
        (1, 128),
        (64, 512),
        (320, 2304),
        (384, 5120),
    ):
        identity, record = quantization_case(device, name, fixture, rows, width)
        records.append(record)
    return identity, records


def engram_case(device, fixture, constants, constants_gpu, split):
    options = fixture.manifest["case"]["options"]
    layout, tokens = (options["layout"], fixture.inputs["tokens"])
    initial = fixture.state["history"]
    workspaces, inputs, guards = ([], [], [])
    try:
        history = device.upload(initial)
        history_rows = len(initial)
        for part in (tokens[:split], tokens[split:]):
            plan = hash_plan(
                rows=len(part),
                history_rows=history_rows,
                pad_id=options["pad_id"],
                vocabulary_size=layout["vocabulary_size"],
                layers=len(layout["primes"]),
                heads=layout["n_heads"],
                max_ngram_size=layout["max_ngram_size"],
            )
            workspace = prepare_hash(plan, device)
            workspaces.append(workspace)
            value = device.upload(part)
            inputs.append(engram.Inputs(value, history, **constants_gpu))
            guards.append((value, part))
            history = workspace.outputs.history
            history_rows = plan.next_history_rows
        identity = native_identity(workspaces[0])
        first_history = np.concatenate((initial, tokens[:split]))[
            -(layout["max_ngram_size"] - 1) :
        ]

        def operation():
            for workspace, values in zip(workspaces, inputs, strict=True):
                engram.hash_rows(workspace, values)

        def check():
            hashes = np.concatenate(
                [device.download(item.outputs.hashes, np.int64) for item in workspaces]
            )
            require(
                equal_bits(hashes, fixture.expected["hashes"]), "Engram hashes differ"
            )
            require(
                equal_bits(
                    device.download(workspaces[1].outputs.history, np.int64),
                    fixture.expected["history"],
                ),
                "Engram continuation history differs",
            )
            require(
                equal_bits(
                    device.download(workspaces[0].outputs.history, np.int64),
                    first_history,
                ),
                "Engram changed its accepted continuation input",
            )
            if split == options["split"]:
                require(
                    equal_bits(first_history, fixture.expected["history_after_split"]),
                    "Engram split differs from the fixed case",
                )
            for value, expected in (
                *guards,
                (inputs[0].history, initial),
                *((constants_gpu[key], array) for key, array in constants.items()),
            ):
                require(
                    equal_bits(device.download(value, np.int64), expected),
                    "Engram changed a read-only input",
                )

        execute_modes(device, workspaces, operation, check)
        if split == options["split"]:
            workspace, values = (workspaces[1], inputs[1])
            rejected(
                lambda: workspace.module.engram_hash(
                    values.tokens,
                    values.history,
                    values.multipliers,
                    values.primes,
                    values.offsets,
                    workspace.outputs.hashes,
                    values.history,
                    options["pad_id"],
                    layout["vocabulary_size"],
                )
            )
        return (
            identity,
            {"case": "engram", "split": split, "modes": ["eager", "graph"]},
        )
    finally:
        try:
            device.synchronize()
        finally:
            for workspace in workspaces:
                workspace.close()


def check_engram(device, fixture):
    layout = fixture.manifest["case"]["options"]["layout"]
    constants = {
        name: np.asarray(layout[name], np.int64)
        for name in ("multipliers", "primes", "offsets")
    }
    constants_gpu = {name: device.upload(value) for name, value in constants.items()}
    records, identity = [], None
    for split in range(len(fixture.inputs["tokens"]) + 1):
        identity, record = engram_case(device, fixture, constants, constants_gpu, split)
        records.append(record)
    return identity, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lookup-fd",
        type=int,
        help="check lookup only, using an inherited sealed fixture memfd",
    )
    options = parser.parse_args()
    output = options.output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repository) or output.exists():
        parser.error("use a new report path outside the repository")
    if options.lookup_fd is not None and options.lookup_fd < 0:
        parser.error("lookup descriptor must be nonnegative")
    from .check_engram_lookup import check_lookup

    _, cases = load_exact_operators()
    identity = device_identity(selected_uuid())
    device = Device()
    report = {
        "fixture_sha256": EXACT_OPERATORS_SHA256,
        "sources": source_identity(),
        "device": identity,
        "native": {"device": native_identity(device)},
        "checks": [],
    }
    try:
        with caller_stream(device):
            # An earlier FFI exception can break the sanitizer's HMM mapping tracking.
            native, checks, provenance = check_lookup(device, options.lookup_fd)
            report["native"]["engram"] = native
            report["checks"].extend(checks)
            report["engram_lookup"] = provenance
            if options.lookup_fd is None:
                for name in MODES:
                    native, checks = check_quantization(device, name, cases[name])
                    report["native"]["quantization"] = native
                    report["checks"].extend(checks)
                native, checks = check_engram(device, cases["engram"])
                report["native"]["engram"] = native
                report["checks"].extend(checks)
    finally:
        try:
            device.synchronize()
        finally:
            device.close()
    require(
        source_identity() == report["sources"], "checker or runtime sources changed"
    )
    report["passed"] = True
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(encoded)
    print(f"Exact checks passed: {len(report['checks'])} cases, Eager and CUDA Graph")


if __name__ == "__main__":
    main()
