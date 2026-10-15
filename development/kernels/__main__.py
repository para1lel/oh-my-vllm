import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .roofline import ExpectedCase, Observation, expected_efficiencies
from .work import work_for
from .workload import PROFILES, SERVING_OPERATORS, modeled_cases

ROOT = Path(__file__).resolve().parents[2]


def case_id(shape):
    value = json.dumps(asdict(shape), sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()[:16]


def source_identity(native_id, binary, context):
    digest = hashlib.sha256(native_id.encode())
    for folder in (ROOT / "development/kernels", ROOT / "python/oh_my_vllm/kernels"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in {".py", ".cu", ".cuh"}:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    with binary.open("rb") as file:
        binary_id = hashlib.file_digest(file, "sha256").hexdigest()
    digest.update(binary_id.encode())
    digest.update(json.dumps(context, sort_keys=True).encode())
    return digest.hexdigest(), binary_id


def plan():
    cases = modeled_cases()
    return {
        "profiles": [asdict(profile) for profile in PROFILES],
        "serving_operators": SERVING_OPERATORS,
        "cases": [
            {
                "id": case_id(case.shape),
                "shape": asdict(case.shape),
                "calls": case.calls,
                "work": asdict(work_for(case.shape)),
            }
            for case in cases
        ],
    }


def evaluate(arguments, report):
    from .checkpoint import Checkpoint
    from .device import Device
    from .ffn import FFNCase, FFNWeights

    cases = [
        case
        for case in modeled_cases()
        if case.shape.operator == arguments.operator
        and (arguments.rows is None or case.shape.rows in arguments.rows)
    ]
    if not cases:
        raise ValueError("no modeled shapes selected")
    if arguments.profile and len(cases) != 1:
        raise ValueError("profiling requires exactly one modeled shape")
    if arguments.operator != "ffn":
        raise ValueError(f"native provider is missing: {arguments.operator}")
    device = Device()
    checkpoint = Checkpoint(arguments.model)
    weights = FFNWeights(device, checkpoint)
    report.update(
        {
            "operator": arguments.operator,
            "mode": arguments.mode,
            "accepted": False,
            "observations": [],
            "checks": {},
            "samples": {},
            "missing_providers": sorted(set(SERVING_OPERATORS) - {"ffn"}),
        }
    )
    first = FFNCase(device, weights, cases[0].shape)
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not selected_gpu or "," in selected_gpu:
        raise ValueError("use the single-GPU wrapper")
    context = {
        "checkpoint": checkpoint.identity,
        "mode": arguments.mode,
        "device": selected_gpu,
        "device_source": device.source_id,
    }
    source_id, binary_id = source_identity(
        first.operation.source_id, first.operation.library, context
    )
    report["source_id"], report["binary_id"] = source_id, binary_id
    report["context"] = context
    report["device"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--id",
            selected_gpu,
            "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    report["cache_policy"] = {
        "method": "ld.global.cg scan before start event",
        "bytes": device.eviction_bytes,
        "l2_bytes": device.l2_bytes,
    }
    try:
        for index, case in enumerate(cases):
            prepared = first if index == 0 else FFNCase(device, weights, case.shape)
            identity = case_id(case.shape)
            check = prepared.check()
            report["checks"][identity] = check
            if not check["passed"]:
                raise ValueError(f"numerical check failed: {identity}")
            if arguments.mode == "graph":
                with device.graph(prepared.run) as operation:
                    operation()
                    device.synchronize()
                    graph_check = prepared.check(execute=False)
                    report["checks"][identity]["graph"] = graph_check
                    if not graph_check["passed"]:
                        raise ValueError(f"graph numerical check failed: {identity}")
                    if arguments.profile:
                        device.profile(operation)
                    else:
                        samples = device.time(operation, prepared.reset)
            else:
                if arguments.profile:
                    device.profile(prepared.run)
                else:
                    samples = device.time(prepared.run, prepared.reset)
            if arguments.profile:
                report["profile_only"] = True
                return 0
            report["samples"][identity] = samples
            observation = Observation(
                case.shape.operator,
                identity,
                case.calls,
                work_for(case.shape).lower_bound_seconds(),
                statistics.median(samples),
                True,
                source_id,
            )
            report["observations"].append(asdict(observation))
            print(
                json.dumps(
                    {
                        "case": identity,
                        "rows": case.shape.rows,
                        "median_seconds": observation.measured_seconds,
                        "efficiency": observation.efficiency(),
                    }
                ),
                flush=True,
            )
        required = {
            case_id(case.shape): ExpectedCase(
                case.shape.operator, case.calls, work_for(case.shape)
            )
            for case in cases
        }
        observations = [Observation(**item) for item in report["observations"]]
        efficiency = expected_efficiencies(observations, required, source_id)
        report["expected_efficiency"] = efficiency
        report["operator_coverage_complete"] = len(cases) == sum(
            case.shape.operator == arguments.operator for case in modeled_cases()
        )
        report["operator_passed"] = (
            report["operator_coverage_complete"]
            and efficiency[arguments.operator] >= 0.60
        )
        print(
            json.dumps(
                {
                    "expected_efficiency": efficiency,
                    "operator_passed": report["operator_passed"],
                    "accepted": False,
                }
            ),
            flush=True,
        )
        return 0 if report["operator_passed"] else 1
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        try:
            device.synchronize()
        except Exception as error:
            report["synchronize_error"] = str(error)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("build")
    profiler = commands.add_parser("profile")
    profiler.add_argument("--model", type=Path, required=True)
    profiler.add_argument("--output", type=Path, required=True)
    profiler.add_argument("--operator", choices=SERVING_OPERATORS, required=True)
    profiler.add_argument("--mode", choices=("eager", "graph"), required=True)
    profiler.add_argument("--rows", type=int, required=True)
    bench = commands.add_parser("bench")
    bench.add_argument("--model", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--operator", choices=SERVING_OPERATORS, required=True)
    bench.add_argument("--mode", choices=("eager", "graph"), required=True)
    bench.add_argument("--rows", type=int, nargs="+")
    bench.add_argument("--profile", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "plan":
        print(json.dumps(plan(), indent=2))
        return 0
    if arguments.command == "build":
        from .build import build

        for name, functions in (
            ("device", ("allocate", "copy_bytes", "synchronize", "evict", "l2_bytes")),
            ("ffn", ("ffn",)),
        ):
            _, identity, _ = build(name, functions)
            print(f"{name}: {identity}")
        return 0
    if arguments.output.resolve().is_relative_to(ROOT):
        parser.error("measurement output must be external to the repository")
    if arguments.command == "profile":
        from .profile import profile

        profile(arguments)
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"accepted": False}
    try:
        result = evaluate(arguments, report)
        return 1 if "synchronize_error" in report else result
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
