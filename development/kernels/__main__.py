import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from oh_my_vllm.kernels.specification import UPSTREAM_COMMIT

from .provenance import ROOT, case_id, source_identity, suite_identity
from .providers import MODULES, RUNTIME_MODULES, prepare
from .roofline import ExpectedCase, Observation, expected_efficiencies
from .work import work_for
from .workload import PROFILES, SERVING_OPERATORS, modeled_cases


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
    device = Device()
    checkpoint = Checkpoint(arguments.model)
    make_case = prepare(arguments.operator, device, checkpoint)
    report.update(
        {
            "operator": arguments.operator,
            "mode": arguments.mode,
            "accepted": False,
            "observations": [],
            "checks": {},
            "samples": {},
            "work": {},
            "missing_providers": sorted(set(SERVING_OPERATORS) - set(RUNTIME_MODULES)),
        }
    )
    first = make_case(cases[0].shape)
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not selected_gpu or "," in selected_gpu:
        raise ValueError("use the single-GPU wrapper")
    context = {
        "checkpoint": dict(checkpoint.identity),
        "mode": arguments.mode,
        "device": selected_gpu,
        "device_source": device.source_id,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    libraries = tuple(
        dict.fromkeys(
            (
                *getattr(first.operation, "libraries", (first.operation.library,)),
                device.library,
            )
        )
    )
    source_id, binary_ids = source_identity(
        first.operation.source_id, libraries, context, arguments.operator
    )
    native_id = first.operation.source_id
    suite_id = suite_identity()
    report["native_id"], report["suite_id"] = native_id, suite_id
    report["source_id"], report["binary_ids"] = source_id, binary_ids
    report["builds"] = [
        json.loads((binary.parent / "manifest.json").read_text())
        for binary in libraries
    ]
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
            prepared = first if index == 0 else make_case(case.shape)
            if checkpoint.identity != context["checkpoint"]:
                raise ValueError("checkpoint identity changed between cases")
            if prepared.operation.source_id != native_id:
                raise ValueError("native provider identity changed between cases")
            identity = case_id(case.shape)
            work = work_for(case.shape)
            report["work"][identity] = asdict(work)
            prepared.reset()
            check = prepared.check()
            report["checks"][identity] = check
            if not check["passed"]:
                raise ValueError(f"numerical check failed: {identity}")
            if arguments.mode == "graph":
                with device.graph(prepared.run) as operation:
                    prepared.reset()
                    operation()
                    device.synchronize()
                    graph_check = prepared.check(execute=False)
                    report["checks"][identity]["graph"] = graph_check
                    if not graph_check["passed"]:
                        raise ValueError(f"graph numerical check failed: {identity}")
                    if arguments.profile:
                        prepared.reset()
                        device.profile(operation)
                    else:
                        samples = device.time(operation, prepared.reset)
            else:
                if arguments.profile:
                    prepared.reset()
                    device.profile(prepared.run)
                else:
                    samples = device.time(prepared.run, prepared.reset)
            current_id, _ = source_identity(
                native_id, libraries, context, arguments.operator
            )
            if current_id != source_id or suite_identity() != suite_id:
                raise ValueError(
                    "provider or reference source changed during evaluation"
                )
            if arguments.profile:
                report["profile_only"] = True
                return 0
            report["samples"][identity] = samples
            observation = Observation(
                case.shape.operator,
                identity,
                case.calls,
                work.lower_bound_seconds(),
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
    audit = commands.add_parser("audit")
    audit.add_argument("--reports", type=Path, nargs="+", required=True)
    audit.add_argument("--output", type=Path, required=True)
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

        for name, functions in MODULES.items():
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
        if arguments.command == "audit":
            from .acceptance import audit

            report.update(audit(arguments.reports))
            print(json.dumps(report, indent=2))
            return 0 if report["accepted"] else 1
        result = evaluate(arguments, report)
        return 1 if "synchronize_error" in report else result
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
