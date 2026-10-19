import argparse
import hashlib
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from development.kernels.workload import numerical_cases

from .cases import (
    HTTP_CONFIGURATIONS,
    TRACE_POLICY,
    CoreCase,
    from_record,
    http_requests,
    manifest,
)
from .public import PUBLIC_NAMES, Unsupported


def own_factory(name):
    module, function = name.split(":")
    return getattr(importlib.import_module(module), function)


def numerical_manifest(operator):
    result = set()
    for shape in numerical_cases():
        if shape.operator != operator or not shape.rows:
            continue
        queries = shape.rows // shape.batch
        result.add(
            CoreCase(
                operator,
                (queries,) * shape.batch,
                (max(queries, shape.context),) * shape.batch,
                (shape.position,) * shape.batch,
                "recurrent" if queries <= 5 else "prefill",
            )
        )
    return sorted(
        result, key=lambda case: (case.queries, case.contexts, case.positions)
    )


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def check_provider(prepared, inputs, measurement, expected):
    import torch

    from .checks import check_outputs

    executable, graph = measurement.executable(prepared)
    measurement.invoke(prepared.reset)
    measurement.invoke(executable)
    torch.cuda.synchronize()
    checks = check_outputs(prepared, inputs, expected)
    return checks, executable, graph


def verify_small_cases(arguments, factory, measurement, evidence):
    from .checks import independent_reference
    from .identity import input_identity, provider_identity, public_binary_identity
    from .inputs import make_inputs
    from .public import prepare

    cases = numerical_manifest(arguments.operator)
    qualifier = CoreCase("gdn_attention", (705,), (705,), (0,), "prefill")
    if (
        arguments.operator == "gdn_attention"
        and "flashinfer_gdn_prefill_cudnn" in arguments.public
        and any(not case.verify for case in cases)
    ):
        cases.append(qualifier)
    for case in cases:
        inputs = make_inputs(case)
        bound_inputs = input_identity(inputs)
        expected = independent_reference(inputs)
        own = factory(case, inputs)
        bound_own = provider_identity(own)
        own_checks, _, _ = check_provider(own, inputs, measurement, expected)
        if provider_identity(own) != bound_own:
            raise ValueError(
                "own sources or binaries changed during the numerical check"
            )
        if input_identity(inputs) != bound_inputs:
            raise ValueError("own provider changed the small-case input tensors")
        row = {
            "case": asdict(case),
            "own": own_checks,
            "own_identity": bound_own,
            "inputs": bound_inputs,
            "public": {},
            "supplemental_cudnn_chain_qualifier": case == qualifier,
        }
        evidence[case.identity] = row
        if not own_checks["passed"]:
            raise ValueError(f"own small numerical check failed: {case.identity}")
        for name in arguments.public:
            if case == qualifier and name != "flashinfer_gdn_prefill_cudnn":
                continue
            try:
                public = prepare(name, case, inputs, arguments.mode)
                checks, _, _ = check_provider(public, inputs, measurement, expected)
                row["public"][name] = {
                    "status": "passed" if checks["passed"] else "failed",
                    "checks": checks,
                    "metadata": public.metadata,
                    "binary_identity": public_binary_identity(),
                }
            except (Unsupported, NotImplementedError) as error:
                row["public"][name] = {"status": "unsupported", "reason": str(error)}
            except Exception as error:
                row["public"][name] = {
                    "status": "error",
                    "reason": f"{type(error).__name__}: {error}",
                }
            if input_identity(inputs) != bound_inputs:
                raise ValueError(f"{name} changed the common small-case input tensors")
    return evidence


def public_numerically_eligible(name, case, evidence, metadata=None):
    selected = [
        row["public"][name]
        for row in evidence.values()
        if from_record(row["case"]).verify == case.verify and name in row["public"]
    ]
    errors = [row["reason"] for row in selected if row["status"] == "error"]
    if errors:
        raise ValueError(f"{name} small-case evaluation failed: {'; '.join(errors)}")
    supported = [row for row in selected if row["status"] != "unsupported"]
    if not supported or any(row["status"] != "passed" for row in supported):
        return False
    if name == "flashinfer_gdn_prefill_cudnn":
        plans = (metadata or {}).get("cudnn_plans", ())
        return bool(plans) and all(
            not plan["batch_invariant"]
            and plan["engine"] in {"exact piece chain", "exact value split"}
            and (plan["default_pieces"] > 0 or plan["default_value_split"])
            for plan in plans
        )
    return True


def evaluate_case(record, arguments, factory, measurement, evidence):
    import torch

    from .checks import check_outputs, snapshot
    from .identity import input_identity, provider_identity, public_binary_identity
    from .inputs import make_inputs
    from .public import prepare

    case = from_record(record["case"])
    inputs = make_inputs(case)
    result = {**record, "inputs": input_identity(inputs), "public": {}, "passed": False}
    own = factory(case, inputs)
    result["own_identity"] = provider_identity(own)
    own_executable, own_graph = measurement.executable(own)
    measurement.invoke(own.reset)
    measurement.invoke(own_executable)
    torch.cuda.synchronize()
    expected = snapshot(own, inputs)
    result["own_checks"] = check_outputs(own, inputs, expected)
    if not result["own_checks"]["passed"]:
        raise ValueError("own provider changed read slots or guard slots")
    result["own"] = {**measurement.time(own, own_executable), "metadata": own.metadata}
    eligible = []
    for name in arguments.public:
        try:
            public = prepare(name, case, inputs, arguments.mode)
            checks, executable, graph = check_provider(
                public, inputs, measurement, expected
            )
            checked_small = public_numerically_eligible(
                name, case, evidence, public.metadata
            )
            entry = {
                "status": "eligible"
                if checks["passed"] and checked_small
                else "numerical_failure",
                "metadata": public.metadata,
                "checks_against_same_input_own": checks,
                "independent_small_checks": checked_small,
            }
            result["public"][name] = entry
            if entry["status"] == "eligible":
                binaries_before = public_binary_identity()
                entry.update(measurement.time(public, executable))
                binaries_after = public_binary_identity()
                if binaries_before != binaries_after:
                    raise ValueError(
                        "public compiled binaries changed during measurement"
                    )
                entry["binary_identity"] = binaries_after
                eligible.append((entry["median_seconds"], name))
            del graph
        except (Unsupported, NotImplementedError) as error:
            result["public"][name] = {"status": "unsupported", "reason": str(error)}
        except Exception as error:
            result["public"][name] = {
                "status": "error",
                "reason": f"{type(error).__name__}: {error}",
            }
    if eligible:
        seconds, name = min(eligible)
        result["fastest_eligible_public"] = name
        result["own_over_public"] = result["own"]["median_seconds"] / seconds
        result["passed"] = result["own"]["median_seconds"] <= seconds
        if any(row["status"] == "error" for row in result["public"].values()):
            result["passed"] = False
            result["failure"] = "public candidate evaluation failed"
    else:
        result["failure"] = "no numerically eligible public baseline"
    if input_identity(inputs) != result["inputs"]:
        raise ValueError("a provider changed the common input tensors")
    if provider_identity(own) != result["own_identity"]:
        raise ValueError("own provider sources or binaries changed during measurement")
    del own_graph
    return result


def run(arguments):
    from .identity import (
        environment_identity,
        files_identity,
        public_binary_identity,
        verify_source_pins,
    )
    from .measure import Measurement, require_gpu_lock

    gpu = require_gpu_lock()
    if os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a") != "10.0a":
        raise ValueError("native cores require TVM_FFI_CUDA_ARCH_LIST=10.0a")
    verify_source_pins()
    root = Path(__file__).resolve().parents[2]
    output = arguments.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("reports must be outside the repository")
    bound_records = files_identity([*arguments.trace, *arguments.http_result])
    all_records = manifest(arguments.trace, arguments.http_result)
    if files_identity(bound_records) != bound_records:
        raise ValueError("trace or HTTP result files changed during case selection")
    records = [
        record
        for record in all_records
        if record["case"]["operator"] == arguments.operator
        and (not arguments.case or record["id"] in arguments.case)
    ]
    if not records or (
        arguments.case and set(arguments.case) != {record["id"] for record in records}
    ):
        raise ValueError("selected case IDs do not match the manifest")
    measurement = Measurement(arguments.mode)
    configurations = set(http_requests(arguments.http_result).values())
    report = {
        "report_version": 1,
        "boundary": "attention core with all required adapters and state exports",
        "full_serving_boundary_evidence": False,
        "operator": arguments.operator,
        "mode": arguments.mode,
        "scope": "one operator and one execution mode",
        "environment": environment_identity(gpu),
        "measurement": measurement.policy(),
        "trace_files": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in arguments.trace
        },
        "http_result_files": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in arguments.http_result
        },
        "trace_selection": TRACE_POLICY,
        "selected_manifest": records,
        "manifest_sha256": hashlib.sha256(
            json.dumps(all_records, sort_keys=True).encode()
        ).hexdigest(),
        "coverage": {
            "modeled": not arguments.case,
            "actual_http_representatives": bool(arguments.trace) and not arguments.case,
            "http_configurations": sorted(configurations),
            "all_24_http_configurations": configurations == HTTP_CONFIGURATIONS,
            "all_trace_shapes": False,
            "http_result_files": len(arguments.http_result),
            "public_candidates_complete": set(arguments.public) == set(PUBLIC_NAMES),
        },
        "numerical_evidence": {},
        "cases": [],
        "passed": False,
    }
    save(output, report)
    try:
        factory = own_factory(arguments.own)
        evidence = report["numerical_evidence"]
        verify_small_cases(arguments, factory, measurement, evidence)
        save(output, report)
        for record in records:
            result = evaluate_case(record, arguments, factory, measurement, evidence)
            report["cases"].append(result)
            save(output, report)
            print(
                json.dumps(
                    {
                        "id": result["id"],
                        "passed": result["passed"],
                        "own_over_public": result.get("own_over_public"),
                    }
                ),
                flush=True,
            )
        report["selected_cases_passed"] = bool(report["cases"]) and all(
            item["passed"] for item in report["cases"]
        )
        report["passed"] = (
            report["selected_cases_passed"]
            and report["coverage"]["modeled"]
            and report["coverage"]["public_candidates_complete"]
            and report["coverage"]["actual_http_representatives"]
            and report["coverage"]["all_24_http_configurations"]
        )
        report["public_binaries"] = public_binary_identity()
        if files_identity(bound_records) != bound_records:
            raise ValueError("trace or HTTP result files changed during measurement")
        for item in [*report["numerical_evidence"].values(), *report["cases"]]:
            bound = item["own_identity"]
            if any(
                files_identity(bound[kind]) != bound[kind]
                for kind in ("sources", "binaries")
            ):
                raise ValueError("own sources or binaries changed during the full run")
            for public in item["public"].values():
                binaries = public.get("binary_identity", {})
                if files_identity(binaries) != binaries:
                    raise ValueError(
                        "public compiled binaries changed during the full run"
                    )
        after = environment_identity(gpu)
        if (
            after["sources"] != report["environment"]["sources"]
            or after["packages"] != report["environment"]["packages"]
            or after["header_contents"] != report["environment"]["header_contents"]
        ):
            raise ValueError("comparison sources changed during measurement")
        save(output, report)
        return 0 if report["passed"] else 1
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        save(output, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--trace", type=Path, action="append", default=[])
    plan.add_argument("--http-result", type=Path, action="append", default=[])
    evaluate = subparsers.add_parser("run")
    evaluate.add_argument(
        "--own", required=True, help="MODULE:FACTORY; accepts CoreCase and CoreInputs"
    )
    evaluate.add_argument(
        "--operator", choices=("fa_attention", "gdn_attention"), required=True
    )
    evaluate.add_argument("--mode", choices=("eager", "graph"), required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--trace", type=Path, action="append", default=[])
    evaluate.add_argument("--http-result", type=Path, action="append", default=[])
    evaluate.add_argument("--case", action="append", default=[])
    evaluate.add_argument(
        "--public", nargs="+", choices=PUBLIC_NAMES, default=list(PUBLIC_NAMES)
    )
    arguments = parser.parse_args()
    if arguments.command == "plan":
        print(json.dumps(manifest(arguments.trace, arguments.http_result), indent=2))
        return 0
    return run(arguments)


if __name__ == "__main__":
    sys.exit(main())
