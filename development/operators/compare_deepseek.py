"""Plan or compare one strict BF16 TP4 up call with a public reference.

Plan uses the standard library only. Numeric workers run serially and exit.
Neither mode measures performance or opens the model-level acceptance gate.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from development.operators.grouped_deepgemm import (
    ComparisonError,
    canonical_hash,
    compare_outputs,
    digest,
    read_case,
    verify_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
STATES = frozenset(
    (
        "planned",
        "missing",
        "unsupported",
        "semantic_mismatch",
        "numeric_failed",
        "unmeasured",
    )
)


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_document(spec):
    path = Path(spec["file"])
    if digest(path) != spec["sha256"]:
        raise ComparisonError("missing", f"document digest mismatch: {path.name}")
    return read_json(path)


def external_directory(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("comparison artifacts must be outside the repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_package_sources(identity):
    document = checked_document(identity["package_sources"])
    package = Path(identity["library"]).resolve().parent
    files = document["files"]
    actual = {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file() and path.resolve() != Path(identity["library"]).resolve()
    }
    if (
        not files
        or set(files) != actual
        or document["source_revisions"] != identity["source_revisions"]
    ):
        raise ComparisonError("missing", "reference package source inventory differs")
    for name, expected in files.items():
        path = (package / name).resolve()
        if not path.is_relative_to(package) or digest(path) != expected:
            raise ComparisonError(
                "missing", f"reference package source differs: {name}"
            )
    return {
        "manifest_sha256": identity["package_sources"]["sha256"],
        "files_sha256": canonical_hash(files),
        "file_count": len(files),
    }


def implementation_identities(candidate, reference):
    native = read_json(candidate)
    public = read_json(reference)
    if digest(native["library"]) != native["binary_id"]:
        raise ComparisonError("missing", "candidate binary differs from its manifest")
    if public["status"] != "HOST_BUILD_AND_CPU_IMPORT_PASSED_GPU_UNVERIFIED":
        raise ComparisonError("missing", "final public reference handoff is absent")
    if digest(public["native"]["path"]) != public["native"]["sha256"]:
        raise ComparisonError("missing", "reference binary differs from its manifest")
    if public["grouped_fp4_source_findings"]["recipe"] != [1, 1, 32]:
        raise ComparisonError("unsupported", "public reference does not declare G32")
    reference_identity = {
        "manifest": str(reference),
        "manifest_sha256": digest(reference),
        "library": public["native"]["path"],
        "binary_sha256": public["native"]["sha256"],
        "upstream_commit": public["source_revisions"]["DeepGEMM"],
        "source_revisions": public["source_revisions"],
        "package_sources": public["package_sources"],
    }
    reference_identity["package_source_summary"] = verify_package_sources(
        reference_identity
    )
    return {
        "candidate": {
            "manifest": str(candidate),
            "manifest_sha256": digest(candidate),
            "library": native["library"],
            "binary_sha256": native["binary_id"],
            "source_id": native["source_id"],
        },
        "reference": reference_identity,
    }


def plan_comparison(spec, candidate, reference):
    required = checked_document(spec["required_inventory"])
    ids = required["required_case_ids"]
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids):
        raise ComparisonError(
            "missing", "independent required_case_ids are empty or duplicate"
        )
    for source in required["sources"]:
        checked_document(source)
    rows = {
        case_id: {
            "case_id": case_id,
            "state": "missing",
            "reason": "required case not supplied",
        }
        for case_id in ids
    }
    if len(spec["cases"]) != 1:
        raise ComparisonError("unsupported", "select exactly one real TP4 up case")
    item = spec["cases"][0]
    case_id = item["case_id"]
    if case_id not in rows:
        raise ComparisonError(
            "missing", "selected case is absent from independent required inventory"
        )
    identities = None
    case = None
    try:
        case = read_case(item)
        identities = implementation_identities(candidate, reference)
        verify_inputs(case)
        rows[case_id] = {
            "case_id": case_id,
            "state": "planned",
            "reason": "CPU plan only",
        }
    except (ComparisonError, OSError, KeyError, ValueError) as error:
        rows[case_id] = {
            "case_id": case_id,
            "state": getattr(error, "state", "missing"),
            "reason": str(error),
        }
    summary = {
        "required_case_ids": ids,
        "measured_case_ids": [],
        "required_inventory_sha256": spec["required_inventory"]["sha256"],
        "required_scope": required["scope"],
        "coverage_complete": required["coverage_complete"],
        "cases": list(rows.values()),
        "identities": identities,
        "model_level_gate_open": False,
        "performance": "unmeasured",
    }
    return case, summary


def output_record(case, output):
    path = output / "output.bf16"
    if not path.is_file() or path.stat().st_size != case.rows * 576 * 2:
        raise ComparisonError(
            "numeric_failed", "worker did not save complete BF16 output"
        )
    return {"output_sha256": digest(path), "output_bytes": path.stat().st_size}


def wait_worker(command, log, timeout):
    def cancel(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, cancel)
    process = None
    try:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        return process.wait(timeout=timeout)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)


def run_worker(
    side, input_path, candidate, reference, case, identities, output, timeout
):
    side_output = output / side
    side_output.mkdir()
    command = [
        str(ROOT / "scripts/with-env.sh"),
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import runpy,sys;sys.path[:0]=sys.argv[1:3];sys.argv=sys.argv[3:];"
        "runpy.run_module('development.operators.compare_deepseek',run_name='__main__')",
        str(ROOT / "python"),
        str(ROOT),
        "compare_deepseek",
        "--worker",
        side,
        "--cases",
        str(input_path),
        "--candidate",
        str(candidate),
        "--reference",
        str(reference),
        "--output",
        str(side_output),
    ]
    with (side_output / "worker.log").open("x") as log:
        code = wait_worker(command, log, timeout)
    if code:
        raise ComparisonError(
            "unmeasured", f"{side} worker exited {code}; retain worker.log"
        )
    receipt = read_json(side_output / "receipt.json")
    if (
        receipt["case_sha256"] != case.case_sha256
        or receipt["side"] != side
        or receipt["implementation"] != identities[side]
        or not receipt["inputs_unchanged"]
        or receipt["output"] != output_record(case, side_output)
    ):
        raise ComparisonError(
            "numeric_failed", f"{side} worker identity or output differs"
        )
    return side_output / "output.bf16"


def numeric_comparison(
    case, summary, input_path, candidate, reference, output, timeout
):
    selected = next(row for row in summary["cases"] if row["case_id"] == case.case_id)
    if selected["state"] != "planned":
        return summary
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not os.environ.get("OH_MY_VLLM_RUN_ID") or len(visible) != 1 or not visible[0]:
        selected.update(
            state="unmeasured",
            reason="numeric mode needs coordinator authorization and one GPU lease",
        )
        return summary
    try:
        paths = [
            run_worker(
                side,
                input_path,
                candidate,
                reference,
                case,
                summary["identities"],
                output,
                timeout,
            )
            for side in ("candidate", "reference")
        ]
        selected.update(compare_outputs(case, *paths))
    except (
        ComparisonError,
        OSError,
        KeyError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as error:
        selected.update(state=getattr(error, "state", "unmeasured"), reason=str(error))
    return summary


def operand_hashes(operands):
    import hashlib

    import torch

    tensors = [operands["activation"], operands["scales"], operands["counts"]]
    tensors.extend(value for pair in operands["weights"] for value in pair)
    return [
        hashlib.sha256(
            value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for value in tensors
    ]


def worker_main(options):
    if options.output.resolve().is_relative_to(ROOT):
        raise ValueError("worker artifacts must be outside the repository")
    from development.operators.grouped_deepgemm import (
        load_operands,
        prepare_deepgemm,
        prepare_native,
    )

    case, summary = plan_comparison(
        read_json(options.cases), options.candidate, options.reference
    )
    selected = next(row for row in summary["cases"] if row["case_id"] == case.case_id)
    if selected["state"] != "planned":
        raise ComparisonError(selected["state"], selected["reason"])
    if not os.environ.get("OH_MY_VLLM_RUN_ID"):
        raise ComparisonError("unmeasured", "worker requires the outer GPU wrapper")
    import torch

    if torch.cuda.device_count() != 1:
        raise ComparisonError("unsupported", "expose exactly the coordinator-owned GPU")
    torch.cuda.set_device(0)
    identities = summary["identities"]
    identity = identities[options.worker]
    if options.worker == "candidate":
        from tvm_ffi import load_module

        module = load_module(identity["library"])
        prepare = prepare_native
    else:
        verify_package_sources(identity)
        package = Path(identity["library"]).parent
        os.environ["DG_JIT_CACHE_DIR"] = str(options.output / "jit")
        sys.path.insert(0, str(package.parent))
        import deep_gemm as module

        if (
            Path(module.__file__).resolve().parent != package
            or Path(module._C.__file__).resolve() != Path(identity["library"]).resolve()
            or digest(module._C.__file__) != identity["binary_sha256"]
        ):
            raise ComparisonError("missing", "loaded public reference identity differs")
        prepare = prepare_deepgemm
    operands = load_operands(case)
    before = operand_hashes(operands)
    invoke, owners = prepare(case, operands, module)
    result = invoke()
    torch.cuda.synchronize()
    unchanged = before == operand_hashes(operands)
    if not unchanged:
        raise ComparisonError("numeric_failed", "input bytes changed during comparison")
    payload = result.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    with (options.output / "output.bf16").open("xb") as stream:
        stream.write(payload)
    verify_inputs(case)
    if options.worker == "reference":
        verify_package_sources(identity)
    write_json(
        options.output / "receipt.json",
        {
            "side": options.worker,
            "case_sha256": case.case_sha256,
            "implementation": identity,
            "inputs_unchanged": unchanged,
            "output": output_record(case, options.output),
            "pid": os.getpid(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "owner_groups": len(operands["weights"]),
            "performance": "unmeasured",
            "jit_binaries": {
                str(path.relative_to(options.output)): digest(path)
                for path in options.output.rglob("*.cubin")
            },
        },
    )
    del owners


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=int, default=300)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--numeric-only", action="store_true")
    mode.add_argument(
        "--worker", choices=("candidate", "reference"), help=argparse.SUPPRESS
    )
    options = parser.parse_args(argv)
    if options.worker:
        worker_main(options)
        return 0
    output = external_directory(options.output)
    spec = read_json(options.cases)
    write_json(output / "input.json", spec)
    case, summary = plan_comparison(spec, options.candidate, options.reference)
    if options.numeric_only and case:
        summary = numeric_comparison(
            case,
            summary,
            output / "input.json",
            options.candidate,
            options.reference,
            output,
            options.worker_timeout,
        )
    write_json(output / "summary.json", summary)
    with (output / "results.jsonl").open("x") as stream:
        for row in summary["cases"]:
            if row["state"] not in STATES:
                raise ValueError("unrecognized comparison state")
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    selected = next(
        row for row in summary["cases"] if row["case_id"] == spec["cases"][0]["case_id"]
    )
    print(json.dumps(selected, allow_nan=False))
    return 0 if selected["state"] in ("planned", "unmeasured") else 2


if __name__ == "__main__":
    raise SystemExit(main())
