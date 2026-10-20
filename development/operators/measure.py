import argparse
import gc
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .accounting import lower_bound, work
from .cases import load_cases
from .fixtures import file_digest
from .performance import (
    accounting_identity,
    compatible_spec,
    device_spec,
    external_directory,
    input_provenance,
    prepare_performance,
    snapshot,
    timing_identity,
    write_report,
)
from .timing import accept, interval, measure
from .validation import LIMITS_PATH, candidate_identity, load_limits


def session(model, report, *, calibration=False, selected=None):
    from oh_my_vllm.operators.runtime.device import Device

    root = external_directory(report)
    cases = [
        case
        for case in load_cases()
        if case.purpose == "performance" and (selected is None or case.id in selected)
    ]
    if selected is not None and not calibration:
        raise ValueError("acceptance requires every fixed performance case")
    spec = device_spec()
    candidate = candidate_identity()
    accounting = accounting_identity()
    timing = timing_identity()
    limits = None if calibration else load_limits("performance")
    limits_hash = None if calibration else file_digest(LIMITS_PATH)
    identity = dict(
        candidate=candidate,
        accounting=accounting,
        timing=timing,
        device=spec,
        mode="calibration" if calibration else "acceptance",
        session=str(uuid4()),
        process=os.getpid(),
    )
    write_report(root / "identity.json", identity)
    results = {}
    for case in cases:
        before = snapshot(spec)
        device = Device()
        with prepare_performance(case, device, model) as prepared:
            seconds = lower_bound(case, spec)
            resource = asdict(work(case))
            if limits is not None:
                approved = limits[case.id]
                if (
                    approved["accounting"] != accounting
                    or approved["timing"] != timing
                    or approved["device"] != compatible_spec(spec)
                    or approved["input_sha256"] != prepared.input_identity
                    or approved["work"] != resource
                    or approved["lower_seconds"] != seconds
                ):
                    raise ValueError(
                        "performance input, accounting, or device identity differs"
                    )
                samples, result = accept(prepared, seconds, approved["max_ratio"])
                decision = asdict(result)
            else:
                samples = measure(prepared)
                median, low, high = interval(
                    [sample.gpu_seconds / seconds for sample in samples]
                )
                decision = dict(
                    median=median, lower=low, upper=high, decision="calibration"
                )
            after = snapshot(spec)
            prepared.weights.checkpoint.verify_identity()
            payload = dict(
                case=case.definition(),
                case_sha256=case.identity,
                input_sha256=prepared.input_identity,
                input_provenance=input_provenance(prepared.fixture),
                checkpoint=prepared.weights.identity,
                work=resource,
                lower_seconds=seconds,
                setup=prepared.setup,
                samples=[asdict(sample) for sample in samples],
                result=decision,
                telemetry=dict(before=before, after=after),
                limits_sha256=limits_hash,
            )
            sha = write_report(root / (case.id + ".json"), payload)
            results[case.id] = dict(report_sha256=sha, decision=decision["decision"])
            print(
                f"{case.id}: {decision['decision']} ratio={decision['median']:.4f}",
                flush=True,
            )
        del prepared, device
        gc.collect()
    if candidate_identity() != candidate or accounting_identity() != accounting:
        raise ValueError("candidate or measurement source changed during the session")
    if limits_hash is not None and file_digest(LIMITS_PATH) != limits_hash:
        raise ValueError("approved limits changed during performance acceptance")
    snapshot(spec)
    write_report(root / "summary.json", results)
    return calibration or all(value["decision"] == "pass" for value in results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Collect one fixed operator performance session"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.id for case in load_cases() if case.purpose == "performance"],
    )
    args = parser.parse_args()
    if args.case is not None and (
        not args.calibration or len(set(args.case)) != len(args.case)
    ):
        parser.error("only diagnostic calibration can select distinct fixed cases")
    return (
        0
        if session(
            args.model, args.report, calibration=args.calibration, selected=args.case
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
