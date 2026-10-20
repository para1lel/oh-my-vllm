import argparse
import json
from pathlib import Path

import numpy as np

from .capture import run
from .cases import load_case_fixture, load_cases
from .compare import (
    BASELINE_SHA256,
    CASE_LIMITS,
    LIMITS,
    compare_endpoints,
    difference,
    load_reference,
    validate_case,
)


def added_case(model, case):
    from .capture_cases import run

    fixture = load_case_fixture(case)
    metadata, outputs = run(model, case, fixture)
    metrics = compare_endpoints(case, outputs, fixture.expected)
    report = dict(
        actual=metadata,
        reference=fixture.manifest["reference"],
        metrics=metrics,
        limits=CASE_LIMITS[case.id],
        fixture_sha256=case.fixture_sha256,
    )
    passed = all(row["passed"] for rows in metrics.values() for row in rows)
    return report, outputs, passed


def short_case(model, mode):
    reference, expected = load_reference()
    metadata, outputs = run(
        model, decode_token=reference["case"]["decode_token"], mode=mode
    )
    validate_case(
        metadata["case"],
        reference["case"],
        metadata["checkpoint"],
        reference["checkpoint"],
    )
    metrics = difference(outputs[mode], expected[mode])
    report = dict(
        mode=mode,
        actual=metadata,
        reference=reference,
        metrics=metrics,
        limits=LIMITS,
        reference_sha256=BASELINE_SHA256,
    )
    return report, outputs, metrics["passed"]


def main():
    cases = {case.id: case for case in load_cases()}
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--case", choices=("short", *cases), default="short")
    args = parser.parse_args()
    if (
        args.report.suffix != ".json"
        or args.report.exists()
        or args.report.with_suffix(".npz").exists()
    ):
        parser.error("use a new external JSON report path")
    if args.report.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("use external artifacts")
    if args.case != "short" and args.mode != "decode":
        parser.error("added cases always compare every fixed endpoint")
    report, outputs, passed = (
        short_case(args.model, args.mode)
        if args.case == "short"
        else added_case(args.model, cases[args.case])
    )
    with args.report.with_suffix(".npz").open("xb") as destination:
        np.savez(destination, **outputs)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["metrics"], sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
