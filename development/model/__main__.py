import argparse
import json
from pathlib import Path

import numpy as np

from .capture import run
from .compare import BASELINE_SHA256, LIMITS, difference, load_reference, validate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    args = parser.parse_args()
    if (
        args.report.suffix != ".json"
        or args.report.exists()
        or args.report.with_suffix(".npz").exists()
    ):
        parser.error("use a new external JSON report path")
    if args.report.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("use external artifacts")
    reference, expected = load_reference(args.reference)
    metadata, outputs = run(
        args.model, decode_token=reference["case"]["decode_token"], mode=args.mode
    )
    validate_case(
        metadata["case"],
        reference["case"],
        metadata["checkpoint"],
        reference["checkpoint"],
    )
    metrics = difference(outputs[args.mode], expected[args.mode])
    report = dict(
        mode=args.mode,
        actual=metadata,
        reference=reference,
        metrics=metrics,
        limits=LIMITS,
        reference_sha256=BASELINE_SHA256,
    )
    with args.report.with_suffix(".npz").open("xb") as destination:
        np.savez(destination, **outputs)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
