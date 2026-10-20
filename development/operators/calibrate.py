import argparse
import json
import subprocess
import sys
from pathlib import Path

from .cases import load_cases
from .fixtures import file_digest
from .performance import compatible_spec, external_directory, write_report
from .timing import Sample, calibrate


def propose(root):
    cases = [case for case in load_cases() if case.purpose == "performance"]
    sessions = [root / f"session-{index}" for index in range(1, 6)]
    identities = [json.loads((path / "identity.json").read_text()) for path in sessions]
    if any(
        identity[key] != identities[0][key]
        for identity in identities
        for key in ("candidate", "accounting", "timing", "device", "mode")
    ):
        raise ValueError(
            "calibration sessions use different candidates, accounting or devices"
        )
    if identities[0]["mode"] != "calibration":
        raise ValueError("ratio proposals require calibration sessions")
    if any(
        len({identity[key] for identity in identities}) != 5
        for key in ("session", "process")
    ):
        raise ValueError("calibration requires five fresh measurement processes")
    summaries = [json.loads((path / "summary.json").read_text()) for path in sessions]
    if any(set(summary) != {case.id for case in cases} for summary in summaries):
        raise ValueError("calibration is missing fixed primary cases")
    proposed = {}
    for case in cases:
        paths = [folder / (case.id + ".json") for folder in sessions]
        reports = [json.loads(path.read_text()) for path in paths]
        for path, report, summary in zip(paths, reports, summaries, strict=True):
            if (
                file_digest(path) != summary[case.id]["report_sha256"]
                or report["case_sha256"] != case.identity
            ):
                raise ValueError(
                    "calibration report differs from its case or recorded identity"
                )
            if any(
                report[key] != reports[0][key]
                for key in (
                    "input_sha256",
                    "input_provenance",
                    "checkpoint",
                    "work",
                    "lower_seconds",
                )
            ):
                raise ValueError(
                    "calibration inputs or resource work differ across sessions"
                )
        samples = [
            [Sample(**sample) for sample in report["samples"]] for report in reports
        ]
        limit = calibrate(samples, reports[0]["lower_seconds"])
        proposed[case.id] = {
            "case_sha256": case.identity,
            "input_sha256": reports[0]["input_sha256"],
            "accounting": identities[0]["accounting"],
            "timing": identities[0]["timing"],
            "device": compatible_spec(identities[0]["device"]),
            "work": reports[0]["work"],
            "lower_seconds": reports[0]["lower_seconds"],
            "max_ratio": limit,
            "provenance": {
                "candidate": identities[0]["candidate"],
                "input": reports[0]["input_provenance"],
                "sessions": [
                    file_digest(folder / "identity.json") for folder in sessions
                ],
                "reports": [file_digest(path) for path in paths],
            },
        }
    return proposed


def run(model, output):
    root = external_directory(output)
    for index in range(1, 6):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "development.operators.measure",
                "--model",
                str(model),
                "--report",
                str(root / f"session-{index}"),
                "--calibration",
            ],
            check=True,
        )
    write_report(root / "proposed-performance-limits.json", propose(root))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Propose operator ratio limits from five independent sessions"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.model, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
