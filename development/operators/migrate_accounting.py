import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .accounting import lower_bound, work
from .cases import load_cases
from .performance import accounting_identity

FIELDS = ("accounting", "work", "lower_seconds", "max_ratio")
MIGRATION = "necessary_work_migration"


def migrate(limits):
    result = deepcopy(limits)
    identity = accounting_identity()
    cases = [case for case in load_cases() if case.purpose == "performance"]
    if set(result["performance"]) != {case.id for case in cases}:
        raise ValueError("performance limits must cover all fixed cases")
    comparisons = []
    for case in cases:
        entry = result["performance"][case.id]
        if entry["accounting"]["version"] != "operator-reference-v3":
            raise ValueError("migration requires operator-reference-v3 limits")
        if entry["case_sha256"] != case.identity:
            raise ValueError("case identity changed before accounting migration")
        previous = {name: deepcopy(entry[name]) for name in FIELDS}
        maximum = previous["max_ratio"] * previous["lower_seconds"]
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("previous absolute latency limit must be positive")
        seconds = lower_bound(
            case, dict(name="NVIDIA B200", l2_bytes=126 * 1024 * 1024)
        )
        ratio = maximum / seconds
        while ratio * seconds > maximum:
            ratio = math.nextafter(ratio, -math.inf)
        entry.update(
            accounting=identity,
            work=asdict(work(case)),
            lower_seconds=seconds,
            max_ratio=ratio,
        )
        entry["provenance"][MIGRATION] = dict(
            previous=previous,
            maximum_seconds=maximum,
            rule="preserve the absolute latency limit; round downward if needed",
        )
        comparisons.append(
            dict(
                case=case.id,
                previous_reference_seconds=previous["lower_seconds"],
                reference_seconds=seconds,
                previous_maximum_seconds=maximum,
                maximum_seconds=ratio * seconds,
            )
        )
    return result, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    options = parser.parse_args()
    limits, comparisons = migrate(json.loads(options.input.read_text()))
    with options.comparison.open("x") as report:
        report.write(json.dumps(comparisons, indent=2) + "\n")
    options.output.write_text(json.dumps(limits, indent=2) + "\n")


if __name__ == "__main__":
    main()
