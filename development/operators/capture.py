import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from oh_my_vllm.worker.device import Device

from .budgets import propose
from .cases import load_cases
from .fixtures import file_digest, load_fixture
from .numerics import compare
from .run import prepare


def measure(case, fixture, model):
    device = Device()
    with prepare(case, device, fixture, model=model) as prepared:
        prepared.reset()
        prepared.invoke()
        actual = prepared.observe()
    fields = {
        name: {
            "kind": "float",
            "normalized_l2": 1e100,
            "max_abs": 1e100,
            "rms_floor": 1e-6,
            "near_zero_max_abs": 1e100,
        }
        if value.dtype.kind == "f"
        else {"kind": "exact"}
        for name, value in fixture.expected.items()
    }
    policy = propose(case, fixture)
    result = compare(actual, fixture, {**policy, "fields": fields})
    exact = {
        name: {
            "actual": actual[name].tolist(),
            "expected": fixture.expected[name].tolist(),
        }
        for name in result.failures
        if name in actual and actual[name].size <= 100
    }
    ranges = {
        name: {
            "max_abs": float(np.max(np.abs(value))) if value.size else 0,
            "rms": float(np.sqrt(np.mean(value**2))) if value.size else 0,
        }
        for name, value in fixture.expected.items()
        if value.dtype.kind == "f"
    }
    return {
        "case_sha256": case.identity,
        "fixture_sha256": case.fixture_sha256,
        "metrics": result.metrics,
        "contract_failures": result.failures,
        "exact_values": exact,
        "reference_ranges": ranges,
    }


def main():
    cases = [case for case in load_cases() if case.purpose == "numerics"]
    parser = argparse.ArgumentParser(
        description="Capture numerical diagnostics without acceptance limits"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", choices=[case.id for case in cases], action="append")
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=False)
    for case in cases:
        if args.case is not None and case.id not in args.case:
            continue
        root = args.fixtures / case.id
        bound = replace(case, fixture_sha256=file_digest(root / "manifest.json"))
        fixture = load_fixture(bound, root)
        report = measure(bound, fixture, args.model)
        with (args.report / (case.id + ".json")).open("x") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
        print(f"captured {case.id}: {report['contract_failures']}", flush=True)


if __name__ == "__main__":
    main()
