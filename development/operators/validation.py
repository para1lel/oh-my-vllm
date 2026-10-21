import json
from pathlib import Path

import numpy as np

from .cases import digest, load_cases
from .fixtures import file_digest, load_fixture
from .numerics import compare

LIMITS_PATH = Path(__file__).with_name("limits.json")


def candidate_identity():
    from oh_my_vllm.operators.runtime.native import cache_directory, source_fingerprint

    from development.model.capture import runtime_identity

    identity = runtime_identity()
    for name in ("device", "ffn", "fa", "gdn", "sample", "mtp", "vision"):
        manifest = json.loads((cache_directory() / f"{name}.json").read_text())
        if manifest["source_fingerprint"] != source_fingerprint(name):
            raise ValueError("native module differs from its current source")
        if file_digest(manifest["library"]) != manifest["binary_id"]:
            raise ValueError("native binary differs from its build identity")
        identity["native"][name] = {
            key: manifest[key]
            for key in ("source_id", "binary_id", "source_fingerprint")
        }
    root = Path(__file__).resolve().parent
    identity["harness"] = digest(
        {
            str(path.relative_to(root)): file_digest(path)
            for path in sorted(root.rglob("*.py"))
        }
    )
    return identity


def load_limits(purpose):
    limits = json.loads(LIMITS_PATH.read_text())
    if limits.get("version") != "operator-limits-v1":
        raise ValueError("unsupported operator limit version")
    selected = limits[purpose]
    cases = [case for case in load_cases() if case.purpose == purpose]
    if set(selected) != {case.id for case in cases}:
        raise ValueError("approved limits do not cover every fixed case")
    for case in cases:
        entry = selected[case.id]
        if entry["case_sha256"] != case.identity:
            raise ValueError("approved limits differ from the case identity")
        if purpose == "numerics" and entry["fixture_sha256"] != case.fixture_sha256:
            raise ValueError("approved limits differ from the fixture identity")
    return selected


def check_dependent_inputs(prepared, fixture, limits):
    if prepared.case.boundary != "ffn" and not (
        prepared.case.boundary == "gdn" and prepared.case.mode == "graph"
    ):
        return {}
    failures = {}
    for cycle in range(3):
        prepared.reset()
        for name in ("value", "residual"):
            prepared.device.module.zero(prepared.inputs[name])
        prepared.invoke()
        if prepared.case.boundary == "ffn":
            for name, tensor in prepared.outputs.items():
                if np.any(prepared.download(tensor)):
                    failures[f"zero.{cycle}.{name}"] = (
                        "zero input produced nonzero output"
                    )
        prepared.reset()
        prepared.invoke()
        restored = compare(prepared.observe(), fixture, limits)
        if not restored.passed:
            failures[f"restored.{cycle}"] = restored.failures
    return failures


def numerical(model, report):
    from oh_my_vllm.operators.runtime.device import Device

    from .run import prepare

    limits = load_limits("numerics")
    identity = file_digest(LIMITS_PATH)
    candidate = candidate_identity()
    report = Path(report).resolve()
    repository = Path(__file__).resolve().parents[2]
    if report == repository or repository in report.parents:
        raise ValueError("keep numerical reports outside the repository")
    report.mkdir(parents=True, exist_ok=False)
    results = {}
    for case in load_cases():
        if case.purpose != "numerics":
            continue
        fixture = load_fixture(case)
        device = Device()
        with prepare(case, device, fixture, model=model) as prepared:
            prepared.reset()
            prepared.invoke()
            result = compare(prepared.observe(), fixture, limits[case.id])
            replay_failures = check_dependent_inputs(prepared, fixture, limits[case.id])
            if replay_failures:
                result.failures["replay"] = replay_failures
        payload = {
            "case_sha256": case.identity,
            "fixture_sha256": case.fixture_sha256,
            "limits_sha256": identity,
            "candidate": candidate,
            "metrics": result.metrics,
            "failures": result.failures,
        }
        destination = report / (case.id + ".json")
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        results[case.id] = {"passed": result.passed, "report_sha256": digest(payload)}
        print(f"{case.id}: {'PASS' if result.passed else 'FAIL'}", flush=True)
    if file_digest(LIMITS_PATH) != identity:
        raise ValueError("approved limits changed during numerical acceptance")
    if candidate_identity() != candidate:
        raise ValueError("candidate or validation source changed during acceptance")
    (report / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return all(result["passed"] for result in results.values())
