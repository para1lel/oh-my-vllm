"""Check the 19 long HTTP rows without upgrading an estimate to a roofline."""

import argparse
import json
import math
from pathlib import Path

from . import deepseek_accounting as source
from . import deepseek_accounting_roofline as roofline
from .deepseek_benchmark import reference_configuration, runtime_reference
from .deepseek_matrix import HTTP_CASES, case_id
from .deepseek_runtime_trace import request_owners, response_manifest
from .gpu import B200_PATTERN

LONG_CASES = {
    case_id(case): case for case in HTTP_CASES if case["output_tokens"] == 8192
}
PENDING_QUALIFICATION = (
    "v3.1 uses BF16 history and empirical collective latency.",
    "v4 is a partial resource floor with no HBM or host credit.",
    "v5 is a protocol estimate whose call-dependent reloads are not necessary work.",
    "Full efficiency acceptance requires the fixed-algorithm v6 resource roofline.",
)


def qualified(version, scope):
    return version == roofline.VERSION and scope == "fixed-algorithm resource roofline"


def _measurement_configuration(result, identity, case):
    version = result.get("accounting_version")
    configuration = reference_configuration(version)
    if identity.get("reference") != configuration:
        raise ValueError("reference identity differs from the selected calculation")
    for key in (
        "binary",
        "source",
        "source_commit",
        "native",
        "checkpoint",
        "tokenizer",
        "topology",
        "harness_numeric_runtime",
        "configuration",
    ):
        if not identity.get(key):
            raise ValueError(f"measurement identity lacks {key}")
    topology = identity["topology"]
    if version == roofline.VERSION and (
        not isinstance(topology, list)
        or len(topology) != 4
        or any(
            not isinstance(device, dict)
            or not isinstance(device.get("name"), str)
            or not B200_PATTERN.search(device["name"])
            for device in topology
        )
    ):
        raise ValueError("v6 requires four reviewed B200 GPU model identities")
    all_modes = identity["configuration"].get("execution_modes")
    modes = all_modes.get(case["strategy"]) if isinstance(all_modes, dict) else None
    if (
        not isinstance(modes, dict)
        or set(modes)
        != {"batch_decode", "decode_graph", "decode_graph_limit", "proposal_graph"}
        or any(
            type(modes[key]) is not bool
            for key in ("batch_decode", "decode_graph", "proposal_graph")
        )
        or type(modes["decode_graph_limit"]) is not int
        or modes["decode_graph_limit"] < 0
    ):
        raise ValueError("measurement lacks actual batch/Graph modes")
    return version, configuration, modes


def _measurement_binding(trace, identity, modes, identifier, case):
    binding = trace["binding"]
    expected = {
        "source": {"commit": identity["source_commit"], "digest": identity["source"]},
        "native_modules": {
            name: {"source": value["source_id"], "binary": value["binary_id"]}
            for name, value in identity["native"].items()
        },
        "checkpoint_config": identity["checkpoint"]["config"],
        "checkpoint_index": identity["checkpoint"]["index"],
        "tokenizer": identity["tokenizer"],
        "gpu_uuids": [device["uuid"] for device in identity["topology"]],
        "runtime_modes": modes,
        "case_id": identifier,
        "strategy": case["strategy"],
        "effective_strategy": case["strategy"],
        "sampling": "greedy",
    }
    runtime_modes = binding.get("runtime_modes")
    if (
        not isinstance(runtime_modes, dict)
        or type(runtime_modes.get("proposal_graph")) is not bool
        or any(binding.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("trace identity differs from the measured run")
    return binding


def validate_measurement(result):
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("HTTP result is absent, partial or failed")
    identifier = case_id(result)
    case = next((case for case in HTTP_CASES if case_id(case) == identifier), None)
    if case is None or any(result.get(key) != value for key, value in case.items()):
        raise ValueError("result is outside the approved matrix")
    if (
        result.get("request_count") != case["batch"]
        or result.get("effective_strategy") != case["strategy"]
    ):
        raise ValueError("HTTP count or effective strategy differs")
    seconds = result.get("seconds")
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("full HTTP seconds must be finite and positive")
    identity = result.get("measurement_identity")
    if not isinstance(identity, dict):
        raise ValueError("result lacks its complete measurement identity")
    version, configuration, modes = _measurement_configuration(result, identity, case)
    trace = result.get("trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("binding"), dict):
        raise ValueError("result lacks its runtime trace")
    manifest = trace.get("manifest", {})
    requests = manifest.get("requests", [])
    if (
        manifest.get("mode") != case["mode"]
        or len(requests) != case["batch"]
        or any(
            row.get("input_tokens") != case["input_tokens"]
            or row.get("output_tokens") != case["output_tokens"]
            for row in requests
        )
    ):
        raise ValueError("runtime manifest does not cover the HTTP case")
    responses = result.get("responses")
    if not isinstance(responses, list) or any(
        not isinstance(row, dict) for row in responses
    ):
        raise ValueError("HTTP result lacks completed responses")
    binding = _measurement_binding(trace, identity, modes, identifier, case)
    for key, value in configuration.items():
        if result.get(key) != value:
            raise ValueError(f"result reference field differs: {key}")
    reference, _metadata = runtime_reference(
        trace, strategy=case["strategy"], expected_binding=binding, version=version
    )
    observed = response_manifest(responses, case)
    if case["strategy"] == "dep4":
        owners = request_owners(trace["calls"])
        for request in observed["requests"]:
            owner = owners.get(request["id"])
            if type(owner) is not int or owner not in range(4):
                raise ValueError("DEP4 response lacks its observed trace owner")
            request["owner_rank"] = owner
    if observed != manifest:
        raise ValueError("HTTP responses differ from the runtime manifest")
    if result.get("reference_acceptance") != reference.acceptance:
        raise ValueError("static reference scales are not runtime evidence")
    if result.get("reference_seconds") != reference.seconds:
        raise ValueError("stored reference value differs from recomputation")
    ratio = source.efficiency_percent(reference.seconds, seconds)
    if result.get("efficiency_percent") != ratio:
        raise ValueError("stored efficiency differs from recomputation")
    common = {key: value for key, value in identity.items() if key != "workload"}
    return identifier, ratio, source.digest(common)


def evaluate(results):
    rows = {}
    identities = set()
    errors = []
    eligibility = []
    for result in results:
        try:
            identifier, ratio, identity = validate_measurement(result)
            if identifier in rows:
                raise ValueError(f"duplicate HTTP case: {identifier}")
            rows[identifier] = ratio
            identities.add(identity)
            eligibility.append(
                qualified(
                    result.get("accounting_version"), result.get("reference_scope")
                )
            )
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(str(error))
    missing = sorted(set(LONG_CASES) - set(rows))
    if len(identities) != 1:
        errors.append("results do not have one complete measurement identity")
    if missing:
        errors.append("the 19 long-output cases are incomplete")
    failed = sorted(
        identifier
        for identifier, ratio in rows.items()
        if identifier in LONG_CASES and ratio < 33
    )
    complete = not missing and not errors
    eligible = bool(eligibility) and all(eligibility)
    passed = complete and eligible and not failed
    return {
        "status": "invalid"
        if errors
        else "ineligible"
        if not eligible
        else "passed"
        if passed
        else "failed",
        "coverage_complete": complete,
        "required_long_cases": 19,
        "observed_long_cases": len(set(rows) & set(LONG_CASES)),
        "missing": missing,
        "failed_threshold_cases": failed,
        "reference_threshold_met": complete and not failed,
        "threshold_percent": 33,
        "reference_qualification": "fixed-algorithm-roofline"
        if eligible
        else "ineligible",
        "pending_qualification": [] if eligible else list(PENDING_QUALIFICATION),
        "target_passed": passed,
        "errors": errors,
        "scope": (
            "19-row efficiency check only; no numerical, capacity or AgentX acceptance"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        report = evaluate([json.loads(path.read_bytes()) for path in arguments.results])
    except (OSError, ValueError) as error:
        report = {"status": "invalid", "target_passed": False, "errors": [str(error)]}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    print(text, end="")
    raise SystemExit(
        0
        if report.get("target_passed")
        else 2
        if report["status"] == "ineligible"
        else 1
    )


if __name__ == "__main__":
    main()
