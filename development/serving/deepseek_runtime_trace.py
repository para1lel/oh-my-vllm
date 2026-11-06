"""Bind completed HTTP responses to four-rank runtime accounting events."""

import json
import re
from pathlib import Path

from .deepseek_accounting import (
    BindingError,
    TraceError,
    build_binding,
    digest,
    validate_trace,
)
from .deepseek_matrix import case_id

CALL_FIELDS = (
    "rank",
    "call",
    "phase",
    "request_rows",
    "expert_assignments",
    "dispatch_rows",
    "touched_experts",
)


def rank_paths(path):
    path = Path(path)
    return [path] + [
        path.with_name(f"{path.stem}-rank{rank}{path.suffix}") for rank in range(1, 4)
    ]


def read_rank_events(path):
    events = []
    for rank, source in enumerate(rank_paths(path)):
        rows = []
        try:
            with source.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TraceError("execution trace requires event objects")
                    if row.get("operation") in (
                        "identity",
                        "accounting_call",
                        "accounting_error",
                    ):
                        rows.append(row)
        except (OSError, ValueError) as error:
            raise TraceError(f"cannot read rank {rank} execution trace") from error
        events.append(rows)
    return events


def validate_identities(events, harness, case):
    expected_native = {
        name: {"source": metadata["source_id"], "binary": metadata["binary_id"]}
        for name, metadata in harness["native"].items()
    }
    uuids = [device["uuid"] for device in harness["topology"]]
    if len(events) != 4 or len(uuids) != 4 or len(set(uuids)) != 4:
        raise BindingError(
            "execution trace requires four different GPU UUIDs and ranks"
        )
    common = None
    for rank, rows in enumerate(events):
        found = [row for row in rows if row.get("operation") == "identity"]
        if len(found) != 1 or rows[0] is not found[0]:
            raise BindingError(f"rank {rank} requires one initial identity")
        identity = found[0]
        execution = identity.get("execution", {})
        runtime = execution.get("runtime") or {}
        cache = identity.get("cache_memory", {})
        if (
            type(runtime.get("rank")) is not int
            or runtime["rank"] != rank
            or type(cache.get("rank")) is not int
            or cache["rank"] != rank
        ):
            raise BindingError("execution trace rank differs from its file")
        expected = {
            "source_commit": harness["source_commit"],
            "source_digest": harness["source"],
            "native_modules": expected_native,
            "gpu_uuid": uuids[rank],
        }
        modes = harness.get("configuration", {}).get("execution_modes")
        if modes is not None:
            expected_modes = modes.get(case["strategy"])
            actual_modes = runtime.get("execution_modes")
            if (
                not isinstance(expected_modes, dict)
                or type(expected_modes.get("proposal_graph")) is not bool
                or not isinstance(actual_modes, dict)
                or type(actual_modes.get("proposal_graph")) is not bool
                or actual_modes != expected_modes
            ):
                raise BindingError(
                    "rank batch/Graph execution modes differ from harness"
                )
        if any(runtime.get(key) != value for key, value in expected.items()):
            raise BindingError(f"rank {rank} runtime identity differs from harness")
        if identity.get("checkpoint") != harness["checkpoint"]:
            raise BindingError("rank checkpoint identity differs from harness")
        if identity.get("tokenizer") != harness["tokenizer"]:
            raise BindingError("rank tokenizer identity differs from harness")
        if (
            cache.get("strategy") != case["strategy"]
            or cache.get("effective_strategy") != case["strategy"]
        ):
            raise BindingError("effective strategy differs: fallback is not evidence")
        config = identity.get("config", {})
        drafts = 5 if case["mode"] == "dspark5" else 0
        if (
            type(config.get("num_speculative_tokens")) is not int
            or config["num_speculative_tokens"] != drafts
        ):
            raise BindingError("runtime mode differs from HTTP case")
        current = (
            config,
            {key: value for key, value in cache.items() if key != "rank"},
            {key: value for key, value in execution.items() if key != "runtime"},
        )
        if common is not None and current != common:
            raise BindingError("rank execution identities differ")
        common = current
    return expected_native, uuids


def response_manifest(responses, case):
    if len(responses) != case["batch"]:
        raise TraceError("HTTP response count differs from the case")
    requests = []
    for ordinal, response in enumerate(responses):
        response_id = response.get("id")
        if (
            not isinstance(response_id, str)
            or re.fullmatch(r"chatcmpl_.*_?[0-9]+", response_id) is None
        ):
            raise TraceError("HTTP response ID requires a positive request ID suffix")
        suffix = response_id.rsplit("_", 1)[-1]
        if not suffix.isascii() or not suffix.isdecimal() or int(suffix) < 1:
            raise TraceError("HTTP response ID requires a positive request ID suffix")
        usage = response.get("usage", {})
        lengths = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        if any(type(value) is not int or value < 1 for value in lengths):
            raise TraceError("HTTP usage lengths must be positive integers")
        if (
            lengths != [case["input_tokens"], case["output_tokens"]]
            or usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0
        ):
            raise TraceError("HTTP token count or prefix check failed")
        requests.append(
            {
                "ordinal": ordinal,
                "id": int(suffix),
                "response_id": response_id,
                "input_tokens": lengths[0],
                "output_tokens": lengths[1],
                "owner_rank": 0,
            }
        )
    if len({row["id"] for row in requests}) != len(requests):
        raise TraceError("HTTP responses repeat a request ID")
    return {"mode": case["mode"], "requests": requests}


def select_calls(events, request_ids):
    calls = {}
    selected = set()
    for rank, rows in enumerate(events):
        for event in rows:
            if event.get("operation") != "accounting_call":
                continue
            if type(event.get("rank")) is not int or event["rank"] != rank:
                raise TraceError("accounting call rank differs from its file")
            call = event.get("call")
            if type(call) is not int or call < 0:
                raise TraceError("accounting call ID must be a nonnegative integer")
            request_rows = event.get("request_rows")
            if not isinstance(request_rows, list) or any(
                not isinstance(row, dict) or type(row.get("id")) is not int
                for row in request_rows
            ):
                raise TraceError("accounting call requires request IDs")
            ids = {row["id"] for row in request_rows}
            if ids & request_ids:
                selected.add(call)
            calls.setdefault(call, []).append(event)
    result = []
    for call in sorted(selected):
        records = calls[call]
        if any(
            row["id"] not in request_ids
            for record in records
            for row in record["request_rows"]
        ):
            raise TraceError("accounting call mixes requests from different HTTP cases")
        for record in records:
            if any(field not in record for field in CALL_FIELDS):
                raise TraceError("accounting call lacks required counters")
            result.append({field: record[field] for field in CALL_FIELDS})
    return result


def request_owners(calls):
    owners = {}
    for record in calls:
        for row in record["request_rows"]:
            prior = owners.setdefault(row["id"], record["rank"])
            if prior != record["rank"]:
                raise TraceError("DEP4 request owner changed during the HTTP run")
    return owners


def assemble_trace(events, responses, *, harness, case):
    for rank, rows in enumerate(events):
        for event in rows:
            if event.get("operation") == "accounting_error":
                raise TraceError(
                    f"accounting_error on rank {rank}, call {event.get('call')}: "
                    f"{event.get('reason')}"
                )
    native, uuids = validate_identities(events, harness, case)
    manifest = response_manifest(responses, case)
    calls = select_calls(events, {row["id"] for row in manifest["requests"]})
    owners = request_owners(calls) if case["strategy"] == "dep4" else {}
    if case["strategy"] == "dep4":
        for request in manifest["requests"]:
            if request["id"] not in owners:
                raise TraceError("DEP4 request lacks an observed owner")
            request["owner_rank"] = owners[request["id"]]
    accepted = proposed = 0
    for record in calls:
        if case["strategy"] == "dep4" or record["rank"] == 0:
            for row in record["request_rows"]:
                for name in ("accepted", "proposed"):
                    if type(row.get(name)) is not int or row[name] < 0:
                        raise TraceError(f"{name} must be a nonnegative integer")
                accepted += row["accepted"]
                proposed += row["proposed"]
    binding = build_binding(
        source={"commit": harness["source_commit"], "digest": harness["source"]},
        native_modules=native,
        checkpoint_config=harness["checkpoint"]["config"],
        checkpoint_index=harness["checkpoint"]["index"],
        tokenizer=harness["tokenizer"],
        gpu_uuids=uuids,
        strategy=case["strategy"],
        effective_strategy=case["strategy"],
        owner_plan={
            "rule": "observed request_rows rank" if owners else "replicated",
            "requests": [
                {"id": row["id"], "owner_rank": row["owner_rank"]}
                for row in manifest["requests"]
            ],
        },
        sampling="greedy",
        case_id=case_id(case),
        accepted=accepted,
        proposed=proposed,
        manifest_sha256=digest(manifest),
    )
    modes = harness.get("configuration", {}).get("execution_modes")
    if modes is not None:
        binding["runtime_modes"] = modes[case["strategy"]]
    trace = {"binding": binding, "manifest": manifest, "calls": calls}
    validate_trace(trace, strategy=case["strategy"], expected_binding=binding)
    return trace
