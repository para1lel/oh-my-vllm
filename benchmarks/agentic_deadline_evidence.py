"""Bind a client deadline abort to original RPC, SDK, and HTTP evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from benchmarks.agentic_rpc_evidence import control_command, control_receipt
from benchmarks.agentic_sdk_trace import read_sdk_spans
from benchmarks.agentic_session_evidence import sdk_api_binding
from benchmarks.agentic_soak_analysis import file_identity


def exactly_one(values: list, description: str):
    if len(values) != 1:
        raise ValueError(f"deadline abort has missing or ambiguous {description}")
    return values[0]


def finite_time(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("deadline abort has an invalid clock")
    return float(value)


def control_records(directory: Path) -> list[dict]:
    commands = {}
    for raw in (directory / "commands.jsonl").read_bytes().splitlines(keepends=True):
        command = json.loads(raw)
        identity = command["id"]
        if identity in commands:
            raise ValueError("deadline abort has duplicate RPC command identities")
        commands[identity] = (command, hashlib.sha256(raw).hexdigest())
    records = [
        json.loads(line)
        for line in (directory / "control-rpc.jsonl").read_text().splitlines()
    ]
    received = {}
    for record in records:
        if record["phase"] == "received":
            number = record["line_number"]
            if type(number) is not int or number <= 0 or number in received:
                raise ValueError("deadline abort has duplicate RPC receipt anchors")
            received[number] = record
    native_results = {}
    expected_receipts = set()
    with gzip.open(directory / "rpc.jsonl.gz", "rb") as source:
        for number, raw in enumerate(source, 1):
            if len(raw) > 16 << 20:
                raise ValueError("deadline abort RPC line exceeds its bound")
            message = json.loads(raw)
            if control_receipt(message):
                expected_receipts.add(number)
            if message.get("type") == "message_end":
                identity = message.get("message", {}).get("responseId")
                native_results[identity] = native_results.get(identity, 0) + 1
            if number in received:
                record = received[number]
                if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
                    raise ValueError("deadline abort RPC receipt bytes changed")
                record["message"] = message
    if set(received) != expected_receipts:
        raise ValueError("deadline abort RPC receipt coverage is incomplete")
    previous = 0
    for record in records:
        monotonic = finite_time(record["monotonic_s"])
        finite_time(record["unix_s"])
        if monotonic < previous:
            raise ValueError("deadline abort RPC clocks moved backwards")
        previous = monotonic
        if record["phase"] in ("dispatch_started", "dispatch_completed"):
            command, digest = commands[record["id"]]
            if digest != record["raw_sha256"] or record["line_number"] is not None:
                raise ValueError("deadline abort command bytes changed")
            record["message"] = command
        elif record["phase"] != "received" or "message" not in record:
            raise ValueError("deadline abort has an unbound RPC record")
        if (
            record["message"].get("type") != record["type"]
            or record["message"].get("id") != record["id"]
        ):
            raise ValueError("deadline abort RPC identity differs from its anchor")
        if (
            record["type"] == "message_end"
            and native_results.get(record["message"]["message"].get("responseId")) != 1
        ):
            raise ValueError("deadline abort has duplicate native result identities")
    expected_commands = {
        identity
        for identity, (message, _) in commands.items()
        if control_command(message)
    }
    for phase in ("dispatch_started", "dispatch_completed"):
        identities = [record["id"] for record in records if record["phase"] == phase]
        if (
            len(identities) != len(expected_commands)
            or set(identities) != expected_commands
        ):
            raise ValueError("deadline abort control command coverage is incomplete")
    return records


def action(records: list[dict], kind: str, deadline: float) -> dict:
    starts = [
        record
        for record in records
        if record["phase"] == "dispatch_started"
        and record["message"].get("type") == kind
        and (kind != "goal" or record["message"].get("op") == "pause")
        and record["monotonic_s"] >= deadline
    ]
    start = exactly_one(starts, f"{kind} dispatch")
    identity = start["id"]
    finish = exactly_one(
        [
            record
            for record in records
            if record["phase"] == "dispatch_completed" and record["id"] == identity
        ],
        f"{kind} flushed command",
    )
    acknowledgement = exactly_one(
        [
            record
            for record in records
            if record["phase"] == "received"
            and record["id"] == identity
            and record["message"].get("type") == "response"
        ],
        f"{kind} acknowledgement",
    )
    response = acknowledgement["message"]
    if (
        response.get("command") != kind
        or response.get("success") is not True
        or start["raw_sha256"] != finish["raw_sha256"]
        or start["monotonic_s"] > finish["monotonic_s"]
        or start["monotonic_s"] > acknowledgement["monotonic_s"]
    ):
        raise ValueError("deadline abort RPC control failed or changed order")
    return {"start": start, "finish": finish, "acknowledgement": acknowledgement}


def deadline_abort(
    directory: Path, session: dict, wire: dict, metric: dict, measured_start: float
) -> dict:
    kind = wire.get("transport_error_kind")
    if (
        kind not in ("BrokenPipeError", "ConnectionResetError")
        or wire.get("error_direction") != "downstream_write"
        or not isinstance(wire.get("error"), str)
        or not wire["error"].startswith(kind + ":")
        or wire.get("transport_outcome") != "failed"
    ):
        raise ValueError("deadline abort lacks a canonical downstream disconnect")
    collection = session["goal_collection"]
    deadline = finite_time(collection["collection_deadline_monotonic_s"])
    deadline_unix = finite_time(collection["collection_deadline_unix_s"])
    if (
        abs(
            deadline_unix
            - finite_time(measured_start)
            - finite_time(session["active_seconds"])
        )
        > 0.002
    ):
        raise ValueError("deadline abort differs from the shared collection window")
    ended = finite_time(collection["controller_ended_monotonic_s"])
    turns = json.loads((directory / "turns.json").read_text())
    turn = exactly_one(
        [item for item in turns if item["transition_reason"] == "collection_deadline"],
        "deadline goal transition",
    )
    if turn["prompt_id"] != wire.get("prompt_id") or not (
        deadline <= turn["active_ended_monotonic_s"] <= ended
    ):
        raise ValueError("deadline abort changed the active goal or collection clock")
    records = control_records(directory)
    offset = deadline_unix - deadline
    for record in records:
        if (
            record["monotonic_s"] >= deadline
            and abs(record["unix_s"] - record["monotonic_s"] - offset) > 0.002
        ):
            raise ValueError("deadline abort wall and monotonic clocks differ")
    pause, abort = action(records, "goal", deadline), action(records, "abort", deadline)
    if not (
        pause["acknowledgement"]["monotonic_s"]
        <= abort["start"]["monotonic_s"]
        <= abort["finish"]["monotonic_s"]
        <= ended
        and abort["acknowledgement"]["monotonic_s"] <= ended
    ):
        raise ValueError("deadline abort preceded its successful goal pause")
    paused_goal = pause["acknowledgement"]["message"].get("data", {}).get("goal", {})
    if (
        paused_goal.get("id") != turn["goal_id"]
        or paused_goal.get("status") != "paused"
    ):
        raise ValueError("deadline abort did not pause the active goal")
    identity = metric["response_id"]
    receipt = exactly_one(
        [
            record
            for record in records
            if record["phase"] == "received"
            and record["message"].get("type") == "message_end"
            and record["message"].get("message", {}).get("responseId") == identity
        ],
        "aborted native message",
    )
    message = receipt["message"]["message"]
    if (
        len(
            [
                record
                for record in records
                if record["phase"] == "received"
                and record["type"] == "message_end"
                and record["monotonic_s"] >= abort["start"]["monotonic_s"]
            ]
        )
        != 1
    ):
        raise ValueError("deadline abort has multiple aborted native requests")
    api_binding = sdk_api_binding(directory, session)
    provider = f"oh-my-vllm-{session['api']}"
    if (
        message.get("role") != "assistant"
        or message.get("stopReason") != "aborted"
        or message.get("api") != api_binding
        or message.get("provider") != provider
        or message.get("model") != "qwen3.8-27b-fp8"
        or not abort["start"]["monotonic_s"] <= receipt["monotonic_s"] <= ended
    ):
        raise ValueError("deadline abort native result identity or clock differs")
    span = exactly_one(
        [
            item
            for item in read_sdk_spans(directory / "sdk-trace")
            if item["attributes"].get("gen_ai.response.id") == identity
        ],
        "SDK result span",
    )
    attributes = span["attributes"]
    if (
        span["scope"] != "@oh-my-pi/pi-agent-core"
        or span["status_code"] != 2
        or attributes.get("error.type") != "aborted"
        or attributes.get("gen_ai.conversation.id") != session["session_id"]
        or attributes.get("gen_ai.provider.name") != provider
        or attributes.get("gen_ai.request.model") != "qwen3.8-27b-fp8"
        or attributes.get("gen_ai.response.model") != "qwen3.8-27b-fp8"
        or attributes.get("omp.gen_ai.request.reasoning.effort") != "max"
        or attributes.get("gen_ai.response.finish_reasons") != ["error"]
        or not re.fullmatch(r"[0-9a-f]{32}", span["trace_id"])
        or int(span["trace_id"], 16) == 0
        or not re.fullmatch(r"[0-9a-f]{16}", span["span_id"])
        or int(span["span_id"], 16) == 0
    ):
        raise ValueError("deadline abort SDK result identity differs")
    completed = finite_time(span["completed_unix_s"])
    started = finite_time(span["started_unix_s"])
    native_completed = finite_time(message["completedAt"]) / 1000
    client_received = finite_time(wire["client_received_unix_s"])
    client_completed = finite_time(wire["client_completed_unix_s"])
    disconnected = finite_time(wire.get("transport_error_monotonic_s"))
    disconnected_unix = finite_time(wire.get("transport_error_unix_s"))
    if abs(disconnected_unix - disconnected - offset) > 0.002:
        raise ValueError("deadline abort disconnect clocks differ")
    if abs(native_completed - completed) > 0.002:
        raise ValueError("deadline abort native and SDK completion clocks differ")
    if not (
        started <= completed
        and started < deadline_unix <= abort["start"]["unix_s"] <= completed
        and completed <= receipt["unix_s"] + 0.002
        and metric["processing_started_unix_s"]
        < deadline_unix
        <= metric["generation_completed_unix_s"]
        and client_received < deadline_unix
        and abort["start"]["monotonic_s"] <= disconnected
        and abort["start"]["unix_s"] <= disconnected_unix <= client_completed
    ):
        raise ValueError("deadline abort does not bind an in-flight deadline request")
    return {
        "kind": "host_collection_deadline_abort",
        "response_id": identity,
        "deadline_monotonic_s": deadline,
        "deadline_unix_s": deadline_unix,
        "pause": pause,
        "abort": abort,
        "native_result": receipt,
        "sdk_result": span,
        "original_files": {
            name: file_identity(directory / name)
            for name in (
                "commands.jsonl",
                "control-rpc.jsonl",
                "rpc.jsonl.gz",
                "turns.json",
            )
        },
    }
