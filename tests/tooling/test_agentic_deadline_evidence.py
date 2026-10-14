"""Only an original, uniquely bound deadline abort can explain a missing terminal."""

import gzip
import hashlib
import json

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from benchmarks.agentic_framework_evidence import audit_framework_session
from tests.support.framework_evidence import request_fixture as fixture
from tests.support.sdk_trace import export, saved_trace


def save_journals(directory, commands, messages, records):
    command_lines = [(json.dumps(item) + "\n").encode() for item in commands]
    message_lines = [(json.dumps(item) + "\n").encode() for item in messages]
    (directory / "commands.jsonl").write_bytes(b"".join(command_lines))
    with gzip.open(directory / "rpc.jsonl.gz", "wb") as output:
        output.write(b"".join(message_lines))
    for record in records:
        if record["phase"] == "received":
            raw = message_lines[record["line_number"] - 1]
        else:
            raw = next(
                raw
                for item, raw in zip(commands, command_lines, strict=True)
                if item["id"] == record["id"]
            )
        record["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    (directory / "control-rpc.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )


def deadline_fixture(directory, api):
    _, metrics, wire = fixture(directory, api, cancelled=True)
    identity, metric = next(iter(metrics.items()))
    metric.update(
        status="completed",
        received_unix_s=1199,
        processing_started_unix_s=1199.1,
        generation_completed_unix_s=1200.01,
        processing_seconds=0.91,
        queue_seconds=0.1,
    )
    path = directory / "wire/response.bin"
    raw = path.read_bytes().split(b"\n\n", 1)[0] + b"\n\n"
    path.write_bytes(raw)
    wire.update(
        response_sha256=hashlib.sha256(raw).hexdigest(),
        prompt_id="goal-command",
        session_id="fixed-session",
        client_received_unix_s=1199,
        client_completed_unix_s=1200.02,
        transport_error_monotonic_s=200.015,
        transport_error_unix_s=1200.015,
        error_direction="downstream_write",
        transport_error_kind="BrokenPipeError",
        error="BrokenPipeError: [Errno 32] Broken pipe",
        transport_outcome="failed",
    )
    binding = "openai-completions" if api == "chat" else "openai-responses"
    attributes = {
        "gen_ai.provider.name": f"oh-my-vllm-{api}",
        "gen_ai.response.id": identity,
        "gen_ai.response.finish_reasons": ["error"],
        "error.type": "aborted",
    }
    session = saved_trace(
        directory,
        export(
            attributes,
            status_code=2,
            start_time_unix_nano=1199000000000,
            end_time_unix_nano=1200003000000,
        ),
        api=api,
    )
    session.update(
        lane=0,
        active_seconds=100,
        goal_collection={
            "collection_deadline_monotonic_s": 200,
            "collection_deadline_unix_s": 1200,
            "controller_ended_monotonic_s": 201,
        },
    )
    commands = [
        {"id": "pause", "type": "goal", "op": "pause"},
        {"id": "abort", "type": "abort"},
    ]
    messages = [
        {
            "id": "pause",
            "type": "response",
            "command": "goal",
            "success": True,
            "data": {"goal": {"id": "goal-1", "status": "paused"}},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": identity,
                "stopReason": "aborted",
                "api": binding,
                "provider": f"oh-my-vllm-{api}",
                "model": "qwen3.8-27b-fp8",
                "completedAt": 1200003,
            },
        },
        {"id": "abort", "type": "response", "command": "abort", "success": True},
    ]
    records = []
    for phase, kind, identity, delta, number in (
        ("dispatch_started", "goal", "pause", 0.001, None),
        ("dispatch_completed", "goal", "pause", 0.0015, None),
        ("received", "response", "pause", 0.002, 1),
        ("dispatch_started", "abort", "abort", 0.0025, None),
        ("received", "message_end", None, 0.004, 2),
        ("dispatch_completed", "abort", "abort", 0.005, None),
        ("received", "response", "abort", 0.006, 3),
    ):
        records.append(
            {
                "phase": phase,
                "type": kind,
                "id": identity,
                "monotonic_s": 200 + delta,
                "unix_s": 1200 + delta,
                "line_number": number,
            }
        )
    (directory / "turns.json").write_text(
        json.dumps(
            [
                {
                    "transition_reason": "collection_deadline",
                    "prompt_id": "goal-command",
                    "goal_id": "goal-1",
                    "active_ended_monotonic_s": 200,
                }
            ]
        )
    )
    save_journals(directory, commands, messages, records)
    (directory / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    return session, metrics, wire, commands, messages, records


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_server_completion_racing_client_abort_has_a_separate_outcome(tmp_path, api):
    session, metrics, *_ = deadline_fixture(tmp_path, api)
    points, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    assert points == []
    assert audit["completed_requests"] == audit["cancelled_requests"] == 0
    assert audit["deadline_aborted_completed_requests"] == 1
    [outcome] = audit["outcomes"]
    assert outcome["status"] == "completed"
    assert outcome["protocol"]["success"] is False
    assert outcome["deadline_abort_proof"]["kind"] == "host_collection_deadline_abort"


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("completed", [1200.0001, float("inf"), float("nan"), True])
def test_deadline_tail_requires_a_finite_disconnect_after_abort(
    tmp_path, api, completed
):
    session, metrics, wire, *_ = deadline_fixture(tmp_path, api)
    wire["client_completed_unix_s"] = completed
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("output_tokens", [7, 262144])
def test_deadline_tail_keeps_the_actual_output_budget(tmp_path, api, output_tokens):
    session, metrics, *_ = deadline_fixture(tmp_path, api)
    next(iter(metrics.values()))["output_tokens"] = output_tokens
    with pytest.raises(ValueError, match="actual model output budget"):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("failure", ["early", "missing", "clock_step"])
def test_late_handler_cleanup_does_not_prove_deadline_disconnect(
    tmp_path, api, failure
):
    session, metrics, wire, *_ = deadline_fixture(tmp_path, api)
    if failure == "early":
        wire["transport_error_monotonic_s"] = 200.0001
        wire["transport_error_unix_s"] = 1200.0001
    elif failure == "missing":
        wire.pop("transport_error_monotonic_s")
    else:
        wire["transport_error_unix_s"] += 0.01
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize(
    "failure",
    [
        "no_ack",
        "failed_ack",
        "wrong_ack",
        "duplicate_ack",
        "pre_deadline",
        "unflushed",
        "wrong_goal",
        "wrong_rpc_api",
        "wrong_rpc_response",
        "duplicate_rpc_result",
        "clock_step",
        "upstream_failure",
        "no_transport_error",
        "wrong_prompt",
        "ordinary_missing_terminal",
        "source_hash",
        "sdk_hash",
        "unrecorded_ack",
        "unrecorded_command",
    ],
)
def test_partial_or_unrelated_abort_evidence_cannot_supply_acceptance(
    tmp_path, failure
):
    session, metrics, wire, commands, messages, records = deadline_fixture(
        tmp_path, "responses"
    )
    if failure == "no_ack":
        records.pop()
    elif failure == "failed_ack":
        messages[-1]["success"] = False
    elif failure == "wrong_ack":
        messages[-1]["command"] = "goal"
    elif failure == "duplicate_ack":
        records.append(dict(records[-1]))
    elif failure == "pre_deadline":
        records[3]["monotonic_s"], records[3]["unix_s"] = 199, 1199
    elif failure == "unflushed":
        records.pop(5)
    elif failure == "wrong_goal":
        messages[0]["data"]["goal"]["id"] = "another-goal"
    elif failure == "wrong_rpc_api":
        messages[1]["message"]["api"] = "openai-completions"
    elif failure == "wrong_rpc_response":
        messages[1]["message"]["responseId"] = "another-request"
    elif failure == "duplicate_rpc_result":
        messages.append(messages[1])
    elif failure == "clock_step":
        records[3]["unix_s"] += 0.01
    elif failure == "upstream_failure":
        wire["error_direction"] = "upstream_read"
    elif failure == "no_transport_error":
        wire["error"] = None
    elif failure == "wrong_prompt":
        wire["prompt_id"] = "unrelated-goal"
    elif failure == "ordinary_missing_terminal":
        session.pop("goal_collection")
    elif failure == "unrecorded_ack":
        messages.append(messages[-1])
    elif failure == "unrecorded_command":
        commands.append({"id": "untracked-abort", "type": "abort"})
    save_journals(tmp_path, commands, messages, records)
    if failure == "source_hash":
        with (tmp_path / "commands.jsonl").open("ab") as output:
            output.write(b'{"id":"abort", "type":"abort"}\n')
    elif failure == "sdk_hash":
        (tmp_path / "sdk-trace/0000.pb").write_bytes(b"modified")
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize(
    "failure",
    [
        "session",
        "provider",
        "model",
        "error",
        "status",
        "duplicate",
        "zero_id",
        "early_completion",
        "negative_duration",
    ],
)
def test_changed_sdk_evidence_cannot_explain_an_aborted_native_result(
    tmp_path, failure
):
    session, metrics, *_ = deadline_fixture(tmp_path, "responses")
    path = tmp_path / "sdk-trace/0000.pb"
    request = ExportTraceServiceRequest.FromString(path.read_bytes())
    spans = request.resource_spans[0].scope_spans[0].spans
    span = spans[0]
    if failure == "duplicate":
        duplicate = spans.add()
        duplicate.CopyFrom(span)
        duplicate.span_id = b"\x03" * 8
    elif failure == "zero_id":
        span.trace_id = b"\0" * 16
    elif failure == "status":
        span.status.code = 0
    elif failure == "early_completion":
        span.end_time_unix_nano = 1199999000000
    elif failure == "negative_duration":
        span.start_time_unix_nano = 1199999500000
        span.end_time_unix_nano = 1199999000000
    else:
        key = {
            "session": "gen_ai.conversation.id",
            "provider": "gen_ai.provider.name",
            "model": "gen_ai.request.model",
            "error": "error.type",
        }[failure]
        next(
            item for item in span.attributes if item.key == key
        ).value.string_value = "unrelated"
    raw = request.SerializeToString()
    path.write_bytes(raw)
    (path.parent / "exports.jsonl").write_text(
        json.dumps(
            {
                "file": "0000.pb",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        + "\n"
    )
    (path.parent / "closed.json").write_text(
        json.dumps({"closed": True, "exports": 1, "bytes": len(raw), "error": None})
    )
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize("changed", ["line", "hash"])
def test_receipt_requires_the_original_decompressed_rpc_line(tmp_path, changed):
    session, metrics, *_ = deadline_fixture(tmp_path, "chat")
    path = tmp_path / "control-rpc.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if changed == "line":
        records[4]["line_number"] = 999
    else:
        records[4]["raw_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 1100)


def test_deadline_marker_cannot_relax_a_normal_successful_usage_check(tmp_path):
    session, metrics, _ = fixture(tmp_path, "chat")
    session["goal_collection"] = {"collection_deadline_monotonic_s": 102}
    next(iter(metrics.values()))["output_tokens"] = 999
    with pytest.raises(ValueError, match="usage differs"):
        audit_framework_session(tmp_path, session, metrics, 100)


def test_one_abort_cannot_exempt_multiple_model_requests(tmp_path):
    session, metrics, wire, commands, messages, records = deadline_fixture(
        tmp_path, "responses"
    )
    first_id, first_metric = next(iter(metrics.items()))
    second_id = "resp_fixture_2"
    metrics[second_id] = dict(first_metric, response_id=second_id, request_id=2)
    second_body = (
        (tmp_path / "wire/response.bin")
        .read_bytes()
        .replace(first_id.encode(), second_id.encode())
    )
    (tmp_path / "wire/second-response.bin").write_bytes(second_body)
    second_wire = dict(
        wire,
        response_file="second-response.bin",
        response_ids=[second_id],
        response_sha256=hashlib.sha256(second_body).hexdigest(),
    )
    (tmp_path / "wire/http.jsonl").write_text(
        json.dumps(wire) + "\n" + json.dumps(second_wire) + "\n"
    )
    messages.append(
        {
            "type": "message_end",
            "message": dict(messages[1]["message"], responseId=second_id),
        }
    )
    records.append(
        dict(records[4], line_number=4, monotonic_s=200.007, unix_s=1200.007)
    )
    save_journals(tmp_path, commands, messages, records)
    with pytest.raises(ValueError, match="multiple aborted native requests"):
        audit_framework_session(tmp_path, session, metrics, 1100)
