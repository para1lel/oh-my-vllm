"""Only an original, uniquely bound deadline abort can explain a missing terminal."""

import gzip
import hashlib
import json

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from benchmarks.agentic_framework_evidence import audit_framework_session
from benchmarks.agentic_lane_transport import lane_transport_evidence
from benchmarks.agentic_soak import explained_aborts
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


def deadline_fixture(directory, api, *, status="completed"):
    _, metrics, wire = fixture(directory, api, cancelled=True)
    identity, metric = next(iter(metrics.items()))
    metric.update(
        status=status,
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
        workload="translation",
        workload_collection={
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
@pytest.mark.parametrize("collection_key", ["workload_collection", "goal_collection"])
def test_server_completion_racing_client_abort_has_a_separate_outcome(
    tmp_path, api, collection_key
):
    """The neutral workload key and the historical goal-only key both audit."""
    session, metrics, *_ = deadline_fixture(tmp_path, api)
    session[collection_key] = session.pop("workload_collection")
    points, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    assert points == []
    assert audit["completed_requests"] == audit["cancelled_requests"] == 0
    assert audit["deadline_aborted_completed_requests"] == 1
    [outcome] = audit["outcomes"]
    assert outcome["status"] == "completed"
    assert outcome["protocol"]["success"] is False
    assert outcome["deadline_abort_proof"]["kind"] == "host_collection_deadline_abort"
    assert outcome["deadline_abort_proof"]["deadline_unix_s"] == 1200


def test_neutral_collection_record_wins_over_a_stale_historical_key(tmp_path):
    session, metrics, *_ = deadline_fixture(tmp_path, "chat")
    session["goal_collection"] = {
        "collection_deadline_monotonic_s": 102,
        "collection_deadline_unix_s": 1102,
        "controller_ended_monotonic_s": 103,
    }
    _points, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    assert audit["deadline_aborted_completed_requests"] == 1
    [outcome] = audit["outcomes"]
    assert outcome["deadline_abort_proof"]["deadline_monotonic_s"] == 200


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_translation_session_without_disconnect_proof_cannot_claim_an_abort(
    tmp_path, api
):
    session, metrics, wire, *_ = deadline_fixture(tmp_path, api)
    wire["error"] = None
    wire.pop("transport_error_kind")
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(ValueError, match="lacks a canonical downstream disconnect"):
        audit_framework_session(tmp_path, session, metrics, 1100)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_translation_session_with_an_early_disconnect_cannot_explain_itself(
    tmp_path, api
):
    session, metrics, wire, *_ = deadline_fixture(tmp_path, api)
    wire["transport_error_monotonic_s"] = 199.9
    wire["transport_error_unix_s"] = 1199.9
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(ValueError, match="deadline abort"):
        audit_framework_session(tmp_path, session, metrics, 1100)


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
        session.pop("workload_collection", None)
        session.pop("goal_collection", None)
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
    session["workload_collection"] = {"collection_deadline_monotonic_s": 102}
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


def model_hop(
    directory, digest, *, path, failed=True, received=199.2, completed=200.008
):
    """One recorded model-hop entry in the mounted-socket journal shape.

    The fixture's recorded abort action starts at 200.0025 and its client
    disconnect happens at 200.015, so an attempt that completes at 200.008 lies
    inside the window the original evidence proves.
    """
    entry = {
        "path": path,
        "body_sha256": digest,
        "upstream_dispatched": True,
        "received_monotonic_s": received,
        "completed_monotonic_s": completed,
    }
    if failed:
        entry.update(
            error="BrokenPipeError: [Errno 32] Broken pipe",
            error_kind="BrokenPipeError",
            error_direction="downstream_write",
        )
    return entry


def write_model_hops(directory, bridge, guest):
    """Write one model-only lane closure record set from hop entry lists."""
    bridge, guest = list(bridge), list(guest)
    transport = directory / "transport"
    transport.mkdir(exist_ok=True)
    for name, entries in (("model", bridge), ("guest-model", guest)):
        (transport / f"{name}.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries)
        )
    (transport / "bridge.json").write_text(
        json.dumps(
            {
                "closed": True,
                "errors": [],
                "handlers_released": True,
                "remaining_sockets": [],
                "telemetry_upstream": None,
                "journals": {"bridge-model": {"entries": len(bridge), "error": None}},
            }
        )
    )
    (transport / "guest-close.json").write_text(
        json.dumps(
            {
                "released": True,
                "servers": [
                    {
                        "errors": [],
                        "journal_error": None,
                        "journal_entries": len(guest),
                    }
                ],
            }
        )
    )


def audited_lane(directory, api, *, status="completed"):
    """Run the collector path once and return its audit, outcome and digest."""
    directory.mkdir(parents=True, exist_ok=True)
    session, metrics, wire, *_ = deadline_fixture(directory, api, status=status)
    _, audit = audit_framework_session(directory, session, metrics, 1100)
    assert len(audit["outcomes"]) == 1
    digest = hashlib.sha256(
        (directory / "wire" / wire["request_file"]).read_bytes()
    ).hexdigest()
    return audit, audit["outcomes"][0], digest, wire


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_collector_path_binds_every_mounted_hop(tmp_path, api, status):
    """The framework audit, its explanation and the transport audit agree.

    A service cancellation used to skip the deadline proof entirely, so the
    soak runner produced no explanation and the real hop disconnects failed
    transport acceptance.
    """
    audit, outcome, digest, wire = audited_lane(tmp_path, api, status=status)
    assert outcome["status"] == status
    assert outcome["client_completion"] == "collection_deadline_aborted"
    proof = outcome["deadline_abort_proof"]
    assert proof["disconnect_direction"] == "downstream_write"
    assert proof["deadline_monotonic_s"] == 200

    explained = explained_aborts(tmp_path, audit)
    assert set(explained) == {digest}
    bound = explained[digest][0]
    assert bound["direction"] == "downstream_write"
    assert bound["deadline_monotonic_s"] == 200
    # The window is the recorded abort action through the client disconnect.
    assert bound["abort_monotonic_s"] == bound["first_monotonic_s"] == 200.0025
    assert bound["last_monotonic_s"] == 200.015

    entry = model_hop(tmp_path, digest, path=wire["path"])
    write_model_hops(tmp_path, [entry], [entry])
    evidence = lane_transport_evidence(tmp_path, explained={"model": explained})
    assert evidence["bridge-model"]["reached_records"] == 0
    assert evidence["bridge-model"]["explained_aborts"] == 1
    assert evidence["guest-model"]["explained_aborts"] == 1
    assert evidence["unused_explanations"] == []
    assert evidence["explained_aborts"] == {"model": [digest]}


def test_cancelled_outcome_keeps_unknown_cause_when_proof_evidence_is_missing(
    tmp_path,
):
    """A cancellation stays unproved, not fatal, when optional evidence is gone.

    Only the completed-server tail is required to fail closed; a cancelled
    request that cannot be bound keeps an explicit diagnostic instead of
    aborting the whole framework audit.
    """
    session, metrics, *_ = deadline_fixture(tmp_path, "chat", status="cancelled")
    (tmp_path / "turns.json").unlink()
    _, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    outcome = audit["outcomes"][0]
    assert outcome["status"] == "cancelled"
    assert "deadline_abort_proof" not in outcome
    assert outcome["cancellation_evidence"].startswith("FileNotFoundError")
    assert explained_aborts(tmp_path, audit) == {}


def test_cancelled_outcome_keeps_unknown_cause_when_proof_evidence_is_malformed(
    tmp_path,
):
    """Malformed optional evidence is diagnosed instead of failing the audit."""
    session, metrics, *_ = deadline_fixture(tmp_path, "chat", status="cancelled")
    (tmp_path / "turns.json").write_text("{not json")
    _, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    outcome = audit["outcomes"][0]
    assert "deadline_abort_proof" not in outcome
    assert outcome["cancellation_evidence"].startswith("JSONDecodeError")
    assert explained_aborts(tmp_path, audit) == {}


def test_cancelled_outcome_without_canonical_evidence_claims_no_abort(tmp_path):
    """A cancellation stays unproved when its original disconnect differs."""
    session, metrics, wire, *_ = deadline_fixture(tmp_path, "chat", status="cancelled")
    wire.update(
        transport_error_kind="ConnectionRefusedError",
        error="ConnectionRefusedError: refused",
    )
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(wire) + "\n")
    _, audit = audit_framework_session(tmp_path, session, metrics, 1100)
    outcome = audit["outcomes"][0]
    assert outcome["status"] == "cancelled"
    assert "deadline_abort_proof" not in outcome
    assert "canonical downstream disconnect" in outcome["cancellation_evidence"]
    assert explained_aborts(tmp_path, audit) == {}
    # Without an explanation the recorded hop failure stays visible.
    digest = hashlib.sha256(
        (tmp_path / "wire" / wire["request_file"]).read_bytes()
    ).hexdigest()
    entry = model_hop(tmp_path, digest, path=wire["path"])
    write_model_hops(tmp_path, [entry], [entry])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(tmp_path)


@pytest.mark.parametrize("shape", ["asymmetric", "zero_errors"])
def test_collector_path_accepts_completed_hops(tmp_path, shape):
    """One hop may complete while the other records the explained failure."""
    audit, _, digest, wire = audited_lane(tmp_path, "chat")
    explained = explained_aborts(tmp_path, audit)
    failed = model_hop(tmp_path, digest, path=wire["path"])
    relayed = model_hop(tmp_path, digest, path=wire["path"], failed=False)
    bridge, guest = (failed, relayed) if shape == "asymmetric" else (relayed, relayed)
    write_model_hops(tmp_path, [bridge], [guest])
    evidence = lane_transport_evidence(tmp_path, explained={"model": explained})
    assert evidence["bridge-model"]["explained_aborts"] == (
        1 if shape == "asymmetric" else 0
    )
    assert evidence["guest-model"]["explained_aborts"] == 0
    assert evidence["unused_explanations"] == (
        [] if shape == "asymmetric" else [f"model:{digest}"]
    )


def test_collector_path_rejects_an_unrelated_upstream_failure(tmp_path):
    """A deadline bound never covers a failure in another direction or window."""
    audit, _, digest, wire = audited_lane(tmp_path, "chat")
    explained = explained_aborts(tmp_path, audit)
    unrelated = model_hop(tmp_path, digest, path=wire["path"])
    unrelated.update(
        error="TimeoutError: upstream stalled",
        error_kind="TimeoutError",
        error_direction="upstream_read",
    )
    write_model_hops(tmp_path, [unrelated], [unrelated])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(tmp_path, explained={"model": explained})

    # A same-direction failure that ended before the deadline or just before the
    # recorded abort action is a different attempt, not the deadline release.
    for completed in (199.9999, 200.0024):
        predeadline = model_hop(
            tmp_path, digest, path=wire["path"], completed=completed
        )
        write_model_hops(tmp_path, [predeadline], [predeadline])
        with pytest.raises(
            ValueError, match="unexplained bridge model transport failure"
        ):
            lane_transport_evidence(tmp_path, explained={"model": explained})


def test_collector_path_keeps_distinct_originals_with_equal_bodies(tmp_path):
    """Two aborted originals with identical bodies keep two distinct bounds."""
    audit, outcome, digest, wire = audited_lane(tmp_path, "chat")
    doubled = {**audit, "outcomes": [outcome, dict(outcome)]}
    explained = explained_aborts(tmp_path, doubled)
    assert len(explained[digest]) == 2

    # Two recorded originals with equal bytes must be covered by two attempts.
    (tmp_path / "wire/http.jsonl").write_text((json.dumps(wire) + "\n") * 2)
    failed = model_hop(tmp_path, digest, path=wire["path"])
    write_model_hops(tmp_path, [failed], [failed])
    with pytest.raises(ValueError, match="is missing a recorded attempt"):
        lane_transport_evidence(tmp_path, explained={"model": explained})

    write_model_hops(tmp_path, [failed, failed], [failed, failed])
    evidence = lane_transport_evidence(tmp_path, explained={"model": explained})
    assert evidence["bridge-model"]["explained_aborts"] == 2
    assert evidence["guest-model"]["explained_aborts"] == 2
    assert evidence["unused_explanations"] == []
