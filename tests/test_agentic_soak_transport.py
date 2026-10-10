"""A completed SDK turn and server metric distinguish closes from truncation."""

import copy
import hashlib

import pytest
from test_agentic_soak import metric, task

from benchmarks.agentic_soak_analysis import bind_points


def evidence(api="chat", reason="tool_calls"):
    model = metric(api=api)
    agent = task(api=api)
    request = agent["requests"][0]
    request.update(
        error="ConnectionResetError: [Errno 104] Connection reset by peer",
        error_direction="downstream_write",
        transport_error_kind="ConnectionResetError",
        transport_outcome="terminal_disconnect",
        response_bytes=3371,
        protocol_completion={
            "api": api,
            "response_id": model["response_id"],
            "success": True,
            "terminal_kind": "chat_done"
            if api == "chat"
            else "responses_incomplete"
            if reason == "length"
            else "responses_completed",
            "finish_reason": reason,
            "raw_terminal_offset": 3371,
            "terminal_forwarded": False,
            "errors": [],
        },
        client_completion={
            "response_id": model["response_id"],
            "api": "openai-completions" if api == "chat" else "openai-responses",
            "model": "qwen3.8-27b-fp8",
            "stop_reason": {
                "stop": "stop",
                "tool_calls": "toolUse",
                "length": "length",
            }[reason],
            "message_count": 1,
            "successful": True,
            "message_sha256": hashlib.sha256(b"client message_end").hexdigest(),
        },
    )
    if reason == "length":
        request["response_sha256"] = hashlib.sha256(b"original response").hexdigest()
        request["client_completion"]["output_tokens"] = model["output_tokens"]
        protocol = request["protocol_completion"]
        protocol.update(
            model="qwen3.8-27b-fp8",
            output_tokens=model["output_tokens"],
            pending_tool_call_count=0,
        )
        if api == "responses":
            protocol["declared_output_budget"] = model["output_tokens"]
        request["controlled_length_completion"] = {
            "response_id": model["response_id"],
            "api": api,
            "model": "qwen3.8-27b-fp8",
            "declared_output_budget": model["output_tokens"],
            "sdk_output_tokens": model["output_tokens"],
            "sse_output_tokens": model["output_tokens"],
            "client_message_sha256": request["client_completion"]["message_sha256"],
            "response_sha256": request["response_sha256"],
            "nonempty_text": True,
            "no_tool_call": True,
            "completed_server_metric_required": True,
            "stop_reason": "length",
            "terminal_kind": protocol["terminal_kind"],
            "pending_tool_call_count": 0,
            "raw_terminal_offset": protocol["raw_terminal_offset"],
            "response_bytes": request["response_bytes"],
        }
    return model, agent, request


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("reason", ["stop", "tool_calls", "length"])
def test_completed_text_tool_and_length_turns_keep_the_original_error(api, reason):
    model, agent, request = evidence(api, reason)
    original = copy.deepcopy(request)
    points = bind_points([model], [agent], 100)
    assert len(points) == 1
    assert points[0]["status"] == "completed"
    assert points[0]["output_tps"] == pytest.approx(20 / 3)
    assert points[0]["post_terminal_transport_close"]
    assert request == original


@pytest.mark.parametrize("direction", ["upstream_read", "downstream_write"])
@pytest.mark.parametrize("kind", ["ConnectionResetError", "BrokenPipeError"])
def test_transport_close_after_all_three_completions_is_diagnostic(direction, kind):
    model, agent, request = evidence()
    request.update(
        error_direction=direction, transport_error_kind=kind, error=kind + ": closed"
    )
    assert bind_points([model], [agent], 100)[0]["post_terminal_transport_close"]


@pytest.mark.parametrize(
    "scope,key,value",
    [
        ("request", "error_direction", "upstream_headers"),
        ("request", "error_direction", {}),
        ("request", "transport_error_kind", "TimeoutError"),
        ("request", "transport_error_kind", {}),
        ("request", "transport_outcome", "failed"),
        ("protocol_completion", "success", False),
        ("protocol_completion", "finish_reason", {}),
        ("protocol_completion", "response_id", "another-response"),
        ("protocol_completion", "api", "responses"),
        ("protocol_completion", "terminal_kind", "response.failed"),
        ("protocol_completion", "errors", ["explicit SSE error"]),
        ("protocol_completion", "raw_terminal_offset", 3372),
        ("client_completion", "successful", False),
        ("client_completion", "response_id", "another-response"),
        ("client_completion", "api", "openai-responses"),
        ("client_completion", "model", "another-model"),
        ("client_completion", "stop_reason", "error"),
        ("client_completion", "stop_reason", "stop"),
        ("client_completion", "message_count", 2),
        ("client_completion", "message_count", True),
        ("client_completion", "message_sha256", "not-a-message-digest"),
    ],
)
def test_truncated_wrong_or_ambiguous_completion_evidence_still_fails(
    scope, key, value
):
    model, agent, request = evidence()
    target = request if scope == "request" else request[scope]
    target[key] = value
    with pytest.raises(ValueError, match="normal model request failed"):
        bind_points([model], [agent], 100)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_protocol_and_client_success_cannot_replace_completed_server_state(status):
    model, agent, _ = evidence()
    model["status"] = status
    with pytest.raises(ValueError, match="no completed server metric"):
        bind_points([model], [agent], 100)


def test_incomplete_protocol_requires_the_matching_length_completion():
    model, agent, request = evidence("responses", "stop")
    request["protocol_completion"]["terminal_kind"] = "responses_incomplete"
    with pytest.raises(ValueError, match="normal model request failed"):
        bind_points([model], [agent], 100)


def test_existing_healthy_request_needs_no_transport_error_override():
    point = bind_points([metric()], [task()], 100)[0]
    assert not point["post_terminal_transport_close"]


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_controlled_length_without_transport_error_keeps_its_original_reason(api):
    model, agent, request = evidence(api, "length")
    request.pop("error")
    original = copy.deepcopy(request)
    point = bind_points([model], [agent], 100)[0]
    assert point["controlled_length_completion"]
    assert not point["post_terminal_transport_close"]
    assert request == original


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "scope,key,value",
    [
        ("request", "controlled_length_completion", None),
        ("request", "controlled_length_completion", True),
        ("request", "controlled_length_completion", {}),
        ("request", "response_sha256", "bad-digest"),
        ("request", "response_bytes", True),
        ("proof", "response_id", "another-response"),
        ("proof", "api", "unknown"),
        ("proof", "model", "other-model"),
        ("proof", "declared_output_budget", True),
        ("proof", "declared_output_budget", 20.0),
        ("proof", "declared_output_budget", 0),
        ("proof", "declared_output_budget", 21),
        ("proof", "sdk_output_tokens", 19),
        ("proof", "sse_output_tokens", 21),
        ("proof", "client_message_sha256", "a" * 64),
        ("proof", "response_sha256", "a" * 64),
        ("proof", "nonempty_text", None),
        ("proof", "nonempty_text", 1),
        ("proof", "no_tool_call", False),
        ("proof", "completed_server_metric_required", None),
        ("proof", "stop_reason", "stop"),
        ("proof", "terminal_kind", "response.failed"),
        ("proof", "pending_tool_call_count", None),
        ("proof", "pending_tool_call_count", False),
        ("proof", "pending_tool_call_count", 1),
        ("proof", "raw_terminal_offset", None),
        ("proof", "raw_terminal_offset", True),
        ("proof", "raw_terminal_offset", 3370),
        ("proof", "response_bytes", 3372),
        ("client", "output_tokens", None),
        ("client", "output_tokens", True),
        ("client", "output_tokens", 19),
        ("client", "output_tokens", 21),
        ("protocol", "output_tokens", True),
        ("protocol", "output_tokens", 19),
        ("protocol", "model", "other-model"),
        ("protocol", "pending_tool_call_count", None),
        ("protocol", "pending_tool_call_count", False),
        ("protocol", "pending_tool_call_count", 1),
        ("protocol", "raw_terminal_offset", 0),
        ("protocol", "raw_terminal_offset", 3372),
        ("metric", "output_tokens", True),
        ("metric", "output_tokens", 19),
        ("metric", "output_tokens", 21),
    ],
)
def test_length_budget_identity_and_independent_metric_must_agree(
    api, scope, key, value
):
    model, agent, request = evidence(api, "length")
    request.pop("error")
    target = {
        "request": request,
        "proof": request["controlled_length_completion"],
        "protocol": request["protocol_completion"],
        "client": request["client_completion"],
        "metric": model,
    }[scope]
    target[key] = value
    with pytest.raises(ValueError, match="controlled length proof differs"):
        bind_points([model], [agent], 100)


def test_response_echo_budget_must_match_original_request_and_server():
    model, agent, request = evidence("responses", "length")
    request.pop("error")
    request["protocol_completion"]["declared_output_budget"] = 21
    with pytest.raises(ValueError, match="controlled length proof differs"):
        bind_points([model], [agent], 100)


def test_unknown_api_cannot_certify_a_controlled_length():
    from benchmarks.agentic_soak_budget import validate_length_metric

    model, _, request = evidence("responses", "length")
    model["api"] = "unknown"
    request["protocol_completion"]["api"] = "unknown"
    request["controlled_length_completion"]["api"] = "unknown"
    with pytest.raises(ValueError, match="controlled length proof differs"):
        validate_length_metric(request, model, "unknown")
