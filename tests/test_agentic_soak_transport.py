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
