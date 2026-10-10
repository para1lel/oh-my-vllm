"""A legal token limit needs complete SDK, SSE and declared-budget evidence."""

import copy
import hashlib
import json
from functools import partial

import pytest
from test_speculative_service_evidence import agent_evidence

from benchmarks.agentic_soak import agentic
from benchmarks.agentic_soak_protocol import (
    StreamCompletion,
    attach_client_completions,
    certify_length_completions,
    completion_counts,
    controlled_length_completion,
)


def evidence(api, prefix=None):
    events, requests = agent_evidence(api)
    final, request = events[-1]["message"], requests[0]
    final.update(stopReason="length", usage={"output": 2})
    request["request"].update(
        stream=True, **{"max_tokens" if api == "chat" else "max_output_tokens": 2}
    )
    rid, model = final["responseId"], final["model"]
    if api == "chat":
        payloads = [
            {
                "id": rid,
                "model": model,
                "choices": [{"index": 0, "finish_reason": "length"}],
            },
            {
                "id": rid,
                "model": model,
                "choices": [],
                "usage": {"completion_tokens": 2},
            },
            "[DONE]",
        ]
    else:
        payloads = [
            {
                "type": "response.incomplete",
                "response": {
                    "id": rid,
                    "model": model,
                    "status": "incomplete",
                    "error": None,
                    "max_output_tokens": 2,
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"output_tokens": 2},
                    "output": [],
                },
            }
        ]
    raw = b"".join(
        b"data: "
        + (item if isinstance(item, str) else json.dumps(item)).encode()
        + b"\n\n"
        for item in ([prefix] if prefix is not None else []) + payloads
    )
    parser = StreamCompletion(api, 65536)
    for byte in raw:
        parser.feed(bytes([byte]))
    parser.terminal_forwarded = True
    request.update(
        protocol_completion=parser.snapshot(),
        transport_outcome="completed",
        response_bytes=len(raw),
        response_sha256=hashlib.sha256(raw).hexdigest(),
    )
    attach_client_completions(events, requests, api)
    return events, requests


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_length_remains_strict_by_default_but_soak_certifies_all_three_budget_counts(
    api,
):
    events, requests = evidence(api)
    before = copy.deepcopy(events)
    with pytest.raises(ValueError, match="final answer"):
        agentic.validate_agent_evidence(events, requests, api)
    result = agentic.validate_agent_evidence(
        events,
        requests,
        api,
        length_completion_check=partial(controlled_length_completion, api=api),
    )
    assert result["final_stop_reason"] == "length"
    proof = result["controlled_length_completion"]
    assert (
        proof["declared_output_budget"]
        == proof["sdk_output_tokens"]
        == proof["sse_output_tokens"]
        == 2
    )
    assert result["tool_results_in_final_model_request"] is True
    assert result["successful_source_reads"]
    assert requests[0]["controlled_length_completion"] == proof
    assert events == before


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("budget", [None, False, True, 0, -1, 2.0, "2", 3])
def test_length_rejects_unproven_or_different_declared_limits(api, budget):
    events, requests = evidence(api)
    requests[0]["request"]["max_tokens" if api == "chat" else "max_output_tokens"] = (
        budget
    )
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api=api)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "section,key,value",
    [
        ("protocol_completion", "output_tokens", None),
        ("protocol_completion", "output_tokens", True),
        ("protocol_completion", "output_tokens", 1),
        ("protocol_completion", "model", "wrong-model"),
        ("protocol_completion", "response_id", "resp_other"),
        ("protocol_completion", "success", False),
        ("protocol_completion", "errors", ["truncated"]),
        ("protocol_completion", "terminal_kind", None),
        ("protocol_completion", "raw_terminal_offset", False),
        ("client_completion", "output_tokens", 1),
        ("client_completion", "successful", False),
        ("client_completion", "api", "other"),
        ("client_completion", "response_id", "resp_other"),
        ("client_completion", "message_count", 2),
        ("client_completion", "message_sha256", "0" * 64),
    ],
)
def test_length_rejects_partial_mismatched_or_ambiguous_proofs(
    api, section, key, value
):
    events, requests = evidence(api)
    requests[0][section][key] = value
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api=api)


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "text", "text": "  "}],
        [{"type": "text", "text": "An answer"}, {"type": "toolCall", "name": "read"}],
    ],
)
def test_length_rejects_empty_text_and_unresolved_tools(api, content):
    events, requests = evidence(api)
    events[-1]["message"]["content"] = content
    attach_client_completions(events, requests, api)
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api=api)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_certified_length_does_not_replace_real_source_reads_or_forwarding(api):
    events, requests = evidence(api)
    check = partial(controlled_length_completion, api=api)
    events[1]["isError"] = True
    with pytest.raises(ValueError, match="successfully read"):
        agentic.validate_agent_evidence(
            events, requests, api, length_completion_check=check
        )
    events, requests = evidence(api)
    requests[0]["request"]["messages" if api == "chat" else "input"].pop(0)
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(
            events, requests, api, length_completion_check=check
        )


def test_intermediate_length_calls_are_certified_and_counted_without_relabeling():
    events, requests = evidence("responses")
    certify_length_completions(events, requests, "responses")
    assert requests[0]["protocol_completion"]["terminal_kind"] == "responses_incomplete"
    tasks = [
        {
            "api": "responses",
            "requests": requests,
            "tool_verification": {"final_stop_reason": "length"},
        },
        {
            "api": "chat",
            "requests": [],
            "tool_verification": {"final_stop_reason": "stop"},
        },
    ]
    assert completion_counts(tasks) == {
        "normal_length_final_agent_tasks": 1,
        "normal_length_model_requests": 1,
        "normal_length_model_requests_by_api": {"chat": 0, "responses": 1},
    }
    events.append(copy.deepcopy(events[-1]))
    attach_client_completions(events, requests, "responses")
    with pytest.raises(ValueError, match="ambiguous SDK"):
        certify_length_completions(events, requests, "responses")


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_raw_partial_tool_call_cannot_be_hidden_by_missing_sdk_tool_object(api):
    prefix = (
        {
            "id": "chatcmpl_test_17",
            "model": "qwen3.8-27b-fp8",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"path":'}}
                        ]
                    },
                }
            ],
        }
        if api == "chat"
        else {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_partial",
                "name": "read",
                "arguments": "",
            },
        }
    )
    events, requests = evidence(api, prefix)
    assert requests[0]["protocol_completion"]["pending_tool_call_count"] == 1
    assert not any(c["type"] == "toolCall" for c in events[-1]["message"]["content"])
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api=api)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_model_identity_must_match_all_sse_frames_not_only_last_frame(api):
    prefix = (
        {"id": "chatcmpl_test_17", "model": "other", "choices": []}
        if api == "chat"
        else {
            "type": "response.created",
            "response": {"id": "resp_test_17", "model": "other"},
        }
    )
    events, requests = evidence(api, prefix)
    assert "response_model_mismatch" in requests[0]["protocol_completion"]["errors"]
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api=api)


@pytest.mark.parametrize("kind", ["delta", "done"])
def test_raw_function_arguments_without_added_item_still_prevent_text_length_proof(
    kind,
):
    prefix = {
        "type": "response.function_call_arguments." + kind,
        "output_index": 0,
        "item_id": "fc_partial",
        "delta": '{"path":',
    }
    events, requests = evidence("responses", prefix)
    assert requests[0]["protocol_completion"]["pending_tool_call_count"] == 1
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(
            events[-1]["message"], requests[0], api="responses"
        )


@pytest.mark.parametrize("calls", [["partial"], {"index": 0}])
def test_malformed_chat_tool_fragments_cannot_be_certified_as_text_only(calls):
    prefix = {
        "id": "chatcmpl_test_17",
        "model": "qwen3.8-27b-fp8",
        "choices": [{"index": 0, "delta": {"tool_calls": calls}}],
    }
    events, requests = evidence("chat", prefix)
    assert requests[0]["protocol_completion"]["success"] is False
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api="chat")


def test_response_limit_echo_must_match_request():
    events, requests = evidence("responses")
    requests[0]["protocol_completion"]["declared_output_budget"] = 3
    with pytest.raises(ValueError):
        controlled_length_completion(
            events[-1]["message"], requests[0], api="responses"
        )


def test_chat_sdk_budget_alias_matches_and_conflicting_aliases_fail():
    events, requests = evidence("chat")
    body = requests[0]["request"]
    body["max_completion_tokens"] = body.pop("max_tokens")
    assert (
        controlled_length_completion(events[-1]["message"], requests[0], api="chat")[
            "declared_output_budget"
        ]
        == 2
    )
    body["max_tokens"] = 3
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api="chat")
    body["max_tokens"] = 2.0
    with pytest.raises(ValueError, match="budget/protocol/client"):
        controlled_length_completion(events[-1]["message"], requests[0], api="chat")


def test_sdk_output_token_count_must_be_integer_not_boolean():
    events, requests = evidence("responses")
    events[-1]["message"]["usage"]["output"] = True
    attach_client_completions(events, requests, "responses")
    with pytest.raises(ValueError):
        controlled_length_completion(
            events[-1]["message"], requests[0], api="responses"
        )
