"""Reject incorrect runtime context anchors and ambiguous compaction evidence."""

import copy
from datetime import datetime

import pytest

from benchmarks.agentic_session_evidence import (
    audit_context_usage,
    compaction_continuations,
    match_summary,
    rpc_completions,
    sdk_completion_events,
    tool_output,
)
from benchmarks.agentic_soak_transport import completed_transport_close

TIMESTAMP = "2026-10-12T00:00:01Z"
TIME = datetime.fromisoformat(TIMESTAMP.replace("Z", "+00:00")).timestamp()


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_turn_end_repeat_does_not_replace_or_duplicate_sdk_completion(api, duplicate):
    from test_agentic_soak_transport import evidence

    _, _, request = evidence(api)
    identity = request["client_completion"]["response_id"]
    message = {"role": "assistant", "responseId": identity}
    completion = {
        "type": "message_end",
        "message": message,
        "message_sha256": request["client_completion"]["message_sha256"],
    }
    # OMP emits the same assistant message in its normal turn_end event.
    events = [completion, {"type": "turn_end", "message": message, "toolResults": []}]
    if duplicate:
        events.append(copy.deepcopy(completion))
        with pytest.raises(ValueError, match="duplicate RPC"):
            rpc_completions(events)
    else:
        assert rpc_completions(events) == {identity: message}
    matched = sdk_completion_events(events, identity)
    request["client_completion"].update(
        message_count=len(matched), message_sha256=matched[0]["message_sha256"]
    )
    assert completed_transport_close(request, identity, api) is not duplicate
    assert not sdk_completion_events(events[1:2], identity)


def anchor(tokens=100, epoch=0, timestamp=TIMESTAMP):
    return {
        "timestamp": timestamp,
        "message": {
            "responseId": "response-1",
            "usage": {"input": tokens - 80, "cacheRead": 80, "output": 40},
            "contextSnapshot": {"promptTokens": tokens, "compactionEpoch": epoch},
        },
    }


def state(tokens=100, collected=TIME + 1):
    return {
        "collected_unix_s": collected,
        "isSettled": True,
        "contextUsage": {
            "contextWindow": 262144,
            "tokens": tokens,
            "percent": 100 * tokens / 262144,
        },
    }


def test_settled_display_uses_provider_input_without_newly_generated_output():
    observations = audit_context_usage([state()], {"response-1": anchor()}, [])
    assert observations[0]["source"] == "provider_input"
    adjusted = audit_context_usage([state(140)], {"response-1": anchor()}, [])
    assert adjusted[0]["source"] == "provider_input_with_local_adjustments"
    assert adjusted[0]["local_adjustment_tokens"] == 40


@pytest.mark.parametrize(
    "mutation", ["window", "percent", "negative", "nan", "missing"]
)
def test_incorrect_omp_runtime_usage_cannot_pass(mutation):
    sample = state()
    if mutation == "window":
        sample["contextUsage"]["contextWindow"] = 65536
    elif mutation == "percent":
        sample["contextUsage"]["percent"] += 1
    elif mutation == "negative":
        sample["contextUsage"]["tokens"] = -1
    elif mutation == "nan":
        sample["contextUsage"]["tokens"] = float("nan")
    else:
        sample["contextUsage"] = None
    with pytest.raises(ValueError, match="OMP"):
        audit_context_usage([sample], {"response-1": anchor()}, [])


def test_compaction_rebases_display_until_new_provider_response():
    compact = {"id": "compact-1", "timestamp": "2026-10-12T00:00:02Z"}
    observations = audit_context_usage(
        [state(30, TIME + 2)], {"response-1": anchor()}, [compact]
    )
    assert observations[0]["source"] == "local_estimate"
    next_anchor = anchor(50, 1, "2026-10-12T00:00:03Z")
    next_anchor["message"]["responseId"] = "response-2"
    observations = audit_context_usage(
        [state(50, TIME + 3)],
        {"response-1": anchor(), "response-2": next_anchor},
        [compact],
    )
    assert observations[0]["source"] == "provider_input"
    assert observations[0]["response_id"] == "response-2"


def test_compaction_entry_alone_does_not_prove_continuation():
    entry = {"id": "compact-1", "timestamp": TIMESTAMP}
    assert compaction_continuations(
        [entry],
        [{"purpose": "compaction_summary", "processing_started_unix_s": TIME + 1}],
    ) == [{"entry_id": "compact-1", "continued": False}]
    assert compaction_continuations(
        [entry], [{"purpose": "task", "processing_started_unix_s": TIME + 1}]
    )[0]["continued"]


def summary(tokens, identity):
    return {
        "id": identity,
        "timestamp": TIMESTAMP,
        "purpose": "compaction_summary",
        "api": "openai-completions",
        "provider": "oh-my-vllm-chat",
        "model": "qwen3.8-27b-fp8",
        "stopReason": "stop",
        "usage": {"input": tokens - 80, "cacheRead": 80, "output": 30},
    }


def record():
    return {
        "response_ids": ["response-1"],
        "protocol_completion": {"finish_reason": "stop"},
        "expected_provider": "oh-my-vllm-chat",
        "upstream_started_unix_s": TIME - 1,
        "sdk_turn_completed_unix_s": TIME + 1,
    }


def test_split_turn_summary_completion_order_does_not_change_association():
    entries = [summary(200, "second"), summary(100, "first")]
    matched = match_summary(
        entries, record(), {"input_tokens": 100}, 30, "openai-completions"
    )
    assert matched["id"] == "first"
    assert [item["id"] for item in entries] == ["second"]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "stop_reason", "lifecycle", "provider"]
)
def test_ambiguous_or_inconsistent_summary_is_rejected(mutation):
    item = summary(100, "first")
    entries = [item]
    if mutation == "duplicate":
        entries.append(copy.deepcopy(item))
    elif mutation == "stop_reason":
        item["stopReason"] = "length"
    elif mutation == "lifecycle":
        item["timestamp"] = "2026-10-12T00:05:00Z"
    else:
        item["provider"] = "other"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        match_summary(
            entries, record(), {"input_tokens": 100}, 30, "openai-completions"
        )


def test_tool_results_accept_text_parts_without_losing_read_content():
    assert tool_output([{"type": "text", "text": "source"}]) == "source"
    assert tool_output("source") == "source"


def test_each_completed_work_item_needs_its_own_forwarded_source_read(tmp_path):
    from benchmarks.agentic_session_evidence import work_item_reads

    starts = {
        "read-1": {"prompt_id": "turn-1", "args": {"path": "crates/source.rs:1-40"}},
        "read-2": {"prompt_id": "turn-2", "args": {"path": "notes/progress.md"}},
    }
    turns = [{"prompt_id": "turn-1"}, {"prompt_id": "turn-2"}]
    tools, forwarded = {key: {} for key in starts}, set(starts)
    with pytest.raises(ValueError, match="work item lacks"):
        work_item_reads(starts, tools, forwarded, turns, tmp_path)
    starts["read-2"]["args"]["path"] = "tests/test_source.py"
    verified = work_item_reads(starts, tools, forwarded, turns, tmp_path)
    assert len(verified) == 2
    assert all(item["forwarded_source_reads"] == 1 for item in verified)
