"""Bind persistent-session requests, SDK completions, tools and server metrics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from benchmarks.agentic_context import CONTEXT_WINDOW
from benchmarks.agentic_sdk_trace import read_sdk_spans
from benchmarks.agentic_soak_analysis import request_metric
from benchmarks.agentic_soak_transport import completed_transport_close


def sdk_input(message: dict) -> int:
    values = [
        message["usage"].get(key, 0) for key in ("input", "cacheRead", "cacheWrite")
    ]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("OMP input usage has invalid token counts")
    return sum(values)


def entry_time(entry: dict) -> float:
    return datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")).timestamp()


def sdk_completion_events(events: list[dict], identity: str) -> list[dict]:
    """A turn-end event can repeat a message without another SDK completion."""
    return [
        event
        for event in events
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
        and event["message"].get("responseId") == identity
    ]


def rpc_completions(events: list[dict]) -> dict:
    """Reject repeated SDK completions for healthy and disconnected streams."""
    messages = {}
    for event in events:
        message = event.get("message", {})
        if event.get("type") != "message_end" or message.get("role") != "assistant":
            continue
        identity = message.get("responseId")
        if not identity or identity in messages:
            raise ValueError("missing or duplicate RPC SDK completion identity")
        messages[identity] = message
    return messages


def match_summary(
    entries: list[dict], record: dict, proof: dict, output: int, api: str
):
    """Parallel split-turn summaries must match usage and lifecycle uniquely."""
    reason = {"stop": "stop", "tool_calls": "toolUse", "length": "length"}.get(
        record["protocol_completion"]["finish_reason"]
    )
    candidates = [
        entry
        for entry in entries
        if (entry.get("api") or entry.get("api_binding")) == api
        and entry.get("responseId", record["response_ids"][0])
        == record["response_ids"][0]
        and entry.get("provider") == record["expected_provider"]
        and entry.get("model") == "qwen3.8-27b-fp8"
        and entry.get("stopReason") == reason
        and ("compaction" in entry["purpose"] or "summary" in entry["purpose"])
        and sdk_input(entry) == proof["input_tokens"]
        and entry["usage"]["output"] == output
        and record["upstream_started_unix_s"] - 1
        <= entry_time(entry)
        <= record["sdk_turn_completed_unix_s"] + 1
    ]
    if len(candidates) != 1:
        raise ValueError("summary SDK usage match is missing or ambiguous")
    result = candidates[0]
    entries.remove(result)
    return result


def trace_messages(directory: Path, session: dict) -> list[dict]:
    """Keep native SDK result fields distinct from the configured API binding."""
    binding = session["sdk_trace"]
    expected_api = (
        "openai-completions" if session["api"] == "chat" else "openai-responses"
    )
    if (
        binding["directory"] != "sdk-trace"
        or binding["source"] != "omp_sdk_otlp"
        or binding["api_binding_source"] != "isolated_provider_configuration"
        or binding["api_binding"] != expected_api
    ):
        raise ValueError("SDK trace configuration binding differs")
    if set(binding["configuration_sha256"]) != {"models.yml", "settings.yml"}:
        raise ValueError("SDK provider configuration identities are incomplete")
    for name, digest in binding["configuration_sha256"].items():
        if (
            name not in ("models.yml", "settings.yml")
            or hashlib.sha256((directory / "state" / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError("SDK provider configuration hash differs")
    models = json.loads((directory / "state/models.yml").read_text())
    provider = models["providers"][f"oh-my-vllm-{session['api']}"]
    if provider["api"] != expected_api:
        raise ValueError("SDK provider API binding differs")
    states = json.loads((directory / "states.json").read_text())
    if not states or any(
        state["sessionId"] != session["session_id"]
        or state["model"].get("api") != expected_api
        or state["model"].get("provider") != f"oh-my-vllm-{session['api']}"
        or state["model"].get("id") != "qwen3.8-27b-fp8"
        for state in states
    ):
        raise ValueError("SDK runtime API binding differs")
    messages, identities = [], set()
    for span in read_sdk_spans(directory / "sdk-trace"):
        values = span["attributes"]
        purpose = values.get("omp.gen_ai.oneshot.kind", "task")
        counts = [
            values.get(key, 0)
            for key in (
                "gen_ai.usage.cache_read.input_tokens",
                "gen_ai.usage.cache_creation.input_tokens",
                "gen_ai.usage.reasoning.output_tokens",
            )
        ]
        budget = values.get("gen_ai.request.max_tokens")
        identity = values.get("gen_ai.response.id")
        input_tokens = values.get("gen_ai.usage.input_tokens")
        output_tokens = values.get("gen_ai.usage.output_tokens")
        reason = values.get("gen_ai.response.finish_reasons")
        if (
            span["scope"] != "@oh-my-pi/pi-agent-core"
            or span["status_code"] not in (0, 1)
            or not identity
            or identity in identities
            or values.get("gen_ai.conversation.id") != session["session_id"]
            or values.get("gen_ai.provider.name") != f"oh-my-vllm-{session['api']}"
            or values.get("gen_ai.response.model") != "qwen3.8-27b-fp8"
            or type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
            or reason not in (["stop"], ["tool_calls"], ["length"])
            or values.get("gen_ai.request.model") != "qwen3.8-27b-fp8"
            or values.get("omp.gen_ai.request.reasoning.effort") != "max"
            or purpose
            not in (
                "task",
                "compaction_summary",
                "compaction_short_summary",
                "compaction_turn_prefix",
            )
            or any(type(value) is not int or value < 0 for value in counts)
            or sum(counts[:2]) > input_tokens
            or counts[2] > output_tokens
            or type(values.get("omp.gen_ai.usage.total_tokens")) is not int
            or values["omp.gen_ai.usage.total_tokens"] != input_tokens + output_tokens
            or (
                budget is not None
                and (type(budget) is not int or not 0 < budget <= 32768)
            )
            or (purpose != "task" and budget is None)
            or not re.fullmatch(r"[0-9a-f]{32}", span["trace_id"])
            or int(span["trace_id"], 16) == 0
            or not re.fullmatch(r"[0-9a-f]{16}", span["span_id"])
            or int(span["span_id"], 16) == 0
            or not math.isfinite(span["started_unix_s"])
            or not math.isfinite(span["completed_unix_s"])
            or not 0 < span["started_unix_s"] <= span["completed_unix_s"]
        ):
            raise ValueError("SDK result trace is incomplete or inconsistent")
        identities.add(identity)
        messages.append(
            {
                "source": span["source"],
                "id": span["span_id"],
                "responseId": identity,
                "session_id": session["session_id"],
                "timestamp": datetime.fromtimestamp(
                    span["completed_unix_s"], tz=UTC
                ).isoformat(),
                "purpose": purpose,
                "provider": values["gen_ai.provider.name"],
                "model": values["gen_ai.response.model"],
                "api_binding": expected_api,
                "api_binding_source": binding["api_binding_source"],
                "stopReason": {
                    "stop": "stop",
                    "tool_calls": "toolUse",
                    "length": "length",
                }[reason[0]],
                "usage": {"input": input_tokens, "output": output_tokens},
                "sdk_trace": span,
            }
        )
    return messages


def tool_output(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item.get("text", "")
            for item in value
            if item.get("type") in ("text", "input_text")
        )
    return ""


def compaction_continuations(compactions: list[dict], points: list[dict]) -> list[dict]:
    return [
        {
            "entry_id": entry["id"],
            "continued": any(
                point["purpose"] == "task"
                and point["processing_started_unix_s"] >= entry_time(entry)
                for point in points
            ),
        }
        for entry in compactions
    ]


def audit_context_usage(
    states: list[dict], anchors: dict, compactions: list[dict]
) -> list[dict]:
    """Check provider anchors and classify estimates after history compaction."""
    observations = []
    ordered = sorted(anchors.values(), key=entry_time)
    epochs = [item["message"]["contextSnapshot"]["compactionEpoch"] for item in ordered]
    if any(right < left for left, right in pairwise(epochs)):
        raise ValueError("OMP context snapshot epoch moved backwards")
    for state in states:
        usage = state.get("contextUsage")
        previous = [
            item for item in ordered if entry_time(item) <= state["collected_unix_s"]
        ]
        if usage is None:
            if previous:
                raise ValueError(
                    "OMP lost runtime context usage after a model response"
                )
            continue
        tokens = usage.get("tokens")
        percent = usage.get("percent")
        if (
            usage.get("contextWindow") != CONTEXT_WINDOW
            or isinstance(tokens, bool)
            or not isinstance(tokens, (int, float))
            or not math.isfinite(tokens)
            or not 0 <= tokens <= CONTEXT_WINDOW
            or not isinstance(percent, (int, float))
            or not math.isclose(percent, 100 * tokens / CONTEXT_WINDOW, abs_tol=1e-8)
        ):
            raise ValueError("OMP runtime context window or percentage differs")
        anchor = previous[-1] if previous else None
        rewritten = anchor and any(
            entry_time(anchor) <= entry_time(item) <= state["collected_unix_s"]
            for item in compactions
        )
        source = "local_estimate" if not anchor or rewritten else "provider_input"
        adjustment = (
            tokens - anchor["message"]["contextSnapshot"]["promptTokens"]
            if source == "provider_input"
            else None
        )
        if adjustment:
            source = "provider_input_with_local_adjustments"
        observations.append(
            {
                "at_unix_s": state["collected_unix_s"],
                "usage": usage,
                "source": source,
                "local_adjustment_tokens": adjustment,
                "response_id": anchor["message"].get("responseId") if anchor else None,
            }
        )
    return observations


def text_content(message: dict) -> str:
    return "\n".join(
        item.get("text", "")
        for item in message.get("content", [])
        if item.get("type") == "text"
    )


def work_item_reads(
    read_starts: dict,
    successful_tools: dict,
    forwarded: set,
    turns: list[dict],
    workspace: Path,
    *,
    goal_mode: bool = False,
) -> list[dict]:
    """Each completed audit item must read immutable source, not only notes."""
    verified = []
    for turn in turns:
        calls = []
        for identity in set(read_starts) & set(successful_tools) & forwarded:
            event = read_starts[identity]
            if event.get("prompt_id") != turn["prompt_id"]:
                continue
            filename = event.get("args", {}).get("path", "")
            filename = re.sub(r":\d+(?:-\d+)?$", "", filename)
            try:
                relative = (
                    (workspace / filename).resolve().relative_to(workspace.resolve())
                )
            except ValueError:
                continue
            if relative.parts and relative.parts[0] in (
                "crates",
                "python",
                "tests",
                "docs",
                "benchmarks",
                *(("TASK.md", "goaldb") if goal_mode else ()),
            ):
                calls.append(identity)
        if not calls and not goal_mode:
            raise ValueError("completed audit work item lacks forwarded source reads")
        verified.append(
            {"prompt_id": turn["prompt_id"], "forwarded_source_reads": len(calls)}
        )
    return verified


def terminal_usage(path: Path, api: str) -> dict:
    usage = None
    with path.open("rb") as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            value = (
                event.get("usage")
                if api == "chat"
                else event.get("response", {}).get("usage")
            )
            if value:
                usage = value
    if usage is None:
        raise ValueError("completed stream has no token usage")
    return {
        "input": usage["prompt_tokens" if api == "chat" else "input_tokens"],
        "output": usage["completion_tokens" if api == "chat" else "output_tokens"],
    }


def audit_session(
    directory: Path, session: dict, metrics: dict, measured_start: float
) -> tuple[list[dict], dict]:
    """Read long request bodies one at a time; do not retain repeated histories."""
    events = json.loads((directory / "events.json").read_text())
    states = json.loads((directory / "states.json").read_text())
    turns = json.loads((directory / "turns.json").read_text())
    if session.get("goal_id"):
        expected = {turn["prompt_id"]: turn["goal_id"] for turn in turns}
        for event in events:
            goal = event.get("goal")
            if (
                event.get("type") == "goal_updated"
                and goal
                and goal.get("id") != expected.get(event.get("prompt_id"))
            ):
                raise ValueError("native goal event has an uncommanded identity")
    entries, anchors = [], {}
    with (directory / session["entries_file"]).open() as journal:
        for line in journal:
            entry = json.loads(line)
            if entry["type"] in ("compaction", "model_usage"):
                entries.append(entry)
            elif (
                entry["type"] == "message"
                and entry["message"].get("role") == "assistant"
            ):
                message = entry["message"]
                if message.get("stopReason") in ("stop", "toolUse", "length"):
                    identity = message.get("responseId")
                    if not identity or identity in anchors:
                        raise ValueError(
                            "OMP journal has missing or duplicate response identity"
                        )
                    anchors[identity] = {
                        **entry,
                        "message": {
                            key: value
                            for key, value in message.items()
                            if key != "content"
                        },
                    }
    messages = rpc_completions(events)
    extra_usage = [entry for entry in entries if entry["type"] == "model_usage"]
    traced = trace_messages(directory, session)
    traces = {item["responseId"]: item for item in traced}
    extra_usage.extend(item for item in traced if item["purpose"] != "task")
    points = []
    successful_tools = {
        event["toolCallId"]: event
        for event in events
        if event.get("type") == "tool_execution_end" and event.get("isError") is False
    }
    read_starts = {
        event["toolCallId"]: event
        for event in events
        if event.get("type") == "tool_execution_start"
        and event.get("toolName") == "read"
    }
    forwarded, normal_ids = set(), set()
    wire = directory / "wire"
    with (wire / "http.jsonl").open() as records:
        for line in records:
            record = json.loads(line)
            identities = record.get("response_ids", [])
            if record.get("response_status") != 200 or len(identities) != 1:
                raise ValueError(
                    "persistent model request failed or lost response identity"
                )
            identity = identities[0]
            if identity in normal_ids:
                raise ValueError("duplicate persistent response identity")
            normal_ids.add(identity)
            metric = request_metric(metrics[identity])
            terminal = record["protocol_completion"]
            if (
                metric["status"] != "completed"
                or not terminal["success"]
                or terminal["errors"]
            ):
                raise ValueError(
                    "persistent request has no successful server/SSE completion"
                )
            proof = record["context_proof"]
            if (
                proof["total_budget"] != proof["input_tokens"] + proof["output_budget"]
                or proof["total_budget"] > CONTEXT_WINDOW
            ):
                raise ValueError("context proof exceeds actual input/output budget")
            usage = terminal_usage(wire / record["response_file"], session["api"])
            if usage != {
                "input": proof["input_tokens"],
                "output": metric["output_tokens"],
            }:
                raise ValueError(
                    "CPU token count differs from independent serving usage"
                )
            trace = traces.get(identity)
            if (
                trace is None
                or trace["usage"] != usage
                or trace["sdk_trace"]["started_unix_s"]
                > record["upstream_started_unix_s"] + 1
                or trace["sdk_trace"]["completed_unix_s"]
                < record["upstream_started_unix_s"] - 1
                or (
                    "gen_ai.request.max_tokens" in trace["sdk_trace"]["attributes"]
                    and trace["sdk_trace"]["attributes"]["gen_ai.request.max_tokens"]
                    != proof["output_budget"]
                )
            ):
                raise ValueError("wire request lacks matching genuine SDK result trace")
            message = messages.get(identity)
            purpose = "task"
            expected_api = (
                "openai-completions" if session["api"] == "chat" else "openai-responses"
            )
            if message is None:
                matching_turns = [
                    turn for turn in turns if turn["prompt_id"] == record["prompt_id"]
                ]
                if len(matching_turns) != 1:
                    raise ValueError("summary has no settled originating prompt")
                record.update(
                    expected_provider=f"oh-my-vllm-{session['api']}",
                    sdk_turn_completed_unix_s=matching_turns[0]["completed_unix_s"],
                )
                message = match_summary(
                    extra_usage, record, proof, metric["output_tokens"], expected_api
                )
                purpose = message["purpose"]
            if (
                (message.get("api") or message.get("api_binding")) != expected_api
                or message.get("provider") != f"oh-my-vllm-{session['api']}"
                or message.get("model") != "qwen3.8-27b-fp8"
                or message.get("errorMessage")
            ):
                raise ValueError(
                    "SDK completion uses a different model/API or has an error"
                )
            if message.get("stopReason") not in ("stop", "toolUse", "length"):
                raise ValueError("SDK completion has an unsuccessful stop reason")
            if message["stopReason"] != trace["stopReason"] or (purpose == "task") != (
                trace["purpose"] == "task"
            ):
                raise ValueError("SDK result trace differs from the RPC completion")
            if message["stopReason"] != {
                "stop": "stop",
                "tool_calls": "toolUse",
                "length": "length",
            }.get(terminal["finish_reason"]):
                raise ValueError("SDK and SSE stop reasons differ")
            if record.get("error"):
                same_id_events = sdk_completion_events(events, identity)
                record["client_completion"] = {
                    "response_id": identity,
                    "api": message.get("api") or message["api_binding"],
                    "model": message["model"],
                    "stop_reason": message["stopReason"],
                    "message_count": len(same_id_events),
                    "message_sha256": same_id_events[0].get("message_sha256")
                    if same_id_events
                    else None,
                    "successful": not message.get("errorMessage"),
                }
                if purpose != "task":
                    record["client_completion"].pop("api")
                    record["client_completion"].pop("message_count")
                    record["client_completion"].pop("message_sha256")
                    record["client_completion"].update(
                        source="omp_sdk_otlp",
                        api_binding=trace["api_binding"],
                        api_binding_source=trace["api_binding_source"],
                        span_count=1,
                        span_id=trace["sdk_trace"]["span_id"],
                        trace_id=trace["sdk_trace"]["trace_id"],
                        export_file=trace["sdk_trace"]["export_file"],
                        export_sha256=trace["sdk_trace"]["export_sha256"],
                    )
                if not completed_transport_close(record, identity, session["api"]):
                    raise ValueError(
                        "persistent transport lacks independent completion"
                    )
            if message["usage"]["output"] != metric["output_tokens"]:
                raise ValueError("SDK and server output counts differ")
            if sdk_input(message) != proof["input_tokens"]:
                raise ValueError("OMP does not know the actual input token count")
            if purpose == "task":
                persisted = anchors[identity]["message"]
                snapshot = persisted.get("contextSnapshot", {})
                if (
                    snapshot.get("promptTokens") != proof["input_tokens"]
                    or sdk_input(persisted) != proof["input_tokens"]
                    or persisted["usage"]["output"] != metric["output_tokens"]
                ):
                    raise ValueError(
                        "OMP persisted context anchor differs from serving usage"
                    )
            if (
                message["stopReason"] == "length"
                and metric["output_tokens"] != proof["output_budget"]
            ):
                raise ValueError(
                    "length completion differs from requested output budget"
                )
            body = json.loads((wire / record["request_file"]).read_text())
            history = (
                body.get("messages", [])
                if session["api"] == "chat"
                else body.get("input", [])
            )
            for item in history:
                if session["api"] == "chat" and item.get("role") == "tool":
                    call, output = item.get("tool_call_id"), item.get("content", "")
                elif item.get("type") == "function_call_output":
                    call, output = item.get("call_id"), item.get("output", "")
                else:
                    continue
                for client_call, event in successful_tools.items():
                    if call != client_call.split("|", 1)[0]:
                        continue
                    text = text_content(event.get("result", {}))
                    if text and text in tool_output(output):
                        forwarded.add(client_call)
            points.append(
                {
                    **metric,
                    "lane": session["lane"],
                    "session_id": session["session_id"],
                    "purpose": purpose,
                    "input_tokens": proof["input_tokens"],
                    "output_budget": proof["output_budget"],
                    "reasoning_effort": proof["reasoning_effort"],
                    "elapsed_minutes": (
                        metric["processing_started_unix_s"] - measured_start
                    )
                    / 60,
                    "output_tps": metric["output_tokens"]
                    / metric["processing_seconds"],
                }
            )
    if extra_usage or set(messages) - normal_ids or set(traces) != normal_ids:
        raise ValueError("SDK requests and wire requests do not have complete coverage")
    reads = set(read_starts) & set(successful_tools) & forwarded
    if (
        not reads
        or not turns
        or any(state["sessionId"] != session["session_id"] for state in states)
    ):
        raise ValueError(
            "persistent session lacks actual forwarded reads or continuous identity"
        )
    work_reads = work_item_reads(
        read_starts,
        successful_tools,
        forwarded,
        turns,
        Path(session.get("workspace_path", directory / "workspace")),
        goal_mode=session.get("workspace", {}).get("kind") == "native_goal_database",
    )
    if not any(item["forwarded_source_reads"] for item in work_reads):
        raise ValueError("persistent workload has no forwarded task or source read")
    compactions = [entry for entry in entries if entry["type"] == "compaction"]
    if any(not entry.get("summary", "").strip() for entry in compactions):
        raise ValueError("compaction has no usable history summary")
    continuations = compaction_continuations(compactions, points)
    if any(not item["continued"] for item in continuations):
        raise ValueError("observed compaction has no successful continuation")
    context_usage = audit_context_usage(states, anchors, compactions)
    return points, {
        "lane": session["lane"],
        "api": session["api"],
        "session_id": session["session_id"],
        "completed_turns": len(turns),
        "completed_requests": len(points),
        "forwarded_source_reads": len(reads),
        "work_item_reads": work_reads,
        "compactions": len(compactions),
        "maximum_input_tokens": max(point["input_tokens"] for point in points),
        "maximum_total_budget": max(
            point["input_tokens"] + point["output_budget"] for point in points
        ),
        "controller_gap_seconds": [
            max(0, right["started_unix_s"] - left["completed_unix_s"])
            for left, right in pairwise(turns)
        ],
        "compaction_continuations": continuations,
        "compaction_continuation_covered": bool(continuations)
        and all(item["continued"] for item in continuations),
        "process": session["process"],
        "goal_artifacts": session.get("goal_artifacts"),
        "goal_collection": session.get("goal_collection"),
        "goals": turns if session.get("goal_id") else None,
        "active_seconds": session["active_seconds"],
        "context_usage": context_usage,
        "entries_sha256": hashlib.sha256(
            (directory / session["entries_file"]).read_bytes()
        ).hexdigest(),
    }
