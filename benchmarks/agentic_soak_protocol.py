"""Incremental stream terminals and independent OMP consumption evidence."""

from __future__ import annotations

import hashlib
import json
import re

from benchmarks.agentic_soak_transport import completed_transport_close


class StreamCompletion:
    """A captured terminal is separate from successful delivery to the client."""

    def __init__(self, api: str, maximum_event_bytes: int):
        self.api = api
        self.maximum_event_bytes = maximum_event_bytes
        self.pending = bytearray()
        self.ids = set()
        self.errors = []
        self.offset = 0
        self.event = None
        self.data_lines = []
        self.event_bytes = 0
        self.finish_reason = None
        self.terminal_kind = None
        self.terminal_id = None
        self.terminal_offset = None
        self.terminal_forwarded = False
        self.output_tokens = None
        self.model = None
        self.declared_output_budget = None
        self.tool_indices = set()

    def bad(self, reason):
        if reason not in self.errors and len(self.errors) < 8:
            self.errors.append(reason)

    def identity(self, value):
        prefix = "chatcmpl_" if self.api == "chat" else "resp_"
        if not isinstance(value, str) or not value.startswith(prefix):
            self.bad("invalid_response_identity")
            return
        self.ids.add(value)
        if len(self.ids) != 1:
            self.bad("response_identity_mismatch")

    def terminal(self, kind, identity):
        if self.terminal_kind is not None:
            self.bad("duplicate_terminal")
            return
        self.terminal_kind = kind
        self.terminal_id = identity
        self.terminal_offset = self.offset

    def model_identity(self, value):
        if value is not None:
            if not isinstance(value, str) or (
                self.model is not None and value != self.model
            ):
                self.bad("response_model_mismatch")
            else:
                self.model = value

    def tool_index(self, value):
        if type(value) is not int or value < 0:
            self.bad("invalid_tool_index")
        else:
            self.tool_indices.add(value)

    def parse(self, payload):
        if payload == b"[DONE]":
            if self.api == "chat":
                if self.finish_reason not in ("stop", "tool_calls", "length"):
                    self.bad("chat_done_without_finish_reason")
                self.terminal("chat_done", next(iter(self.ids), None))
            elif self.terminal_kind is None:
                self.bad("responses_done_without_response_terminal")
            return
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeError):
            self.bad("invalid_sse_json")
            return
        if not isinstance(event, dict):
            self.bad("nonobject_sse_json")
            return
        response = event.get("response")
        if event.get("error") or event.get("type") in ("error", "response.failed"):
            self.bad("server_sse_error")
        if isinstance(response, dict):
            if response.get("error") or response.get("status") == "failed":
                self.bad("server_response_error")
            self.identity(response.get("id"))
            self.model_identity(response.get("model"))
        if self.api == "chat":
            self.identity(event.get("id"))
            self.model_identity(event.get("model"))
            if isinstance(event.get("usage"), dict):
                tokens = event["usage"].get("completion_tokens")
                if self.output_tokens is not None and tokens != self.output_tokens:
                    self.bad("output_usage_mismatch")
                self.output_tokens = tokens
            choices = event.get("choices")
            if not isinstance(choices, list):
                self.bad("invalid_chat_choices")
                return
            for choice in choices:
                if not isinstance(choice, dict):
                    self.bad("invalid_chat_choice")
                    continue
                reason = choice.get("finish_reason")
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    self.bad("invalid_chat_delta")
                elif delta.get("tool_calls") is not None:
                    if not isinstance(delta["tool_calls"], list):
                        self.bad("invalid_chat_tool_calls")
                    else:
                        for call in delta["tool_calls"]:
                            if not isinstance(call, dict):
                                self.bad("invalid_chat_tool_call")
                            else:
                                self.tool_index(call.get("index"))
                if reason is not None:
                    if choice.get("index") != 0 or reason not in (
                        "stop",
                        "tool_calls",
                        "length",
                    ):
                        self.bad("invalid_chat_finish")
                    if self.finish_reason is not None:
                        self.bad("duplicate_chat_finish")
                    self.finish_reason = reason
            return
        kind = event.get("type")
        if isinstance(kind, str) and kind.startswith(
            "response.function_call_arguments."
        ):
            self.tool_index(event.get("output_index"))
        item = event.get("item", {})
        if isinstance(item, dict) and item.get("type") == "function_call":
            self.tool_index(event.get("output_index"))
        if kind not in ("response.completed", "response.incomplete"):
            return
        if self.event is not None and self.event != kind:
            self.bad("sse_event_type_mismatch")
        if not isinstance(response, dict):
            self.bad("responses_terminal_without_response")
            return
        output = response.get("output")
        if not isinstance(output, list):
            self.bad("invalid_responses_output")
            output = []
        for index, item in enumerate(output):
            if isinstance(item, dict) and item.get("type") == "function_call":
                self.tool_index(index)
        if kind == "response.incomplete":
            details = response.get("incomplete_details")
            if (
                response.get("status") != "incomplete"
                or not isinstance(details, dict)
                or details.get("reason") != "max_output_tokens"
            ):
                self.bad("invalid_responses_incomplete")
            self.finish_reason = "length"
            terminal_kind = "responses_incomplete"
        else:
            if response.get("status") != "completed":
                self.bad("invalid_responses_completed")
            self.finish_reason = (
                "tool_calls"
                if any(
                    isinstance(item, dict) and item.get("type") == "function_call"
                    for item in output
                )
                else "stop"
            )
            terminal_kind = "responses_completed"
        self.terminal(terminal_kind, response.get("id"))
        self.model_identity(response.get("model"))
        self.declared_output_budget = response.get("max_output_tokens")
        usage = response.get("usage")
        self.output_tokens = (
            usage.get("output_tokens") if isinstance(usage, dict) else None
        )

    def feed(self, chunk: bytes):
        self.pending.extend(chunk)
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending[:] = rest
            self.offset += len(line) + 1
            self.event_bytes += len(line) + 1
            if self.event_bytes > self.maximum_event_bytes:
                self.bad("oversized_sse_event")
                raise ValueError("server SSE event exceeds the evidence proxy bound")
            line = line.rstrip(b"\r")
            if line.startswith(b"event:"):
                self.event = line[6:].strip().decode("utf-8", errors="replace")
            elif line.startswith(b"data:"):
                payload = line[5:]
                self.data_lines.append(
                    payload[1:] if payload.startswith(b" ") else payload
                )
            elif not line:
                if self.data_lines:
                    self.parse(b"\n".join(self.data_lines))
                self.event = None
                self.data_lines = []
                self.event_bytes = 0
        if self.event_bytes + len(self.pending) > self.maximum_event_bytes:
            self.bad("oversized_sse_event")
            raise ValueError("server SSE event exceeds the evidence proxy bound")

    def snapshot(self):
        success = (
            self.terminal_kind is not None
            and len(self.ids) == 1
            and self.terminal_id in self.ids
            and not self.errors
        )
        return {
            "api": self.api,
            "response_id": self.terminal_id,
            "success": success,
            "terminal_kind": self.terminal_kind,
            "finish_reason": self.finish_reason,
            "raw_terminal_offset": self.terminal_offset,
            "terminal_forwarded": self.terminal_forwarded,
            "errors": list(self.errors),
            "output_tokens": self.output_tokens,
            "model": self.model,
            "declared_output_budget": self.declared_output_budget,
            "pending_tool_call_count": len(self.tool_indices),
        }


def attach_client_completions(events: list[dict], requests: list[dict], api: str):
    """Record same-ID message_end consumption; the server proof is independent."""
    expected_api = "openai-completions" if api == "chat" else "openai-responses"
    by_id = {}
    for event in events:
        message = event.get("message", {})
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            by_id.setdefault(message.get("responseId"), []).append(message)
    for request in requests:
        ids = request.get("response_ids", [])
        identity = ids[0] if len(ids) == 1 else None
        messages = by_id.get(identity, [])
        message = messages[0] if len(messages) == 1 else {}
        successful = (
            len(messages) == 1
            and message.get("api") == expected_api
            and message.get("model") == "qwen3.8-27b-fp8"
            and message.get("stopReason") in ("stop", "toolUse", "length")
            and not message.get("errorMessage")
            and bool(message.get("content"))
        )
        request["client_completion"] = {
            "response_id": identity,
            "api": message.get("api"),
            "model": message.get("model"),
            "stop_reason": message.get("stopReason"),
            "output_tokens": (
                message["usage"].get("output")
                if isinstance(message.get("usage"), dict)
                else None
            ),
            "message_count": len(messages),
            "successful": successful,
            "message_sha256": (
                hashlib.sha256(
                    json.dumps(message, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                if message
                else None
            ),
        }


def controlled_length_completion(final: dict, request: dict, *, api: str) -> dict:
    """Certify a declared output limit; the offline collector checks the server.

    This opt-in oracle does not change SDK events or the captured SSE terminal.
    The generic agent acceptance still requires natural termination by default.
    """
    if api not in ("chat", "responses"):
        raise ValueError("invalid controlled length API")
    expected_api = {"chat": "openai-completions", "responses": "openai-responses"}[api]
    protocol, client, body = (
        request.get("protocol_completion", {}),
        request.get("client_completion", {}),
        request.get("request", {}),
    )
    if not all(isinstance(value, dict) for value in (final, protocol, client, body)):
        raise ValueError("invalid controlled length evidence")
    budget = (
        body.get("max_completion_tokens", body.get("max_tokens"))
        if api == "chat"
        else body.get("max_output_tokens")
    )
    usage = final.get("usage", {})
    content = final.get("content", [])
    identity = final.get("responseId")
    offset, size = protocol.get("raw_terminal_offset"), request.get("response_bytes")
    digest = hashlib.sha256(
        json.dumps(final, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        type(budget) is not int
        or budget <= 0
        or (
            api == "chat"
            and "max_tokens" in body
            and "max_completion_tokens" in body
            and (
                type(body["max_tokens"]) is not int
                or type(body["max_completion_tokens"]) is not int
                or body["max_tokens"] != body["max_completion_tokens"]
            )
        )
        or not isinstance(content, list)
        or not content
        or any(
            not isinstance(item, dict) or item.get("type") == "toolCall"
            for item in content
        )
        or not any(
            item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
            for item in content
        )
        or final.get("api") != expected_api
        or final.get("model") != "qwen3.8-27b-fp8"
        or body.get("model") != final["model"]
        or final.get("stopReason") != "length"
        or final.get("errorMessage")
        or not isinstance(usage, dict)
        or type(usage.get("output")) is not int
        or usage["output"] != budget
        or request.get("response_status") != 200
        or body.get("stream") is not True
        or not isinstance(identity, str)
        or not identity.startswith("chatcmpl_" if api == "chat" else "resp_")
        or request.get("response_ids") != [identity]
        or client.get("message_count") != 1
        or type(client.get("message_count")) is not int
        or client.get("successful") is not True
        or client.get("response_id") != identity
        or client.get("api") != expected_api
        or client.get("model") != final["model"]
        or client.get("stop_reason") != "length"
        or client.get("message_sha256") != digest
        or type(client.get("output_tokens")) is not int
        or client["output_tokens"] != budget
        or protocol.get("success") is not True
        or protocol.get("errors") != []
        or not isinstance(protocol.get("terminal_forwarded"), bool)
        or protocol.get("api") != api
        or protocol.get("response_id") != identity
        or protocol.get("finish_reason") != "length"
        or protocol.get("terminal_kind")
        != ("chat_done" if api == "chat" else "responses_incomplete")
        or protocol.get("model") != final["model"]
        or type(protocol.get("pending_tool_call_count")) is not int
        or protocol["pending_tool_call_count"] != 0
        or type(protocol.get("output_tokens")) is not int
        or protocol["output_tokens"] != budget
        or (
            api == "responses"
            and (
                type(protocol.get("declared_output_budget")) is not int
                or protocol["declared_output_budget"] != budget
            )
        )
        or type(offset) is not int
        or type(size) is not int
        or not 0 < offset <= size
        or not isinstance(request.get("response_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", request["response_sha256"]) is None
        or (
            request.get("error")
            and not completed_transport_close(request, identity, api)
        )
        or (
            not request.get("error") and request.get("transport_outcome") != "completed"
        )
    ):
        raise ValueError(
            "controlled length completion lacks matching budget/protocol/client proof"
        )
    proof = {
        "response_id": identity,
        "api": api,
        "model": final["model"],
        "stop_reason": "length",
        "terminal_kind": protocol["terminal_kind"],
        "declared_output_budget": budget,
        "sdk_output_tokens": usage["output"],
        "sse_output_tokens": protocol["output_tokens"],
        "client_message_sha256": digest,
        "response_sha256": request["response_sha256"],
        "raw_terminal_offset": offset,
        "response_bytes": size,
        "nonempty_text": True,
        "no_tool_call": True,
        "pending_tool_call_count": 0,
        "completed_server_metric_required": True,
    }
    request["controlled_length_completion"] = proof
    return proof


def certify_length_completions(events: list[dict], requests: list[dict], api: str):
    """Certify every length-ended model call, including a non-final turn."""
    by_id = {}
    for event in events:
        message = event.get("message", {})
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            by_id.setdefault(message.get("responseId"), []).append(message)
    for request in requests:
        if (
            request.get("client_completion", {}).get("stop_reason") == "length"
            or request.get("protocol_completion", {}).get("finish_reason") == "length"
        ):
            messages = by_id.get(
                request.get("protocol_completion", {}).get("response_id"), []
            )
            if len(messages) != 1:
                raise ValueError(
                    "controlled length completion has ambiguous SDK messages"
                )
            controlled_length_completion(messages[0], request, api=api)


def completion_counts(tasks: list[dict]) -> dict:
    """Report length terminals after completed request metrics have been bound."""
    counts = {
        api: sum(
            request.get("protocol_completion", {}).get("finish_reason") == "length"
            for task in tasks
            if task["api"] == api
            for request in task["requests"]
        )
        for api in ("chat", "responses")
    }
    return {
        "normal_length_final_agent_tasks": sum(
            task.get("tool_verification", {}).get("final_stop_reason") == "length"
            for task in tasks
        ),
        "normal_length_model_requests": sum(counts.values()),
        "normal_length_model_requests_by_api": counts,
    }
