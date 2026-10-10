"""Incremental stream terminals and independent OMP consumption evidence."""

from __future__ import annotations

import hashlib
import json


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
        if self.api == "chat":
            self.identity(event.get("id"))
            choices = event.get("choices")
            if not isinstance(choices, list):
                self.bad("invalid_chat_choices")
                return
            for choice in choices:
                if not isinstance(choice, dict):
                    self.bad("invalid_chat_choice")
                    continue
                reason = choice.get("finish_reason")
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
