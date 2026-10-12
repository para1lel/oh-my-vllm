"""Completion records distinguish a transport close from a failed model call."""

import re


def sdk_completion_valid(client: dict, sdk_api: str) -> bool:
    """RPC messages and native SDK spans have separate recorded origins."""
    if client.get("source") == "omp_sdk_otlp":
        count = client.get("span_count")
        return (
            client.get("api_binding") == sdk_api
            and client.get("api_binding_source") == "isolated_provider_configuration"
            and "api" not in client
            and type(count) is int
            and count == 1
            and isinstance(client.get("export_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", client["export_sha256"]) is not None
            and isinstance(client.get("export_file"), str)
            and re.fullmatch(r"[0-9]{4}\.pb", client["export_file"]) is not None
            and isinstance(client.get("trace_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", client["trace_id"]) is not None
            and isinstance(client.get("span_id"), str)
            and re.fullmatch(r"[0-9a-f]{16}", client["span_id"]) is not None
        )
    count = client.get("message_count")
    digest = client.get("message_sha256")
    return (
        client.get("source") in (None, "omp_rpc_message_end")
        and client.get("api") == sdk_api
        and type(count) is int
        and count == 1
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    )


def completed_transport_close(request: dict, response_id: str, api: str) -> bool:
    """Require protocol and SDK completion alongside the caller's server check."""
    if not isinstance(api, str):
        return False
    sdk_api = {"chat": "openai-completions", "responses": "openai-responses"}.get(api)
    protocol = request.get("protocol_completion", {})
    client = request.get("client_completion", {})
    if not isinstance(protocol, dict) or not isinstance(client, dict) or not sdk_api:
        return False
    kind = request.get("transport_error_kind")
    error = request.get("error")
    if (
        not isinstance(kind, str)
        or kind not in {"ConnectionResetError", "BrokenPipeError"}
        or not isinstance(error, str)
        or not error.startswith(kind + ":")
        or not isinstance(request.get("error_direction"), str)
        or request["error_direction"] not in {"upstream_read", "downstream_write"}
        or request.get("transport_outcome") != "terminal_disconnect"
    ):
        return False
    reason = protocol.get("finish_reason")
    if not isinstance(reason, str):
        return False
    stop_reason = {"stop": "stop", "tool_calls": "toolUse", "length": "length"}.get(
        reason
    )
    if not stop_reason:
        return False
    terminal = protocol.get("terminal_kind")
    if api == "chat":
        valid_terminal = terminal == "chat_done"
    else:
        valid_terminal = (
            terminal == "responses_completed" and reason in {"stop", "tool_calls"}
        ) or (terminal == "responses_incomplete" and reason == "length")
    offset, size = protocol.get("raw_terminal_offset"), request.get("response_bytes")
    return (
        valid_terminal
        and protocol.get("success") is True
        and protocol.get("errors") == []
        and isinstance(protocol.get("terminal_forwarded"), bool)
        and protocol.get("api") == api
        and protocol.get("response_id") == response_id
        and isinstance(offset, int)
        and not isinstance(offset, bool)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < offset <= size
        and client.get("successful") is True
        and client.get("response_id") == response_id
        and sdk_completion_valid(client, sdk_api)
        and client.get("model") == "qwen3.8-27b-fp8"
        and client.get("stop_reason") == stop_reason
    )
