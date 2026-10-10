"""Independent server binding for certified output-budget completion."""

import re


def validate_length_metric(request: dict, metric: dict, api: str) -> bool:
    """Bind a certified output limit to the independent completed server count."""
    protocol = request.get("protocol_completion", {})
    client = request.get("client_completion", {})
    proof = request.get("controlled_length_completion")
    length = (
        isinstance(protocol, dict) and protocol.get("finish_reason") == "length"
    ) or (isinstance(client, dict) and client.get("stop_reason") == "length")
    if not length and proof is None:
        return False

    def require(condition):
        if not condition:
            raise ValueError(
                "normal model request failed: controlled length proof differs"
            )

    require(
        isinstance(protocol, dict)
        and isinstance(client, dict)
        and isinstance(proof, dict)
        and api in ("chat", "responses")
    )
    identity = metric["response_id"]
    expected_api = "openai-completions" if api == "chat" else "openai-responses"
    require(
        request.get("response_ids") == [identity]
        and protocol.get("response_id") == client.get("response_id") == identity
        and proof.get("response_id") == identity
        and protocol.get("api") == proof.get("api") == metric["api"] == api
        and client.get("api") == expected_api
        and protocol.get("model")
        == client.get("model")
        == proof.get("model")
        == "qwen3.8-27b-fp8"
    )
    require(
        protocol.get("success") is True
        and protocol.get("errors") == []
        and protocol.get("finish_reason") == client.get("stop_reason") == "length"
        and protocol.get("terminal_kind")
        == ("chat_done" if api == "chat" else "responses_incomplete")
        and client.get("successful") is True
        and type(client.get("message_count")) is int
        and client["message_count"] == 1
        and metric["status"] == "completed"
        and proof.get("nonempty_text") is True
        and proof.get("no_tool_call") is True
        and proof.get("completed_server_metric_required") is True
        and proof.get("stop_reason") == "length"
        and proof.get("terminal_kind") == protocol.get("terminal_kind")
        and type(protocol.get("pending_tool_call_count")) is int
        and protocol["pending_tool_call_count"] == 0
        and type(proof.get("pending_tool_call_count")) is int
        and proof["pending_tool_call_count"] == 0
    )
    counts = [
        proof.get(key)
        for key in ("declared_output_budget", "sdk_output_tokens", "sse_output_tokens")
    ] + [
        protocol.get("output_tokens"),
        client.get("output_tokens"),
        metric["output_tokens"],
    ]
    require(all(type(value) is int and value > 0 for value in counts))
    require(len(set(counts)) == 1)
    if api == "responses":
        require(
            type(protocol.get("declared_output_budget")) is int
            and protocol["declared_output_budget"] == counts[0]
        )
    for key, original in (
        ("client_message_sha256", client.get("message_sha256")),
        ("response_sha256", request.get("response_sha256")),
    ):
        require(
            isinstance(original, str)
            and re.fullmatch(r"[0-9a-f]{64}", original) is not None
            and proof.get(key) == original
        )
    offset, size = protocol.get("raw_terminal_offset"), request.get("response_bytes")
    require(type(offset) is int and type(size) is int and 0 < offset <= size)
    require(
        type(proof.get("raw_terminal_offset")) is int
        and proof["raw_terminal_offset"] == offset
        and type(proof.get("response_bytes")) is int
        and proof["response_bytes"] == size
    )
    return True
