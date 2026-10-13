import hashlib

from tests.support.soak import metric, task


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
