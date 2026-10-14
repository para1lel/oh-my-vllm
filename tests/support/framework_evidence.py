"""Independent wire and server request fixtures for framework audits."""

import hashlib
import json


def request_fixture(tmp_path, api, *, cancelled=False):
    identity = ("chatcmpl_" if api == "chat" else "resp_") + "fixture_1"
    model = "qwen3.8-27b-fp8"
    if api == "chat":
        events = [
            {"id": identity, "model": model, "choices": [{"index": 0, "delta": {}}]}
        ]
        if not cancelled:
            events.append(
                {
                    "id": identity,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 6},
                }
            )
    else:
        response = {"id": identity, "model": model, "status": "in_progress"}
        events = [{"type": "response.created", "response": response}]
        if not cancelled:
            events.append(
                {
                    "type": "response.completed",
                    "response": dict(
                        response,
                        status="completed",
                        output=[{"type": "message"}],
                        usage={"input_tokens": 4, "output_tokens": 6},
                    ),
                }
            )
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    if cancelled:
        error = {"code": "server_error", "message": "request cancelled or timed out"}
        if api == "chat":
            event = {"error": dict(error, type="server_error")}
        else:
            event = dict(error, type="error", param=None, sequence_number=1)
        body += b"event: error\n" if api == "responses" else b""
        body += b"data: " + json.dumps(event).encode() + b"\n\n"
    elif api == "chat":
        body += b"data: [DONE]\n\n"
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "request.json").write_text(
        json.dumps(
            {
                "model": model,
                "stream": True,
                "max_tokens" if api == "chat" else "max_output_tokens": 6,
                "messages" if api == "chat" else "input": [],
            }
        )
    )
    (wire / "response.bin").write_bytes(body)
    record = {
        "request_file": "request.json",
        "path": "/v1/chat/completions" if api == "chat" else "/v1/responses",
        "response_file": "response.bin",
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "response_status": 200,
        "response_ids": [identity],
        "session_id": "same-session",
        "context_proof": {
            "input_tokens": 4,
            "output_budget": 6,
            "total_budget": 10,
            "reasoning_effort": "xhigh",
        },
    }
    (wire / "http.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "states.json").write_text(json.dumps([{"sessionId": "same-session"}]))
    # The original native journal can contain missing IDs or task errors.
    (tmp_path / "events.json").write_text(json.dumps([{"type": "task_failed"}]))
    session = {"session_id": "same-session", "lane": 0, "api": api}
    metric = {
        "api": api,
        "epoch": 1,
        "request_id": 1,
        "response_id": identity,
        "status": "cancelled" if cancelled else "completed",
        "output_tokens": 2 if cancelled else 6,
        "received_unix_s": 100,
        "processing_started_unix_s": 101,
        "generation_completed_unix_s": 102,
        "queue_seconds": 1,
        "processing_seconds": 1,
    }
    return session, {identity: metric}, record
