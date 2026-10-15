"""Real sealed lanes translate a staged book through both API transports.

The lane runs the fixed OMP binary inside a real rootless container with
``--network none`` and only the disposable workspace plus the bridge socket
directory, so every model request travels guest loopback HTTP -> AF_UNIX
bridge -> host evidence proxy. The fake upstream answers a deterministic
tool ladder: read a staged volume, attempt a forbidden corpus write, write one
translation output, then finish. The test proves tool use, the read-only
guard, staged-input integrity, preserved SDK exports, and full cleanup.
"""

import gzip
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.agentic_corpus import STAGED_DIRECTORY, Corpus
from benchmarks.agentic_lane_transport import lane_transport_evidence
from benchmarks.agentic_sandbox import docker
from benchmarks.agentic_session_evidence import text_content, trace_messages
from benchmarks.agentic_sessions import RpcClient
from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy
from benchmarks.agentic_translation import require_intact
from tests.support.corpus import synthetic_book

pytestmark = pytest.mark.omp

STEPS = (
    ("read", {"path": "corpus/pg731.txt"}),
    ("write", {"path": "corpus/pg731.txt", "content": "stolen"}),
    ("edit", {"i": "remove a corpus line", "input": "[corpus/pg731.txt#A1B2]\nREM"}),
    (
        "edit",
        {"i": "move the specification", "input": "[TASK.md#A1B2]\nMV notes/task.md"},
    ),
    ("write", {"path": "translation/pg731.md", "content": "第一章 译文\n"}),
    (None, "Translation started."),
)
TRANSLATION = "第一章 译文\n"


def sse(api: str, step: int, model: str, index: int) -> bytes:
    """Build one compliant SSE turn for the given api and ladder step."""
    tool, payload = STEPS[step]
    identity = f"{'chatcmpl' if api == 'chat' else 'resp'}_translation_{index}"
    if api == "chat":
        delta = {"role": "assistant"}
        if tool:
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(payload)},
                }
            ]
        else:
            delta["content"] = payload
        events = [
            {"id": identity, "model": model, "choices": [{"index": 0, "delta": delta}]},
            {
                "id": identity,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            },
        ]
        return (
            b"".join(b"data: " + json.dumps(item).encode() + b"\n\n" for item in events)
            + b"data: [DONE]\n\n"
        )
    if tool:
        item = {
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": tool,
            "arguments": json.dumps(payload),
            "status": "completed",
        }
    else:
        item = {
            "type": "message",
            "id": f"msg_{index}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": payload, "annotations": []}],
        }
    response = {
        "id": identity,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [item],
        "usage": {"input_tokens": 40, "output_tokens": 10},
    }
    events = [
        {
            "type": "response.created",
            "response": dict(response, status="in_progress", output=[]),
        },
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return b"".join(
        b"event: "
        + item["type"].encode()
        + b"\ndata: "
        + json.dumps(item).encode()
        + b"\n\n"
        for item in events
    )


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_sealed_lane_translates_with_tool_use_and_preserved_sdk_export(
    tmp_path, agentic_sandbox_image, api
):
    book, identity, _manifest = synthetic_book(tmp_path)
    corpus = Corpus(book, identity)
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            step = min(len(bodies) - 1, len(STEPS) - 1)
            frame = sse(api, step, body["model"], len(bodies))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    directory = tmp_path / "lane"
    directory.mkdir()
    client = None
    try:
        with disk_proxy(
            f"http://127.0.0.1:{server.server_port}/v1",
            directory / "wire",
            ModelSlots(),
            request_timeout=180,
        ) as (proxy, records):
            client = RpcClient(
                "omp",
                api,
                0,
                directory,
                proxy,
                threading.Event(),
                120,
                workload="translation",
                sandbox_image=agentic_sandbox_image,
                corpus=corpus,
            )
            client.initialize()
            sealed = client.sandbox.inspect()
            assert sealed["network_mode"] == "none"
            assert sealed["devices"] == []
            assert len(sealed["mounts"]) == 2
            client.prompt("Translate the staged corpus into Simplified Chinese.", 0)
            result = client.collect()
        events = client.events
        reads = [
            event
            for event in events
            if event.get("type") == "tool_execution_end"
            and event.get("toolName") == "read"
            and "corpus/pg731.txt" in json.dumps(event)
        ]
        assert reads and not reads[0].get("isError"), events
        writes = [
            event
            for event in events
            if event.get("type") == "tool_execution_end"
            and event.get("toolName") == "write"
        ]
        blocked = [event for event in writes if event.get("isError")]
        assert blocked and "read-only" in json.dumps(blocked)
        assert [event for event in writes if not event.get("isError")]
        edits = [
            event
            for event in events
            if event.get("type") == "tool_execution_end"
            and event.get("toolName") == "edit"
        ]
        # Pinned OMP edit payloads are hashline operations, so the guard rejects
        # the tool itself instead of trusting a single path argument.
        assert edits and all(event.get("isError") for event in edits)
        assert all("write tool" in json.dumps(event) for event in edits)
        assert (client.workspace / "translation/pg731.md").read_text() == TRANSLATION
        # The in-memory audit keeps identity and usage only, so the native session
        # journal snapshot carries the assistant text for every turn.
        assistant = [
            entry["message"]
            for entry in map(
                json.loads, (directory / "entries.jsonl").read_text().splitlines()
            )
            if entry.get("type") == "message"
            and entry.get("message", {}).get("role") == "assistant"
        ]
        assert len(assistant) == len(records) == len(bodies)
        assert "Translation started." in text_content(assistant[-1])

        # Staged read-only inputs are unchanged and the post-run gate accepts them.
        assert require_intact(client, corpus)["staged_inputs_verified"] is True
        for volume in corpus.record()["volumes"]:
            staged = client.workspace / STAGED_DIRECTORY / volume["name"]
            assert hashlib.sha256(staged.read_bytes()).hexdigest() == volume["sha256"]

        assert len(bodies) == len(records) >= len(STEPS)
        assert {body["model"] for body in bodies} == {"qwen3.8-27b-fp8"}
        marker = "messages" if api == "chat" else "input"
        assert all(marker in body for body in bodies)
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
    assert not client.workspace.exists()
    assert not client.sandbox.root.exists()
    # The original RPC capture is complete once the fixed process exits.
    with gzip.open(directory / "rpc.jsonl.gz", "rt") as stream:
        captured = [
            record["message"]
            for record in map(json.loads, stream)
            if record.get("type") == "message_end"
            and record.get("message", {}).get("role") == "assistant"
        ]
    assert len(captured) == len(records)
    assert "Translation started." in text_content(captured[-1])
    assert not docker(
        "container",
        "ls",
        "--all",
        "--quiet",
        "--filter",
        f"name=^/{client.sandbox.name}$",
    )
    assert json.loads((directory / "sandbox-cleanup.json").read_text())["closed"]
    # Both bridges released and every hop reached the recorded host proxy.
    transport = lane_transport_evidence(directory)
    assert transport["bridge-model"]["reached_records"] == len(captured)
    assert transport["guest-model"]["reached_records"] == len(captured)
    assert transport["bridge-telemetry"]["reached_records"] >= 1
    assert transport["guest-telemetry"]["reached_records"] >= 1
    assert transport["bridge_errors"] == [] and transport["guest_errors"] == []
    assert json.loads((directory / "transport/bridge.json").read_text())["closed"]
    # Native SDK exports survived the sealed socket path and bind to the api.
    assert result["sdk_trace"]["capture_error"] is None
    messages = trace_messages(directory, result)
    assert len(messages) == len(captured)
    assert {message["responseId"] for message in messages} == {
        f"{'chatcmpl' if api == 'chat' else 'resp'}_translation_{index}"
        for index in range(1, len(captured) + 1)
    }
    assert {message["purpose"] for message in messages} == {"task"}
    assert {message["provider"] for message in messages} == {f"oh-my-vllm-{api}"}
