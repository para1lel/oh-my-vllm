"""Pinned OMP request equality and actual SDK task/automatic-compaction exports.

The CPU fake API supplies deterministic content and usage. This checks client
instrumentation, not target inference or actual tokenizer accuracy.
"""

import hashlib
import json
import shutil
import socket
import subprocess
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from benchmarks.agentic_context import isolated_environment
from benchmarks.agentic_sdk_trace import SdkTrace, read_sdk_spans
from benchmarks.agentic_session_evidence import trace_messages
from benchmarks.agentic_sessions import RpcClient


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("completed_history", [False, True])
def test_pinned_omp_observer_preserves_requests_and_real_automatic_compaction(
    tmp_path, monkeypatch, api, completed_history
):
    omp = shutil.which("omp")
    if omp is None:
        pytest.skip("Pinned OMP 18.4.3 executable is not installed")
    version = subprocess.run(
        [omp, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        env=isolated_environment(tmp_path / "version-state"),
    )
    assert version.returncode == 0 and "18.4.3" in version.stdout
    slot = tmp_path / "client"
    records = []
    record_lock = threading.Lock()
    first_parallel = threading.Event()
    prefix = "\n".join(
        f"{i:05d}: {hashlib.sha256(str(i).encode()).hexdigest()} "
        f"states source invariant {i * i}."
        for i in range(2400)
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with record_lock:
                parallel = completed_history and 2 <= len(records) < 4
            if parallel:
                # The two real compaction calls run in Promise.all. Exercise
                # opposite admission orders with the deterministic fake API.
                prefix_call = (
                    "MUST summarize prefix for retained suffix:" in json.dumps(body)
                )
                if prefix_call != enabled:
                    assert first_parallel.wait(timeout=10)
            with record_lock:
                records.append({"path": self.path, "request": body})
                n = len(records)
                (slot / "request-bodies.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                if completed_history and n == 3:
                    first_parallel.set()
            ident = f"resp_fixture_{n}"
            text = (
                prefix
                if n == (2 if completed_history else 1)
                else (
                    "## Goal\nContinue the source audit.\n"
                    "## Progress\nEvidence retained.\n"
                    "## Next\nRead the remaining source."
                )
            )
            prompt = 196700 if n == (2 if completed_history else 1) else 40
            out = 25000 if n == (2 if completed_history else 1) else 25
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if "chat/completions" in self.path:
                events = [
                    {
                        "id": ident,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": text},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": ident,
                        "object": "chat.completion.chunk",
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt,
                            "completion_tokens": out,
                            "total_tokens": prompt + out,
                        },
                    },
                ]
                for x in events:
                    self.wfile.write(("data: " + json.dumps(x) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                item = {
                    "id": "msg_fixture_" + str(n),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
                response = {
                    "id": ident,
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": body["model"],
                    "output": [item],
                    "usage": {
                        "input_tokens": prompt,
                        "output_tokens": out,
                        "total_tokens": prompt + out,
                    },
                }
                events = [
                    {
                        "type": "response.created",
                        "response": dict(response, status="in_progress", output=[]),
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": dict(item, status="in_progress", content=[]),
                    },
                    {
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": item["id"],
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": item["id"],
                        "delta": text,
                    },
                    {
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": item["id"],
                        "text": text,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
                for seq, x in enumerate(events):
                    x["sequence_number"] = seq
                    self.wfile.write(
                        (
                            "event: " + x["type"] + "\ndata: " + json.dumps(x) + "\n\n"
                        ).encode()
                    )
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_popen = subprocess.Popen
    original_environment = SdkTrace.environment

    def popen(command, **kwargs):
        if command[0] == omp:
            command = [*command, "--session", str(slot / "fixed-session.jsonl")]
        return original_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    requests = []
    try:
        for enabled in (False, True):
            slot.mkdir(exist_ok=False)
            (slot / "fixed-session.jsonl").write_text(
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": "01a131a8-7009-74b8-b8a5-81ac62c8611f",
                        "timestamp": "2026-10-12T19:48:04.745Z",
                        "cwd": str(slot / "workspace"),
                    }
                )
                + "\n"
            )
            records.clear()
            first_parallel = threading.Event()
            monkeypatch.setattr(
                SdkTrace,
                "environment",
                original_environment if enabled else lambda self: {},
            )
            client = RpcClient(
                omp,
                api,
                0,
                slot,
                f"http://127.0.0.1:{server.server_port}/v1",
                threading.Event(),
                120,
            )
            try:
                client.initialize()
                client.prompt(
                    "Review the current source invariants and retain all evidence.", 0
                )
                client.prompt(
                    "Continue this audit in the same session "
                    "and summarize the next source invariant.",
                    1,
                )
                if completed_history:
                    client.prompt("Continue after the automatic compaction.", 2)
                if enabled:
                    result = client.collect()
                else:
                    client.check_state()
                    client.checkpoint_entries()
                    client.persist()
            finally:
                client.close()
            assert client.process.returncode == 0
            assert client.process_record["reaped"]
            assert any(
                event["type"] == "auto_compaction_end" for event in client.events
            )
            requests.append(list(records))
            spans = read_sdk_spans(slot / "sdk-trace")
            if enabled:
                messages = trace_messages(slot, result)
                expected_count = 6 if completed_history else 4
                assert len(messages) == len(records) == expected_count
                assert {message["responseId"] for message in messages} == {
                    f"resp_fixture_{i}" for i in range(1, expected_count + 1)
                }
                purposes = {message["purpose"] for message in messages}
                expected = {
                    "task",
                    "compaction_turn_prefix",
                    "compaction_short_summary",
                }
                if completed_history:
                    expected.add("compaction_summary")
                assert purposes == expected
                assert all(
                    message["session_id"] == result["session_id"]
                    for message in messages
                )
                assert len(spans) == expected_count
            shutil.move(slot, tmp_path / ("on" if enabled else "off"))
        # The workspace, fixed session, source and fake response IDs are equal.
        # Compare the complete JSON bodies, including xhigh and native budgets.
        if completed_history:
            assert len(requests[0]) == len(requests[1]) == 6
            assert requests[0][:2] == requests[1][:2]
            assert requests[0][4:] == requests[1][4:]
            assert requests[0][2] != requests[1][2]
            assert Counter(
                json.dumps(record, sort_keys=True) for record in requests[0][2:4]
            ) == Counter(
                json.dumps(record, sort_keys=True) for record in requests[1][2:4]
            )
        else:
            assert requests[0] == requests[1]
        for record in requests[1]:
            body = record["request"]
            assert "thinking_token_budget" not in body
            effort = body.get(
                "reasoning_effort", body.get("reasoning", {}).get("effort")
            )
            assert effort == "xhigh"
        with socket.socket() as connection:
            endpoint = urlsplit(client.sdk_trace.endpoint)
            assert connection.connect_ex((endpoint.hostname, endpoint.port)) != 0
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        assert not thread.is_alive()
