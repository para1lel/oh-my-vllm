"""A finished SDK stream can close before the next upstream terminal frame."""

import io
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from benchmarks.agentic_soak_protocol import attach_client_completions
from benchmarks.agentic_soak_proxy import (
    ModelSlots,
    disk_proxy,
    record_transport_error,
    relay_stream,
)
from benchmarks.agentic_soak_transport import completed_transport_close

IDENTITY = "chatcmpl_tail_1"


def event(reason=None):
    return (
        b"data: "
        + json.dumps(
            {
                "id": IDENTITY,
                "choices": [{"index": 0, "finish_reason": reason}],
            }
        ).encode()
        + b"\n\n"
    )


USAGE = b'data: {"id":"chatcmpl_tail_1","choices":[],"usage":{}}\n\n'
DONE = b"data: [DONE]\n\n"


class Reader:
    def __init__(self, pieces):
        self.pieces = iter(pieces)
        self.calls = 0

    def read(self, _size):
        self.calls += 1
        piece = next(self.pieces, b"")
        if isinstance(piece, Exception):
            raise piece
        return piece

    read1 = read


def capture(pieces, exception=ConnectionResetError, *, timeout=None):
    reader, output, record, writes = Reader(pieces), io.BytesIO(), {}, []

    def write(chunk):
        writes.append(chunk)
        raise exception("SDK consumed finish and closed")

    try:
        relay_stream(reader, output, write, record, "chat", set_read_timeout=timeout)
    except Exception as error:
        record_transport_error(record, error)
    return reader, output.getvalue(), record, writes


@pytest.mark.parametrize("exception", [ConnectionResetError, BrokenPipeError])
def test_finish_then_disconnect_captures_separate_done_without_further_writes(
    exception,
):
    timeouts = []
    reader, raw, record, writes = capture(
        [event("tool_calls"), USAGE, DONE], exception, timeout=timeouts.append
    )
    assert reader.calls == 3
    assert writes == [event("tool_calls")]
    assert raw == event("tool_calls") + USAGE + DONE
    assert record["transport_outcome"] == "terminal_disconnect"
    assert record["error"] == exception.__name__ + ": SDK consumed finish and closed"
    assert record["error_direction"] == "downstream_write"
    assert record["protocol_completion"]["success"] is True
    assert record["protocol_completion"]["terminal_forwarded"] is False
    assert timeouts and all(0 < value <= 2 for value in timeouts)
    assert "capture_error" not in record
    assert not completed_transport_close(
        record, IDENTITY, "chat"
    )  # Missing client proof.


@pytest.mark.parametrize(
    "tail",
    [b"", ConnectionResetError("truncated upstream"), b'data: {"error":"failed"}\n\n'],
)
def test_failed_or_truncated_tail_preserves_first_disconnect_and_secondary_cause(tail):
    reader, _raw, record, writes = capture([event("stop"), tail, DONE])
    assert reader.calls == 2
    assert len(writes) == 1
    assert record["error"] == "ConnectionResetError: SDK consumed finish and closed"
    assert record["error_direction"] == "downstream_write"
    assert record["transport_outcome"] == "failed"
    assert record["protocol_completion"]["success"] is False
    assert record["capture_error"]


def test_before_finish_disconnect_does_not_continue_background_generation():
    reader, raw, record, writes = capture([event(), event("stop"), DONE])
    assert reader.calls == 1
    assert raw == event()
    assert len(writes) == 1
    assert record["transport_outcome"] == "failed"
    assert record["protocol_completion"]["success"] is False


def test_trickling_tail_uses_absolute_deadline_without_resetting_per_read(monkeypatch):
    import benchmarks.agentic_soak_proxy as proxy

    clock = [0.0]
    monkeypatch.setattr(proxy.time, "monotonic", lambda: clock[0])
    timeouts = []

    def timeout(remaining):
        timeouts.append(remaining)
        clock[0] += 0.75

    reader, _raw, record, _writes = capture(
        [event("stop"), USAGE, USAGE, USAGE, DONE], timeout=timeout
    )
    assert reader.calls == 4
    assert timeouts == [2.0, 1.25, 0.5]
    assert record["transport_outcome"] == "failed"
    assert record["capture_error"].startswith("TimeoutError:")
    assert record["capture_error_direction"] == "upstream_read"
    assert record["error"].startswith("ConnectionResetError:")


@pytest.mark.parametrize(
    "normal,send_terminal", [(True, True), (False, False), (True, False)]
)
def test_real_tcp_sdk_close_before_split_done_has_only_normal_completion_grace(
    tmp_path, monkeypatch, normal, send_terminal
):
    import benchmarks.agentic_soak_proxy as proxy

    closed, capturing, exiting = (threading.Event() for _ in range(3))
    finished, errors = threading.Event(), []
    original_timeout = proxy.set_upstream_timeout

    def timeout(response, remaining):
        capturing.set()
        original_timeout(response, remaining)

    monkeypatch.setattr(proxy, "set_upstream_timeout", timeout)

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            try:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(event("stop"))
                self.wfile.flush()
                assert closed.wait(3)
                time.sleep(
                    0.02
                )  # Ensure the loopback RST is visible before the next write.
                self.wfile.write(USAGE)
                self.wfile.flush()
                assert capturing.wait(3)
                assert exiting.wait(3)
                if send_terminal:
                    self.wfile.write(DONE[:-1])
                    self.wfile.flush()
                    time.sleep(0.05)
                    self.wfile.write(DONE[-1:])
                    self.wfile.flush()
                else:
                    # Cancellation shutdown must finish without waiting for this tail.
                    finished.wait(5 if normal else 0.3)
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def completion():
        exiting.set()
        return normal

    started = time.monotonic()
    try:
        with disk_proxy(
            f"http://127.0.0.1:{server.server_port}/v1",
            tmp_path / "wire",
            ModelSlots(),
            request_timeout=20,
            normal_completion=completion,
        ) as (url, records):
            parsed = urlsplit(url)
            client = socket.create_connection((parsed.hostname, parsed.port), timeout=3)
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Length: 2\r\nContent-Type: application/json\r\n\r\n{}"
            )
            body = bytearray()
            while event("stop") not in body:
                chunk = client.recv(8192)
                assert chunk
                body.extend(chunk)
            client.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            client.close()
            closed.set()
            assert capturing.wait(3)
        elapsed = time.monotonic() - started
        assert len(records) == 1
        record = records[0]
        assert record["error_direction"] == "downstream_write"
        assert record["transport_error_kind"] in {
            "ConnectionResetError",
            "BrokenPipeError",
        }
        assert record["transport_outcome"] == (
            "terminal_disconnect" if send_terminal else "failed"
        )
        assert record["protocol_completion"]["success"] is send_terminal
        assert elapsed < 3.5  # No wait for the normal 20-second per-read timeout.
        if not normal:
            assert elapsed < 2
        elif not send_terminal:
            assert elapsed >= 1.9
            assert record["capture_error"]
        if send_terminal:
            raw = (tmp_path / "wire" / record["response_file"]).read_bytes()
            assert raw == event("stop") + USAGE + DONE
            attach_client_completions(
                [
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "responseId": IDENTITY,
                            "api": "openai-completions",
                            "model": "qwen3.8-27b-fp8",
                            "stopReason": "stop",
                            "content": [{"type": "text", "text": "done"}],
                        },
                    }
                ],
                [record],
                "chat",
            )
            assert completed_transport_close(record, IDENTITY, "chat")
    finally:
        closed.set()
        exiting.set()
        finished.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not errors


def test_interrupt_during_normal_grace_still_releases_proxy_listener(tmp_path):
    def interrupted():
        raise InterruptedError("cancel while entering normal completion grace")

    with (
        pytest.raises(InterruptedError, match="cancel while"),
        disk_proxy(
            "http://127.0.0.1:9/v1",
            tmp_path / "wire",
            ModelSlots(),
            normal_completion=interrupted,
        ) as (url, _records),
    ):
        parsed = urlsplit(url)
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((parsed.hostname, parsed.port), timeout=0.5)
