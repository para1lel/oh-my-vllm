"""Mounted sockets bound every hop and keep original transport failures visible."""

import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks.agentic_lane_forwarder import (
    MAX_BODY_BYTES,
    HopJournal,
    close_servers,
    journal_records,
    loopback_server,
    serve,
)
from benchmarks.agentic_lane_transport import (
    LaneBridge,
    deadline_attempt_bound,
    lane_transport_evidence,
    upstream_origin,
)

ROOT = Path(__file__).resolve().parents[2]
STREAM_FIRST = b'data: {"id": "chatcmpl_1"}\n\n'
STREAM_LAST = b'data: {"finish_reason": "stop"}\n\ndata: [DONE]\n\n'
EXPORT = b"\x0a\x04\x08\x01\x12\x00\x12\x06protob\x00"


def upstream_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.gate, server.stalled, server.requests = (
        threading.Event(),
        threading.Event(),
        [],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.stalled.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    assert not thread.is_alive()


class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def start(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        return body

    def chunk(self, payload):
        self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        self.start()
        self.chunk(STREAM_FIRST)
        if not self.server.gate.wait(15):
            raise AssertionError("streamed first event never reached the client")
        self.chunk(STREAM_LAST)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class StalledHandler(ModelHandler):
    """Send one event, then hold the upstream response open."""

    def do_POST(self):
        self.start()
        self.chunk(STREAM_FIRST)
        self.server.gate.set()
        self.server.stalled.wait(30)


class ExportHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.server.gate.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()


def start_upstreams(handler=ModelHandler, export_handler=ExportHandler):
    model, model_thread = upstream_server(handler)
    export, export_thread = upstream_server(export_handler)
    return (model, model_thread), (export, export_thread)


def stop_upstreams(*pairs):
    for server, thread in pairs:
        stop_server(server, thread)


@pytest.fixture
def upstreams():
    pairs = start_upstreams()
    try:
        yield pairs[0][0], pairs[1][0]
    finally:
        stop_upstreams(*pairs)


def raw_client(path: Path, timeout: float = 10) -> socket.socket:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    connection.connect(str(path))
    return connection


def read_response(connection: socket.socket) -> tuple[int, bytes]:
    """Read one complete HTTP response from a raw socket."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    length = None
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    while True:
        if length is not None and len(body) >= length:
            break
        chunk = connection.recv(4096)
        if not chunk:
            break
        body += chunk
    return status, body


def wait_for_handler(server, timeout: float = 5.0):
    """Wait until the server has one live handler thread to release."""
    deadline = time.monotonic() + timeout
    while not server.handler_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.handler_count(), "the request handler never started"


def release_servers(servers, threads):
    assert close_servers(servers, threads, bound=5.0), "handlers were not released"


def entries(directory: Path, name: str) -> list[dict]:
    return journal_records(directory / f"{name}.jsonl")


def abort_bound(**overrides) -> dict:
    """One canonical attempt bound, as the soak runner derives it.

    The window is the recorded abort action through the recorded client
    disconnect: attempt completions inside it belong to the deadline release.
    """
    bound = {
        "reason": "collection deadline abort: host_collection_deadline_abort",
        "response_id": "chatcmpl_1",
        "direction": "downstream_write",
        "kinds": ["BrokenPipeError", "ConnectionResetError"],
        "deadline_monotonic_s": 200.0,
        "abort_monotonic_s": 200.0025,
        "first_monotonic_s": 200.0025,
        "last_monotonic_s": 200.8,
    }
    bound.update(overrides)
    return bound


def failed_entry(
    digest: str,
    *,
    kind: str = "BrokenPipeError",
    direction: str = "downstream_write",
    received: float = 199.5,
    completed: float = 200.5,
) -> dict:
    """One failed hop attempt in the shape the forwarder records."""
    return {
        "path": "/v1/chat/completions",
        "body_sha256": digest,
        "upstream_dispatched": True,
        "error": f"{kind}: peer closed",
        "error_kind": kind,
        "error_direction": direction,
        "received_monotonic_s": received,
        "completed_monotonic_s": completed,
    }


def test_upstream_must_be_a_local_plain_http_origin():
    assert upstream_origin("http://127.0.0.1:8123") == ("127.0.0.1", 8123, "")
    assert upstream_origin("http://127.0.0.1:8123/v1") == ("127.0.0.1", 8123, "/v1")
    for rejected in (
        "http://collector.invalid:8123",
        "https://127.0.0.1:8123",
        "http://127.0.0.1",
        "http://127.0.0.1:8123/v1?token=1",
    ):
        with pytest.raises(ValueError):
            upstream_origin(rejected)


def test_collector_bases_from_the_soak_wiring_are_accepted(tmp_path):
    """The evidence proxy and SDK trace endpoints carry a base path."""
    bridge = LaneBridge(
        tmp_path / "transport",
        "http://127.0.0.1:8123/v1",
        telemetry_upstream="http://127.0.0.1:8124/abc/v1/traces",
        read_timeout=30,
    )
    try:
        assert bridge.guest_model_base() == "http://127.0.0.1:18080/v1"
        assert bridge.guest_endpoint("http://127.0.0.1:8124/abc/v1/traces") == (
            "http://127.0.0.1:18081/abc/v1/traces"
        )
        entry = bridge.entry_point(["/usr/local/bin/omp"])
        assert str(bridge.telemetry_socket) in entry
        assert sorted(path.name for path in bridge.directory.iterdir()) == [
            "model.sock",
            "telemetry.sock",
        ]
    finally:
        bridge.close()
    assert not bridge.model_socket.exists()


def test_model_stream_and_protobuf_export_cross_the_mounted_sockets(
    tmp_path, upstreams
):
    model, export = upstreams
    bridge = LaneBridge(
        tmp_path / "transport",
        f"http://127.0.0.1:{model.server_port}",
        telemetry_upstream=f"http://127.0.0.1:{export.server_port}",
        read_timeout=30,
    )
    guests = [
        loopback_server(
            0,
            str(bridge.model_socket),
            30,
            HopJournal(bridge.directory / "guest-model.jsonl", "guest-model"),
        ),
        loopback_server(
            0,
            str(bridge.telemetry_socket),
            30,
            HopJournal(bridge.directory / "guest-telemetry.jsonl", "guest-telemetry"),
        ),
    ]
    threads = serve(*guests)
    assert all(server.server_address[0] == "127.0.0.1" for server in guests)
    body = json.dumps({"model": "qwen3.8-27b-fp8", "stream": True}).encode()
    received = bytearray()
    seen = threading.Event()

    def read_stream():
        connection = http.client.HTTPConnection(
            "127.0.0.1", guests[0].server_port, timeout=30
        )
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert response.getheader("Transfer-Encoding") is None
        while chunk := response.read1(8192):
            received.extend(chunk)
            if len(received) >= len(STREAM_FIRST):
                seen.set()
        connection.close()

    try:
        reader = threading.Thread(target=read_stream)
        reader.start()
        assert seen.wait(15), "the first event was buffered instead of streamed"
        assert bytes(received)[: len(STREAM_FIRST)] == STREAM_FIRST
        assert bytes(received) != STREAM_FIRST + STREAM_LAST
        model.gate.set()
        reader.join(timeout=15)
        assert not reader.is_alive()
        assert bytes(received) == STREAM_FIRST + STREAM_LAST
        path, headers, forwarded = model.requests[0]
        assert (path, forwarded) == ("/v1/chat/completions", body)
        assert headers["Content-Type"] == "application/json"
        assert headers["Host"] == f"127.0.0.1:{guests[0].server_port}"

        connection = http.client.HTTPConnection(
            "127.0.0.1", guests[1].server_port, timeout=30
        )
        endpoint = "/otel/v1/traces"
        connection.putrequest("POST", endpoint, skip_accept_encoding=True)
        connection.putheader("Content-Type", "application/x-protobuf")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        connection.send(f"{len(EXPORT):x}\r\n".encode() + EXPORT + b"\r\n0\r\n\r\n")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.close()
        path, headers, forwarded = export.requests[0]
        assert (path, forwarded) == (endpoint, EXPORT)
        assert headers["Content-Type"] == "application/x-protobuf"
    finally:
        release_servers(guests, threads)
        bridge.close()
    assert json.loads((bridge.directory / "bridge.json").read_text())["closed"]
    for name in ("model", "guest-model", "telemetry", "guest-telemetry"):
        recorded = entries(bridge.directory, name)
        assert len(recorded) == 1, name
        assert recorded[0]["upstream_dispatched"] is True
        assert "error" not in recorded[0]
        expected = body if name.endswith("model") else EXPORT
        assert recorded[0]["body_sha256"] == hashlib.sha256(expected).hexdigest()
    assert entries(bridge.directory, "model")[0]["response_sha256"] == (
        hashlib.sha256(STREAM_FIRST + STREAM_LAST).hexdigest()
    )


def test_oversized_request_body_is_rejected_before_the_upstream(tmp_path, upstreams):
    model, _ = upstreams
    bridge = LaneBridge(
        tmp_path / "transport", f"http://127.0.0.1:{model.server_port}", read_timeout=30
    )
    guest = loopback_server(
        0,
        str(bridge.model_socket),
        30,
        HopJournal(bridge.directory / "guest-model.jsonl", "guest-model"),
    )
    threads = serve(guest)
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", guest.server_port, timeout=30
        )
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        connection.endheaders()
        assert connection.getresponse().status == 502
        connection.close()
        assert model.requests == []
    finally:
        release_servers([guest], threads)
        bridge.close()
    recorded = entries(bridge.directory, "guest-model")
    assert len(recorded) == 1
    assert recorded[0]["upstream_dispatched"] is False
    assert recorded[0]["error_direction"] == "request_framing"
    assert "exceeds the transport bound" in recorded[0]["error"]
    assert not (bridge.directory / "model.jsonl").exists()


@pytest.mark.parametrize(
    "request_line,message",
    [
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\nContent-Length: 2\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n",
            "conflicting framings",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\nContent-Length: 2\r\n"
            b"Content-Length: 4\r\n\r\n",
            "conflicting content lengths",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: gzip\r\n\r\n",
            "unsupported transfer encoding",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n100000000\r\n",
            "exceeds the transport bound",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n-1\r\n",
            "bounded hexadecimal",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n2\r\nab",
            "no terminator",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n0\r\n",
            "trailer",
        ),
        (
            b"POST /v1/responses HTTP/1.1\r\nHost: lane\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n0\r\nX-Incomplete: yes\r\n",
            "trailer",
        ),
    ],
)
def test_malformed_framing_is_rejected_before_the_upstream(
    tmp_path, upstreams, request_line, message
):
    """A declared 4 GiB chunk or an unterminated trailer never reaches upstream."""
    model, _ = upstreams
    bridge = LaneBridge(
        tmp_path / "transport",
        f"http://127.0.0.1:{model.server_port}",
        read_timeout=30,
        release_seconds=2.0,
    )
    connection = raw_client(bridge.model_socket)
    started = time.monotonic()
    try:
        connection.sendall(request_line)
        connection.shutdown(socket.SHUT_WR)
        status, _ = read_response(connection)
        assert status == 502
        assert time.monotonic() - started < 10
        assert model.requests == []
    finally:
        connection.close()
        bridge.close()
    recorded = entries(bridge.directory, "model")
    assert len(recorded) == 1
    assert recorded[0]["upstream_dispatched"] is False
    assert recorded[0]["error_direction"] in ("request_framing", "request_read")
    assert message in recorded[0]["error"]


def test_valid_chunked_body_with_trailer_terminator_is_forwarded(tmp_path, upstreams):
    model, _ = upstreams
    model.gate.set()
    bridge = LaneBridge(
        tmp_path / "transport", f"http://127.0.0.1:{model.server_port}", read_timeout=30
    )
    payload = b'{"model": "qwen3.8-27b-fp8", "stream": true}'
    connection = raw_client(bridge.model_socket)
    try:
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: lane\r\n"
            b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
            + f"{len(payload):x}\r\n".encode()
            + payload
            + b"\r\n0\r\nX-Checksum: 1\r\n\r\n"
        )
        status, _ = read_response(connection)
        assert status == 200
        assert model.requests[0][2] == payload
    finally:
        connection.close()
        bridge.close()
    recorded = entries(bridge.directory, "model")
    assert recorded[0]["body_sha256"] == hashlib.sha256(payload).hexdigest()
    assert recorded[0]["upstream_dispatched"] is True
    assert "error" not in recorded[0]


def test_slow_client_body_cannot_block_bridge_release(tmp_path, upstreams):
    """A declared body that never arrives is woken by the bounded release."""
    model, _ = upstreams
    bridge = LaneBridge(
        tmp_path / "transport",
        f"http://127.0.0.1:{model.server_port}",
        read_timeout=3600,
        release_seconds=2.0,
    )
    connection = raw_client(bridge.model_socket)
    connection.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: lane\r\n"
        b"Content-Length: 10\r\n\r\n{"  # one body byte of ten
    )
    wait_for_handler(bridge.model)
    started = time.monotonic()
    try:
        bridge.close()
    finally:
        connection.close()
    assert time.monotonic() - started < 10
    record = json.loads((bridge.directory / "bridge.json").read_text())
    assert record["closed"] is True, record
    assert record["handlers_released"] is True
    assert record["removed_sockets"] == ["model.sock"]
    recorded = entries(bridge.directory, "model")[0]
    assert recorded["upstream_dispatched"] is False
    assert recorded["error_direction"] == "request_read"


def test_stalled_upstream_stream_cannot_block_the_telemetry_bridge(tmp_path):
    pairs = start_upstreams(StalledHandler, StalledHandler)
    (model, _model_thread), (export, _export_thread) = pairs
    try:
        bridge = LaneBridge(
            tmp_path / "transport",
            f"http://127.0.0.1:{model.server_port}",
            telemetry_upstream=f"http://127.0.0.1:{export.server_port}",
            read_timeout=3600,
            release_seconds=2.0,
        )
        payload = b"{}"
        connections = [
            raw_client(bridge.model_socket),
            raw_client(bridge.telemetry_socket),
        ]
        for connection in connections:
            connection.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: lane\r\n"
                b"Content-Length: 2\r\n\r\n" + payload
            )
        for server in (model, export):
            assert server.gate.wait(15), "the stalled upstream never received a request"
        received = []
        for connection in connections:
            connection.settimeout(15)
            first = connection.recv(4096)
            assert first.startswith(b"HTTP/1.0 200")
            received.append(bytearray(first))
        for server in (bridge.model, bridge.telemetry):
            wait_for_handler(server)
        started = time.monotonic()
        bridge.close()
        assert time.monotonic() - started < 10
        for connection, relayed in zip(connections, received, strict=True):
            while chunk := connection.recv(4096):
                relayed += chunk
            assert STREAM_FIRST in bytes(relayed)
            connection.close()
    finally:
        stop_upstreams(*pairs)
    record = json.loads((bridge.directory / "bridge.json").read_text())
    assert record["closed"] is True, record
    assert record["removed_sockets"] == ["model.sock", "telemetry.sock"]
    for name in ("model", "telemetry"):
        recorded = entries(bridge.directory, name)[0]
        assert recorded["upstream_dispatched"] is True
        assert recorded["error_direction"] == "upstream_read"


def test_incomplete_handler_release_keeps_sockets_and_reports_it(tmp_path, upstreams):
    """Release failure must keep the socket and fail the cleanup record."""
    model, _ = upstreams
    bridge = LaneBridge(
        tmp_path / "transport", f"http://127.0.0.1:{model.server_port}", read_timeout=30
    )
    try:
        bridge.model.release = lambda timeout=0: False
        with pytest.raises(ValueError, match="release exceeded its bound"):
            bridge.close()
        record = json.loads((bridge.directory / "bridge.json").read_text())
        assert record["closed"] is False
        assert record["handlers_released"] is False
        assert record["remaining_sockets"] == ["model.sock"]
        assert bridge.model_socket.exists()
    finally:
        bridge.model_socket.unlink(missing_ok=True)


@pytest.mark.parametrize("failure", ["upstream", "threads"])
def test_partial_start_rolls_back_the_model_listener(tmp_path, monkeypatch, failure):
    import benchmarks.agentic_lane_transport as transport

    model, model_thread = upstream_server(ModelHandler)
    directory = tmp_path / "transport"
    telemetry = (
        "http://collector.invalid:8124"
        if failure == "upstream"
        else f"http://127.0.0.1:{model.server_port}"
    )

    def fail(*servers):
        raise ImportError("injected startup failure")

    if failure == "threads":
        monkeypatch.setattr(transport, "serve", fail)
    try:
        with pytest.raises((ValueError, ImportError)) as caught:
            LaneBridge(
                directory,
                f"http://127.0.0.1:{model.server_port}",
                telemetry_upstream=telemetry,
                read_timeout=30,
            )
        expected = (
            "local plain HTTP" if failure == "upstream" else "injected startup failure"
        )
        assert expected in str(caught.value)
        assert not (directory / "model.sock").exists()
        assert not (directory / "telemetry.sock").exists()
        record = json.loads((directory / "bridge.json").read_text())
        assert record["handlers_released"] is True
        assert record["remaining_sockets"] == []
        with pytest.raises(OSError):
            raw_client(directory / "model.sock")
    finally:
        stop_server(model, model_thread)


def write_lane_closure(
    lane: Path,
    *,
    model=(),
    guest=(),
    telemetry=None,
    declared=None,
    **overrides,
):
    """Write one complete hop journal and closure record set for a lane."""
    transport = lane / "transport"
    transport.mkdir(parents=True, exist_ok=True)
    configured = ["model"] + (["telemetry"] if telemetry is not None else [])
    contents = {
        "model": model,
        "guest-model": guest,
        "telemetry": telemetry or [],
        "guest-telemetry": telemetry or [],
    }
    for name, entries in contents.items():
        path = transport / f"{name}.jsonl"
        if name.endswith("telemetry") and telemetry is None:
            path.unlink(missing_ok=True)
            continue
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    bridge = {
        "closed": True,
        "errors": [],
        "handlers_released": True,
        "remaining_sockets": [],
        "telemetry_upstream": ["127.0.0.1", 1] if telemetry is not None else None,
        "journals": {
            f"bridge-{name}": {
                "entries": (
                    len(contents[name])
                    if declared is None
                    else declared["model" if name == "model" else "telemetry"]
                ),
                "error": None,
            }
            for name in configured
        },
    }
    guest_report = {
        "released": True,
        "servers": [
            {
                "errors": [],
                "journal_error": None,
                "journal_entries": len(contents[f"guest-{name}"]),
            }
            for name in configured
        ],
    }
    bridge.update(overrides)
    (transport / "bridge.json").write_text(json.dumps(bridge))
    (transport / "guest-close.json").write_text(json.dumps(guest_report))


def test_hop_audit_rejects_unexplained_model_failure(tmp_path):
    lane = tmp_path / "lane"
    (lane / "wire").mkdir(parents=True)
    (lane / "sdk-trace").mkdir()
    (lane / "transport").mkdir()
    body = b'{"model": "qwen3.8-27b-fp8"}'
    (lane / "wire/0000-request.json").write_bytes(body)
    (lane / "wire/http.jsonl").write_text(
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    )
    digest = hashlib.sha256(body).hexdigest()
    reached = {
        "path": "/v1/chat/completions",
        "body_sha256": digest,
        "upstream_dispatched": True,
    }

    write_lane_closure(lane, model=[reached], guest=[reached])
    evidence = lane_transport_evidence(lane)
    assert evidence["bridge-model"]["reached_records"] == 1
    assert evidence["guest-model"]["reached_records"] == 1
    assert evidence["bridge_errors"] == [] and evidence["guest_errors"] == []
    assert set(evidence) == {
        "bridge-model",
        "guest-model",
        "bridge_errors",
        "guest_errors",
        "explained_aborts",
        "unused_explanations",
    }

    other = {**reached, "body_sha256": hashlib.sha256(b"other").hexdigest()}
    write_lane_closure(lane, model=[other], guest=[other])
    with pytest.raises(ValueError, match="not among the original requests"):
        lane_transport_evidence(lane)

    failed = {
        **other,
        "error": "ConnectionRefusedError: refused",
        "error_direction": "upstream_connect",
    }
    write_lane_closure(lane, model=[failed], guest=[failed])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(lane)

    rejected = {
        "path": "/v1/chat/completions",
        "upstream_dispatched": False,
        "error": "FramingError: request body exceeds the transport bound",
        "error_direction": "request_framing",
    }
    # A hop rejection never reaches the collector, so it records no request.
    (lane / "wire/http.jsonl").write_text("")
    write_lane_closure(lane, model=[rejected], guest=[rejected])
    assert lane_transport_evidence(lane)["bridge-model"]["explained_rejections"] == 1
    (lane / "wire/http.jsonl").write_text(
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    )

    # Two identical attempts must not collapse into one recorded request.
    write_lane_closure(lane, model=[reached, reached], guest=[reached, reached])
    with pytest.raises(ValueError, match="relays more attempts than the collector"):
        lane_transport_evidence(lane)

    # A failed retry next to a success needs its own recorded explanation.
    retry = failed_entry(digest)
    write_lane_closure(lane, model=[reached, retry], guest=[reached, retry])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(lane)

    # One success plus one explained failure still describes two attempts for
    # one recorded request: a matching digest alone must not excuse it.
    with pytest.raises(ValueError, match="explains more attempts than the collector"):
        lane_transport_evidence(lane, explained={"model": {digest: [abort_bound()]}})

    # Two recorded originals with identical bodies do explain two attempts.
    two_records = (
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    ) * 2
    (lane / "wire/http.jsonl").write_text(two_records)
    explained = lane_transport_evidence(
        lane, explained={"model": {digest: [abort_bound()]}}
    )
    assert explained["bridge-model"]["reached_records"] == 1
    assert explained["bridge-model"]["explained_aborts"] == 1
    assert explained["explained_aborts"] == {"model": [digest]}

    # Each explanation binds exactly one original attempt: two identical
    # failures against one bound leave the second failure unexplained until its
    # own recorded original supplies another bound.
    write_lane_closure(lane, model=[retry, retry], guest=[retry, retry])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(lane, explained={"model": {digest: [abort_bound()]}})
    doubled = lane_transport_evidence(
        lane, explained={"model": {digest: [abort_bound(), abort_bound()]}}
    )
    assert doubled["bridge-model"]["explained_aborts"] == 2

    # A relayed request whose client aborted downstream keeps its explanation
    # reported as unused instead of failing the lane.
    (lane / "wire/http.jsonl").write_text(
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    )
    write_lane_closure(lane, model=[reached], guest=[reached])
    relayed = lane_transport_evidence(
        lane, explained={"model": {digest: [abort_bound()]}}
    )
    assert relayed["unused_explanations"] == [f"model:{digest}"]

    # An explanation for a hop the lane never configured is not evidence.
    with pytest.raises(ValueError, match="unconfigured hop"):
        lane_transport_evidence(
            lane, explained={"telemetry": {digest: [abort_bound()]}}
        )

    # A lane that recorded no request binds nothing.
    (lane / "wire/http.jsonl").write_text("")
    write_lane_closure(lane)
    assert lane_transport_evidence(lane)["bridge-model"]["reached_records"] == 0

    for override, message in (
        ({"closed": False}, "did not close cleanly"),
        ({"handlers_released": False}, "handlers were not released"),
        ({"errors": ["cleanup failed"]}, "reported cleanup errors"),
        ({"remaining_sockets": ["model.sock"]}, "left mounted sockets behind"),
    ):
        write_lane_closure(lane, **override)
        with pytest.raises(ValueError, match=message):
            lane_transport_evidence(lane)

    write_lane_closure(lane, declared={"model": 2})
    with pytest.raises(ValueError, match="closure record"):
        lane_transport_evidence(lane)

    write_lane_closure(lane)
    guest_report = json.loads((lane / "transport/guest-close.json").read_text())
    guest_report["released"] = False
    (lane / "transport/guest-close.json").write_text(json.dumps(guest_report))
    with pytest.raises(ValueError, match="were not released"):
        lane_transport_evidence(lane)

    write_lane_closure(lane)
    guest_report["released"] = True
    guest_report["servers"][0]["errors"] = ["boom"]
    (lane / "transport/guest-close.json").write_text(json.dumps(guest_report))
    with pytest.raises(ValueError, match="guest lane endpoint reported errors"):
        lane_transport_evidence(lane)

    write_lane_closure(lane, telemetry=[], declared={"model": 0, "telemetry": 3})
    with pytest.raises(ValueError, match="closure record"):
        lane_transport_evidence(lane)


def test_explanation_binds_one_original_direction_and_window(tmp_path):
    """A digest alone never explains an unrelated or mistimed hop failure."""
    lane = tmp_path / "lane"
    (lane / "wire").mkdir(parents=True)
    (lane / "sdk-trace").mkdir()
    body = b'{"model": "qwen3.8-27b-fp8"}'
    (lane / "wire/0000-request.json").write_bytes(body)
    (lane / "wire/http.jsonl").write_text(
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    )
    digest = hashlib.sha256(body).hexdigest()
    bound = {digest: [abort_bound()]}

    # A canonical disconnect for the wrong direction is not this original's.
    wrong_direction = failed_entry(digest, direction="upstream_read")
    write_lane_closure(lane, model=[wrong_direction], guest=[wrong_direction])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(lane, explained={"model": bound})

    # An error the canonical disconnect never records is not this original's.
    wrong_kind = failed_entry(digest, kind="TimeoutError")
    write_lane_closure(lane, model=[wrong_kind], guest=[wrong_kind])
    with pytest.raises(ValueError, match="unexplained bridge model transport failure"):
        lane_transport_evidence(lane, explained={"model": bound})

    # A failure that ended before the deadline, just before the abort action or
    # after the recorded client disconnect is a different attempt, however well
    # its digest matches.
    for received, completed in (
        (150.0, 160.0),
        (199.5, 199.9999),
        (199.5, 200.0024),
        (199.5, 200.801),
    ):
        mistimed = failed_entry(digest, received=received, completed=completed)
        write_lane_closure(lane, model=[mistimed], guest=[mistimed])
        with pytest.raises(
            ValueError, match="unexplained bridge model transport failure"
        ):
            lane_transport_evidence(lane, explained={"model": bound})

    # A completion just after the recorded abort action is this request's own
    # release and stays accepted.
    released = failed_entry(digest, completed=200.0026)
    write_lane_closure(lane, model=[released], guest=[released])
    assert (
        lane_transport_evidence(lane, explained={"model": bound})["bridge-model"][
            "explained_aborts"
        ]
        == 1
    )

    # A completed hop stays legitimate: one hop may have relayed the request
    # successfully while the other recorded the explained failure.
    failed = failed_entry(digest)
    reached = {
        "path": "/v1/chat/completions",
        "body_sha256": digest,
        "upstream_dispatched": True,
    }
    write_lane_closure(lane, model=[failed], guest=[reached])
    asymmetric = lane_transport_evidence(lane, explained={"model": bound})
    assert asymmetric["bridge-model"]["explained_aborts"] == 1
    assert asymmetric["guest-model"]["reached_records"] == 1
    assert asymmetric["guest-model"]["explained_aborts"] == 0

    # With no failed attempt anywhere, that explanation stays reported as
    # unused: the client aborted downstream of hops that relayed the request,
    # and a hop that never relayed it still fails the coverage check.
    write_lane_closure(lane, model=[reached], guest=[reached])
    unused = lane_transport_evidence(
        lane, explained={"model": {digest: [abort_bound()]}}
    )
    assert unused["bridge-model"]["explained_aborts"] == 0
    assert unused["guest-model"]["explained_aborts"] == 0
    assert unused["unused_explanations"] == [f"model:{digest}"]
    # A request the hops never relayed cannot be explained away at all.
    (lane / "wire/http.jsonl").write_text(
        json.dumps(
            {"request_file": "0000-request.json", "path": "/v1/chat/completions"}
        )
        + "\n"
    )
    write_lane_closure(lane, model=[], guest=[])
    with pytest.raises(ValueError, match="is missing a recorded attempt"):
        lane_transport_evidence(lane, explained={"model": {digest: [abort_bound()]}})


def test_deadline_bound_keeps_the_canonical_disconnect_evidence():
    """The bound comes from the recorded proof, not from a digest or a slack."""
    proof = {
        "kind": "host_collection_deadline_abort",
        "response_id": "chatcmpl_1",
        "deadline_monotonic_s": 200.0,
        "disconnect_direction": "downstream_write",
        "disconnect_monotonic_s": 200.8,
        "collection_ended_monotonic_s": 201.0,
        "abort": {"start": {"monotonic_s": 200.0025}},
    }
    bound = deadline_attempt_bound(proof)
    assert bound["direction"] == "downstream_write"
    assert bound["kinds"] == ["BrokenPipeError", "ConnectionResetError"]
    assert bound["deadline_monotonic_s"] == 200.0
    # The window starts at the recorded abort action, not before the deadline.
    assert bound["first_monotonic_s"] == 200.0025
    assert bound["last_monotonic_s"] == 200.8
    for broken, message in (
        ({"disconnect_direction": "upstream_read"}, "canonical disconnect"),
        (
            {"abort": {"start": {"monotonic_s": 199.999}}},
            "reversed abort clocks",
        ),
        ({"disconnect_monotonic_s": 200.0}, "reversed abort clocks"),
    ):
        with pytest.raises(ValueError, match=message):
            deadline_attempt_bound({**proof, **broken})


def test_hop_audit_supports_model_only_and_zero_request_lanes(tmp_path):
    """The one-off collector mounts one model socket and may send nothing."""
    lane = tmp_path / "lane"
    (lane / "wire").mkdir(parents=True)
    (lane / "sdk-trace").mkdir()
    (lane / "wire/http.jsonl").write_text("")
    write_lane_closure(lane)
    evidence = lane_transport_evidence(lane)
    assert evidence["bridge-model"]["reached_records"] == 0
    assert "bridge-telemetry" not in evidence
    # A telemetry journal without its configured socket is not evidence.
    (lane / "transport/telemetry.jsonl").write_text("")
    with pytest.raises(ValueError, match="unconfigured telemetry hop"):
        lane_transport_evidence(lane)


SERVE_ROLLBACK = """
import json, socket, sys, threading
from pathlib import Path

from benchmarks.agentic_lane_forwarder import HopJournal, loopback_server, serve

directory, fail_from = Path(sys.argv[1]), int(sys.argv[2])
servers = [
    loopback_server(
        0,
        str(directory / f"{name}.sock"),
        5,
        HopJournal(directory / f"{name}.jsonl", name),
    )
    for name in ("a", "b")
]
ports = [server.server_address[1] for server in servers]
real_start, calls = threading.Thread.start, {"count": 0}


def failing_start(self):
    calls["count"] += 1
    if calls["count"] >= fail_from:
        raise RuntimeError(f"thread allocation refused at {calls['count']}")
    return real_start(self)


threading.Thread.start = failing_start
failure, notes = None, []
try:
    serve(*servers)
except RuntimeError as error:
    failure, notes = str(error), list(getattr(error, "__notes__", []))
report = {
    "failure": failure,
    "notes": notes,
    "starts": calls["count"],
    "serving": [server.serving for server in servers],
    "alive": [
        server.thread is not None and server.thread.is_alive() for server in servers
    ],
    "sockets": [server.socket.fileno() for server in servers],
    "errors": [server.errors for server in servers],
    "rebound": [],
}
for port in ports:
    probe = socket.socket()
    probe.bind(("127.0.0.1", port))
    probe.close()
    report["rebound"].append(port)
print(json.dumps(report))
"""


@pytest.mark.parametrize("fail_from", [1, 2])
def test_persistent_thread_start_exhaustion_releases_every_listener(
    tmp_path, fail_from
):
    """Cleanup must not need a new thread when allocation stays refused.

    The first listener starts normally when ``fail_from`` is two, so rollback
    has to stop a serving listener while every further allocation fails.
    """
    directory = tmp_path / "transport"
    directory.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", SERVE_ROLLBACK, str(directory), str(fail_from)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        },
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # The original start failure stays authoritative: no cleanup allocation
    # may replace it, so exactly one start attempt beyond it may have happened.
    assert report["failure"] == f"thread allocation refused at {fail_from}", report
    assert report["starts"] == fail_from, report
    assert any("lane listener start failed" in note for note in report["notes"])
    assert report["serving"] == [False, False]
    assert report["alive"] == [False, False]
    assert report["sockets"] == [-1, -1]
    assert report["errors"] == [[], []], report
    assert len(report["rebound"]) == 2


def wait_for_release(server, bound: float = 5.0):
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if not server.tracked_sockets() and not server.handler_count():
            return
        time.sleep(0.01)
    raise AssertionError(
        f"tracked sockets stayed at {server.tracked_sockets()} "
        f"with {server.handler_count()} handlers"
    )


class SmallHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class Http10CloseHandler(SmallHandler):
    """Reply like a close-framed HTTP/1.0 upstream."""

    protocol_version = "HTTP/1.0"


class Http11CloseHandler(SmallHandler):
    """Reply like an HTTP/1.1 upstream that closes the connection."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b"ok")


def drive_real_requests(bridge, model, count: int = 20):
    """Send completed requests through the mounted socket and await release."""
    for index in range(count):
        connection = raw_client(bridge.model_socket)
        body = json.dumps({"model": "m", "index": index}).encode()
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.0\r\nHost: lane\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        status, payload = read_response(connection)
        assert (status, payload) == (200, b"ok")
        connection.close()
        wait_for_release(bridge.model)
    assert len(model.requests) == count


def test_completed_requests_release_their_upstream_sockets(tmp_path):
    """Twenty real POSTs must not retain closed upstream socket owners."""
    model, thread = upstream_server(SmallHandler)
    directory = tmp_path / "transport"
    bridge = LaneBridge(
        directory, f"http://127.0.0.1:{model.server_port}", read_timeout=30
    )
    try:
        drive_real_requests(bridge, model)
    finally:
        bridge.close()
        stop_server(model, thread)
    assert len(journal_records(directory / "model.jsonl")) == 20
    assert json.loads((directory / "bridge.json").read_text())["closed"] is True


@pytest.mark.parametrize("handler", [Http10CloseHandler, Http11CloseHandler])
def test_close_framed_requests_release_their_upstream_sockets(tmp_path, handler):
    """Both production hops use close framing, which detaches HTTPConnection.sock.

    The exact socket object that was registered must still be released, so
    neither an HTTP/1.0 upstream nor an explicit ``Connection: close`` reply
    may leave retained owners behind.
    """
    model, thread = upstream_server(handler)
    directory = tmp_path / "transport"
    bridge = LaneBridge(
        directory, f"http://127.0.0.1:{model.server_port}", read_timeout=30
    )
    try:
        drive_real_requests(bridge, model)
        assert bridge.model.tracked_sockets() == 0
    finally:
        bridge.close()
        stop_server(model, thread)
    assert len(journal_records(directory / "model.jsonl")) == 20


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_guest_entry_point_serves_a_child_process_through_the_socket(
    tmp_path, upstreams
):
    """The container entry point must expose loopback before the client starts."""
    model, export = upstreams
    model_port, telemetry_port = free_port(), free_port()
    bridge = LaneBridge(
        tmp_path / "transport",
        f"http://127.0.0.1:{model.server_port}",
        telemetry_upstream=f"http://127.0.0.1:{export.server_port}",
        read_timeout=30,
        model_port=model_port,
        telemetry_port=telemetry_port,
    )
    child = (
        "import json,urllib.request;"
        f"request=urllib.request.Request('http://127.0.0.1:{model_port}/v1/chat/completions',"
        "data=json.dumps({'model':'m'}).encode(),"
        "headers={'Content-Type':'application/json'});"
        "response=urllib.request.urlopen(request,timeout=20);"
        "print(response.status, len(response.read()))"
    )
    guest_entry = bridge.entry_point([sys.executable, "-c", child])
    command = [
        sys.executable,
        str(ROOT / "benchmarks/agentic_lane_forwarder.py"),
        *guest_entry[2:],
    ]
    assert guest_entry[:2] == ["/usr/bin/python3", "/usr/local/bin/lane-forward.py"]
    try:
        model.gate.set()
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, cwd=ROOT
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("200 ")
        assert model.requests[0][0] == "/v1/chat/completions"
        close = json.loads((bridge.directory / "guest-close.json").read_text())
        assert close["released"] is True
        assert close["servers"][0]["journal_entries"] == 1

        stale_entry = bridge.entry_point([sys.executable, "-c", "pass"])
        bridge.close()
        bad = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/agentic_lane_forwarder.py"),
                *stale_entry[2:],
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
        )
        assert bad.returncode != 0
        assert "missing" in bad.stderr
    finally:
        bridge.close()
