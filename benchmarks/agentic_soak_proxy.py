"""Disk-backed HTTP evidence proxy for the optional real-agent soak collector."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.agentic_soak_protocol import StreamCompletion

MAX_REQUEST_BYTES = 8 << 20
MAX_EVENT_BYTES = 1 << 20
TERMINAL_CAPTURE_SECONDS = 2.0


def record_transport_error(record, error):
    """Keep raw errors; only a captured protocol terminal makes a candidate."""
    if record.get("error"):
        record.setdefault("capture_error", f"{type(error).__name__}: {error}")
        record.setdefault("capture_error_direction", record.get("error_direction"))
        record["transport_outcome"] = "failed"
        return
    record["error"] = f"{type(error).__name__}: {error}"
    record["transport_error_kind"] = type(error).__name__
    record["transport_outcome"] = (
        "terminal_disconnect"
        if isinstance(error, (ConnectionResetError, BrokenPipeError))
        and record.get("error_direction") in ("upstream_read", "downstream_write")
        and record.get("protocol_completion", {}).get("success")
        else "failed"
    )


def relay_stream(response, output, write, record, api, *, set_read_timeout=None):
    """Stop at the protocol terminal; retain transport direction and raw bytes."""
    proof = StreamCompletion(api, MAX_EVENT_BYTES)
    digest, total = hashlib.sha256(), 0
    read = getattr(response, "read1", response.read)
    disconnect, capture_deadline = None, None
    try:
        while True:
            record["error_direction"] = "upstream_read"
            if capture_deadline is not None:
                remaining = capture_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "upstream terminal capture exceeded its deadline"
                    )
                if set_read_timeout is not None:
                    set_read_timeout(remaining)
            chunk = read(8192)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
            record["error_direction"] = "protocol_parse"
            proof.feed(chunk)
            if proof.errors:
                raise ValueError("upstream SSE has invalid or failed protocol evidence")
            if disconnect is None:
                record["error_direction"] = "downstream_write"
                try:
                    write(chunk)
                except (ConnectionResetError, BrokenPipeError) as error:
                    if proof.finish_reason not in ("stop", "tool_calls", "length"):
                        raise
                    disconnect = error
                    record_transport_error(record, error)
                    capture_deadline = time.monotonic() + TERMINAL_CAPTURE_SECONDS
            if proof.terminal_kind is not None:
                proof.terminal_forwarded = disconnect is None
                break
        record["error_direction"] = "protocol_parse"
        if not proof.snapshot()["success"]:
            proof.bad("missing_successful_terminal")
            raise ValueError("upstream SSE ended without a valid successful terminal")
        record["transport_outcome"] = "completed"
    except Exception as error:
        if disconnect is not None:
            record["capture_error"] = f"{type(error).__name__}: {error}"
            record["capture_error_direction"] = record["error_direction"]
        raise
    finally:
        record.update(
            protocol_completion=proof.snapshot(),
            response_sha256=digest.hexdigest(),
            response_bytes=total,
            response_ids=sorted(proof.ids),
        )
        if disconnect is not None:
            record["error_direction"] = "downstream_write"
            record["transport_outcome"] = (
                "terminal_disconnect"
                if proof.snapshot()["success"] and not record.get("capture_error")
                else "failed"
            )
            record["terminal_capture_budget_seconds"] = TERMINAL_CAPTURE_SECONDS


def set_upstream_timeout(response, remaining):
    """Bound the remaining read with the socket owned by urllib's HTTP response."""
    response.fp.raw._sock.settimeout(remaining)


class ModelSlots:
    """Share four model-call slots; a cancelled waiter cannot start late work."""

    def __init__(self, maximum: int = 4):
        if maximum != 4:
            raise ValueError("real-agent soak requires exactly four model-call slots")
        self.semaphore = threading.BoundedSemaphore(maximum)
        self.lock = threading.Lock()
        self.active = self.peak = 0

    @contextmanager
    def slot(self, stop: threading.Event | None = None):
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("evidence proxy closed before slot admission")
            if self.semaphore.acquire(timeout=0.1):
                break
        if stop is not None and stop.is_set():
            self.semaphore.release()
            raise InterruptedError("evidence proxy closed before forwarding")
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
            self.semaphore.release()


@contextmanager
def disk_proxy(
    upstream_url: str,
    directory: Path,
    slots: ModelSlots,
    *,
    request_timeout: float = 180,
    normal_completion=None,
    preflight=None,
    request_context=None,
    cancel_preflight=None,
):
    """Close every owned socket and join handlers before final evidence is read."""
    upstream = urllib.parse.urlsplit(upstream_url)
    if upstream.scheme != "http" or upstream.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise ValueError("soak evidence proxy requires a local plain HTTP service")
    directory.mkdir(parents=True, exist_ok=False)
    records, sockets, lock = [], set(), threading.Lock()
    completed = threading.Condition(lock)
    stop = threading.Event()
    sequence = inflight = 0

    class Connection(http.client.HTTPConnection):
        def connect(self):
            if stop.is_set():
                raise InterruptedError("proxy is closed")
            # Track the connected socket before request/response headers. A
            # client cancelled during preparation cannot leave urlopen blocked.
            original_timeout = self.timeout
            self.timeout = 5
            super().connect()
            self.timeout = original_timeout
            self.sock.settimeout(original_timeout)
            with lock:
                sockets.add(self.sock)
                if stop.is_set():
                    self.sock.shutdown(socket.SHUT_RDWR)
                    raise InterruptedError("proxy closed during connection")

    class TrackingHandler(urllib.request.HTTPHandler):
        def http_open(self, request):
            return self.do_open(Connection, request)

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), TrackingHandler()
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)
            with lock:
                sockets.add(self.connection)
                if stop.is_set():
                    self.connection.shutdown(socket.SHUT_RDWR)

        def log_message(self, format, *args):
            pass

        def forward(self):
            nonlocal sequence, inflight
            with lock:
                index = sequence
                sequence += 1
                inflight += 1
            body_path = directory / f"{index:04d}-request.json"
            response_path = directory / f"{index:04d}-response.bin"
            record = {
                "method": self.command,
                "path": self.path,
                "client_received_unix_s": time.time(),
                "request_file": body_path.name,
                "response_file": response_path.name,
            }
            response = None
            try:
                record["error_direction"] = "request_read"
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= MAX_REQUEST_BYTES:
                    raise ValueError("client request exceeds the evidence proxy bound")
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError("client request body was truncated")
                body_path.write_bytes(data)
                body = json.loads(data) if data else {}
                if self.path not in ("/v1/chat/completions", "/v1/responses"):
                    raise ValueError("soak proxy accepts only model-generation routes")
                if request_context is not None:
                    record.update(request_context())
                if preflight is not None:
                    record["context_proof"] = preflight(body, self.path)
                request = urllib.request.Request(
                    upstream_url.rstrip("/") + self.path.removeprefix("/v1"),
                    data=data if data else None,
                    method=self.command,
                    headers={
                        key: value
                        for key, value in self.headers.items()
                        if key.lower() not in ("host", "content-length", "connection")
                    },
                )
                with slots.slot(stop):
                    if stop.is_set():
                        raise InterruptedError("proxy closed before upstream dispatch")
                    record["upstream_started_unix_s"] = time.time()
                    record["error_direction"] = "upstream_headers"
                    try:
                        response = opener.open(request, timeout=request_timeout)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response, response_path.open("xb") as output:
                        record["response_status"] = response.status
                        record["error_direction"] = "downstream_write"
                        self.send_response(response.status)
                        for key, value in response.headers.items():
                            if key.lower() not in (
                                "content-length",
                                "transfer-encoding",
                                "connection",
                            ):
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()

                        def write(chunk):
                            self.wfile.write(chunk)
                            self.wfile.flush()

                        relay_stream(
                            response,
                            output,
                            write,
                            record,
                            "chat"
                            if self.path == "/v1/chat/completions"
                            else "responses",
                            set_read_timeout=lambda remaining: set_upstream_timeout(
                                response, remaining
                            ),
                        )
            except Exception as error:
                record_transport_error(record, error)
                if response is None:
                    with suppress(OSError):
                        self.send_error(502, "soak evidence forwarding failed")
            finally:
                # Only the handler closes its buffered response. Cross-thread
                # close() can block on its reader lock; shutdown() wakes it first.
                if response is not None:
                    response.close()
                with lock:
                    record.update(
                        response_sha256=record.get(
                            "response_sha256", hashlib.sha256().hexdigest()
                        ),
                        response_bytes=record.get("response_bytes", 0),
                        response_ids=record.get("response_ids", []),
                        client_completed_unix_s=time.time(),
                    )
                    records.append(record)
                    try:
                        with (directory / "http.jsonl").open("a") as output:
                            output.write(json.dumps(record, allow_nan=False) + "\n")
                    finally:
                        inflight -= 1
                        completed.notify_all()

        do_POST = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", records
    except BaseException:
        if cancel_preflight is not None:
            cancel_preflight()
        raise
    finally:
        try:
            if normal_completion is not None and normal_completion():
                deadline = time.monotonic() + TERMINAL_CAPTURE_SECONDS
                with completed:
                    while inflight and (remaining := deadline - time.monotonic()) > 0:
                        completed.wait(remaining)
        finally:
            stop.set()
            # A connection created after this snapshot sees stop in connect/setup
            # and shuts itself down. Existing readers wake before handler joining.
            with lock:
                owned_sockets = list(sockets)
            for connection in owned_sockets:
                with suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("evidence proxy listener did not stop")


def load_requests(directory: Path, *, maximum: int = 32) -> list[dict]:
    """Read a finished task's evidence after every handler has joined."""
    records = [
        json.loads(line) for line in (directory / "http.jsonl").read_text().splitlines()
    ]
    if len(records) > maximum:
        raise ValueError("OMP requests exceeded the bounded evidence window")
    for record in records:
        body = directory / record["request_file"]
        record["request"] = json.loads(body.read_text())
    return records
