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

from scripts.acceptance_proxy import response_ids

MAX_REQUEST_BYTES = 8 << 20
MAX_EVENT_BYTES = 1 << 20


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
    stop = threading.Event()
    sequence = 0

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
            nonlocal sequence
            with lock:
                index = sequence
                sequence += 1
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
            ids, pending = set(), bytearray()
            digest, total = hashlib.sha256(), 0
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= MAX_REQUEST_BYTES:
                    raise ValueError("client request exceeds the evidence proxy bound")
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError("client request body was truncated")
                body_path.write_bytes(data)
                if data:
                    json.loads(data)
                if self.path not in ("/v1/chat/completions", "/v1/responses"):
                    raise ValueError("soak proxy accepts only model-generation routes")
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
                    try:
                        response = opener.open(request, timeout=request_timeout)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response, response_path.open("xb") as output:
                        record["response_status"] = response.status
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
                        read = getattr(response, "read1", response.read)
                        while chunk := read(8192):
                            output.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                            pending.extend(chunk)
                            while b"\n" in pending:
                                line, _, rest = pending.partition(b"\n")
                                pending[:] = rest
                                ids.update(response_ids(line))
                            if len(pending) > MAX_EVENT_BYTES:
                                raise ValueError(
                                    "server SSE event exceeds the evidence proxy bound"
                                )
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        ids.update(response_ids(pending))
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
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
                        response_sha256=digest.hexdigest(),
                        response_bytes=total,
                        response_ids=sorted(ids),
                        client_completed_unix_s=time.time(),
                    )
                    records.append(record)
                    with (directory / "http.jsonl").open("a") as output:
                        output.write(json.dumps(record, allow_nan=False) + "\n")

        do_POST = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", records
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


def load_requests(directory: Path) -> list[dict]:
    """Read a finished task's evidence after every handler has joined."""
    records = [
        json.loads(line) for line in (directory / "http.jsonl").read_text().splitlines()
    ]
    if len(records) > 32:
        raise ValueError("one OMP task exceeded the bounded 32-call evidence window")
    for record in records:
        body = directory / record["request_file"]
        record["request"] = json.loads(body.read_text())
    return records
