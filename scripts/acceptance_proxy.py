"""Record actual OMP HTTP payloads while forwarding to the real model service."""

import hashlib
import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def response_ids(body: bytes) -> list[str]:
    """Extract response identities from complete JSON or streamed SSE records."""
    values = []
    try:
        values.append(json.loads(body))
    except ValueError:
        for line in body.splitlines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                try:
                    values.append(json.loads(line[6:]))
                except ValueError:
                    continue
    ids = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        rid = value.get("response", {}).get("id") or value.get("id")
        if isinstance(rid, str) and rid.startswith(("chatcmpl_", "resp_")):
            ids.add(rid)
    return sorted(ids)


@contextmanager
def recording_proxy(upstream_url: str, *, raw_directory: Path | None = None):
    """Own a loopback listener and retain complete upstream request evidence.

    HTTP/1.0 closes each downstream stream after its body. The upstream SSE
    content is forwarded without reconstruction, tool edits, or extra requests.
    """
    if raw_directory is not None:
        raw_directory.mkdir(parents=True, exist_ok=False)
    records, sockets = [], set()
    lock = threading.Lock()
    stop = threading.Event()

    class Connection(http.client.HTTPConnection):
        def connect(self):
            if stop.is_set():
                raise InterruptedError("proxy closed before upstream connection")
            self.sock = socket.socket(
                socket.AF_INET6 if ":" in self.host else socket.AF_INET,
                socket.SOCK_STREAM,
            )
            self.sock.settimeout(self.timeout)
            with lock:
                sockets.add(self.sock)
                if stop.is_set():
                    self.sock.close()
                    raise InterruptedError("proxy closed during upstream connection")
            self.sock.connect((self.host, self.port))

    class TrackingHandler(urllib.request.HTTPHandler):
        def http_open(self, request):
            return self.do_open(Connection, request)

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), TrackingHandler()
    )

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            with lock:
                sockets.add(self.request)
                if stop.is_set():
                    self.request.close()
            super().setup()

        def log_message(self, format, *args):
            pass

        def forward(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            record = {"method": self.command, "path": self.path}
            if data:
                record["request"] = json.loads(data)
            with lock:
                index = len(records)
                records.append(record)
            suffix = self.path.removeprefix("/v1")
            request = urllib.request.Request(
                upstream_url.rstrip("/") + suffix,
                data=data if data else None,
                method=self.command,
                headers={
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in ("host", "content-length", "connection")
                },
            )
            body = bytearray()
            try:
                try:
                    response = opener.open(request, timeout=900)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
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
                        body.extend(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError) as error:
                record["error"] = str(error)
            finally:
                record["response_sha256"] = hashlib.sha256(body).hexdigest()
                record["response_ids"] = response_ids(body)
                if raw_directory is not None:
                    filename = f"{index:04d}-response.bin"
                    (raw_directory / filename).write_bytes(body)
                    record["response_file"] = filename

        do_GET = forward
        do_POST = forward
        do_DELETE = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", records
    finally:
        stop.set()
        with lock:
            owned_sockets = list(sockets)
        for connection in owned_sockets:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
        server.shutdown()
        server.server_close()  # Joins handlers after their owned sockets wake.
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("recording proxy listener did not stop")
