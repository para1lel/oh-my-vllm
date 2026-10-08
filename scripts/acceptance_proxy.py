"""Record actual OMP HTTP payloads while forwarding to the real model service."""

import hashlib
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
def recording_proxy(upstream_url: str):
    """Own a loopback listener and retain complete upstream request evidence.

    HTTP/1.0 closes each downstream stream after its body. The upstream SSE
    content is forwarded without reconstruction, tool edits, or extra requests.
    """
    records, active = [], []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def forward(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            record = {"method": self.command, "path": self.path}
            if data:
                record["request"] = json.loads(data)
            with lock:
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
                    response = urllib.request.urlopen(request, timeout=120)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    with lock:
                        active.append(response)
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
                with lock:
                    if "response" in locals() and response in active:
                        active.remove(response)

        do_GET = forward
        do_POST = forward
        do_DELETE = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        with lock:
            responses = active.copy()
        for response in responses:
            response.close()
