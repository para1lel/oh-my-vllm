"""Loopback HTTP framing shared by the guest lane process and the host bridge.

OMP and its OTLP exporter need absolute HTTP URLs, and a sandbox with no network
has no route to the host collector. This module supplies both halves of the
mounted-socket transport: the guest loopback endpoints that forward their
original bodies to a Unix socket, and the framing helpers the host AF_UNIX
bridge reuses. It imports only the standard library so the image can copy the
single file.

Every hop records one bounded journal line per request. A failure line keeps the
original error, its direction, the body digest, and the timestamps, so a model
call that never reaches the recorded proxy stays visible.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 8 << 20
MAX_HEADER_BYTES = 64 << 10
MAX_TRAILER_LINES = 64
RELAY_BYTES = 8192
CONNECT_TIMEOUT = 10.0
MAX_JOURNAL_ENTRIES = 4096
MAX_JOURNAL_BYTES = 64 << 20
HANDLER_RELEASE_SECONDS = 5.0
# The accept loop polls its stop event between accepts. Stopping therefore
# needs no new thread, which keeps shutdown working under thread exhaustion.
ACCEPT_POLL_SECONDS = 0.25
CHUNK_SIZE = re.compile(rb"[0-9a-fA-F]{1,16}")
FRAMING_DIRECTIONS = ("request_framing", "request_read", "route_rejection")


class FramingError(ValueError):
    """A rejected request body, with the direction that rejected it."""

    def __init__(self, message: str, direction: str = "request_framing"):
        super().__init__(message)
        self.direction = direction


class HopJournal:
    """One bounded append-only journal per hop; overflow stays visible."""

    def __init__(self, path: Path, hop: str):
        self.path, self.hop = path, hop
        self.lock = threading.Lock()
        self.count = 0
        self.bytes = 0
        self.error = None

    def write(self, record: dict):
        entry = {"hop": self.hop, "index": self.count, **record}
        raw = (json.dumps(entry, allow_nan=False) + "\n").encode()
        with self.lock:
            if self.error:
                return
            if (
                self.count >= MAX_JOURNAL_ENTRIES
                or self.bytes + len(raw) > MAX_JOURNAL_BYTES
            ):
                self.error = "hop journal exceeded its evidence bound"
                return
            try:
                with self.path.open("ab") as output:
                    output.write(raw)
            except OSError as error:
                self.error = f"hop journal write failed: {type(error).__name__}"
                return
            self.count += 1
            self.bytes += len(raw)


def journal_records(path: Path) -> list[dict]:
    """Read a journal that closed inside its bound."""
    if not path.exists():
        return []
    records = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as error:
            raise ValueError(f"hop journal line is not JSON: {path.name}") from error
    if len(records) > MAX_JOURNAL_ENTRIES:
        raise ValueError(f"hop journal exceeds its entries bound: {path.name}")
    return records


class UnixHTTPConnection(http.client.HTTPConnection):
    """Send one HTTP request over a mounted Unix socket."""

    def __init__(self, socket_path: str, timeout: float):
        super().__init__("mounted-socket", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(self.socket_path)
        except BaseException:
            connection.close()
            raise
        self.sock = connection


def read_request_body(handler: BaseHTTPRequestHandler, maximum: int) -> bytes:
    """Read one strictly framed request body inside the transport bound."""
    transfer = handler.headers.get("Transfer-Encoding")
    if transfer is not None and handler.headers.get("Content-Length") is not None:
        raise FramingError("request declares two conflicting framings")
    if transfer is not None:
        if transfer.lower() != "chunked":
            raise FramingError("request uses an unsupported transfer encoding")
        return read_chunked(handler.rfile, maximum)
    declared = handler.headers.get_all("Content-Length") or []
    if len({value.strip() for value in declared}) > 1:
        raise FramingError("request declares conflicting content lengths")
    size = int(declared[0]) if declared else 0
    if not 0 <= size <= maximum:
        raise FramingError("request body exceeds the transport bound")
    body = handler.rfile.read(size)
    if len(body) != size:
        raise FramingError("request body was truncated", "request_read")
    return body


def read_chunked(stream, maximum: int) -> bytes:
    """Decode bounded chunked framing and require an explicit terminator."""
    body = bytearray()
    while True:
        line = stream.readline(MAX_HEADER_BYTES + 1)
        if not line or len(line) > MAX_HEADER_BYTES or not line.endswith(b"\r\n"):
            raise FramingError("chunk header is missing, unterminated or oversized")
        token = line[:-2].split(b";", 1)[0].strip()
        if not CHUNK_SIZE.fullmatch(token):
            raise FramingError("chunk size is not a bounded hexadecimal number")
        size = int(token, 16)
        if size > maximum or len(body) + size > maximum:
            raise FramingError("chunk body exceeds the transport bound")
        if size == 0:
            read_trailer(stream)
            return bytes(body)
        chunk = stream.read(size)
        if len(chunk) != size:
            raise FramingError("chunk body was truncated", "request_read")
        body += chunk
        if stream.read(2) != b"\r\n":
            raise FramingError("chunk body has no terminator")


def read_trailer(stream):
    """Read a bounded trailer that must end with a bare terminator line."""
    for _ in range(MAX_TRAILER_LINES):
        line = stream.readline(MAX_HEADER_BYTES + 1)
        if not line or len(line) > MAX_HEADER_BYTES or not line.endswith(b"\r\n"):
            raise FramingError("chunk trailer is missing, unterminated or oversized")
        if line == b"\r\n":
            return
    raise FramingError("chunk trailer exceeds its line bound")


def relay(connection, handler: BaseHTTPRequestHandler, body: bytes, record: dict):
    """Forward one buffered request and relay the entity with close framing."""
    record["error_direction"] = "request_write"
    connection.connect()
    handler.server.track_upstream(connection)
    connection.putrequest(
        handler.command, handler.path, skip_host=True, skip_accept_encoding=True
    )
    for key, value in handler.headers.items():
        if key.lower() not in ("content-length", "transfer-encoding"):
            connection.putheader(key, value)
    connection.putheader("Content-Length", str(len(body)))
    connection.putheader("Connection", "close")
    connection.endheaders(body)
    record["upstream_dispatched"] = True
    record["upstream_started_unix_s"] = time.time()
    record["error_direction"] = "upstream_read"
    response = connection.getresponse()
    record["response_status"] = response.status
    handler.send_response(response.status)
    for key, value in response.getheaders():
        if key.lower() not in ("content-length", "transfer-encoding", "connection"):
            handler.send_header(key, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    digest, total = hashlib.sha256(), 0
    read = getattr(response, "read1", response.read)
    while True:
        record["error_direction"] = "upstream_read"
        chunk = read(RELAY_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        record["error_direction"] = "downstream_write"
        handler.wfile.write(chunk)
        handler.wfile.flush()
    record.update(response_bytes=total, response_sha256=digest.hexdigest())
    record.pop("error_direction", None)


class TrackedServer(ThreadingHTTPServer):
    """Serve one endpoint and release every tracked socket inside a bound."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, address, *, connect, timeout: float, journal: HopJournal):
        self.upstream_endpoint = None
        self.connect_upstream = connect
        self.read_timeout = timeout
        self.journal = journal
        self.serving = False
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._handler_sockets: set[socket.socket] = set()
        self._handler_threads: set[threading.Thread] = set()
        super().__init__(address, FramingHandler)
        # A bounded poll interval lets run() notice a stop without shutdown().
        self.timeout = ACCEPT_POLL_SECONDS
        self.stop_event = threading.Event()

    def run(self):
        """Serve one accept at a time until the stop event is set."""
        try:
            while not self.stop_event.is_set():
                self.handle_request()
        except BaseException as error:  # pragma: no cover - listener failure
            self.errors.append(
                f"lane listener loop failed: {type(error).__name__}: {error}"
            )
            self.stop_event.set()

    def upstream(self):
        connection = self.connect_upstream()
        if self.upstream_endpoint is None and isinstance(
            connection, http.client.HTTPConnection
        ):
            self.upstream_endpoint = (connection.host, connection.port)
        return connection

    def upstream_timeout(self) -> float:
        return self.read_timeout

    def body_limit(self) -> int:
        return MAX_BODY_BYTES

    def register_client(self, connection):
        with self._lock:
            self._handler_sockets.add(connection)
            self._handler_threads.add(threading.current_thread())

    def release_client(self, connection):
        with self._lock:
            self._handler_sockets.discard(connection)
            self._handler_threads.discard(threading.current_thread())

    def track_upstream(self, connection):
        """Register the exact upstream socket owner for cancellation.

        ``HTTPConnection.getresponse()`` detaches ``connection.sock`` under
        close framing, which both production hops use, so the registered object
        is kept on the connection until this request releases it.
        """
        if connection.sock is not None:
            connection.lane_upstream_socket = connection.sock
            with self._lock:
                self._handler_sockets.add(connection.sock)

    def release_upstream(self, connection):
        """Release the exact upstream socket owner after its request ends."""
        sock = getattr(connection, "lane_upstream_socket", None)
        if sock is not None:
            connection.lane_upstream_socket = None
            with self._lock:
                self._handler_sockets.discard(sock)

    def tracked_sockets(self) -> int:
        """Live tracked sockets; it must return to zero after each request."""
        with self._lock:
            return len(self._handler_sockets)

    def record(self, record: dict):
        self.journal.write(record)
        if self.journal.error:
            self.errors.append(self.journal.error)

    def handler_count(self) -> int:
        """Live handler threads; release waits for these inside its bound."""
        with self._lock:
            return len(self._handler_threads)

    def abort(self):
        """Wake every handler blocked on a client or upstream socket."""
        with self._lock:
            sockets = list(self._handler_sockets)
        for connection in sockets:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

    def release(self, timeout: float = HANDLER_RELEASE_SECONDS) -> bool:
        """Join handler threads inside the bound; report any that stayed alive."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                pending = [
                    thread
                    for thread in self._handler_threads
                    if thread is not threading.current_thread() and thread.is_alive()
                ]
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            pending[0].join(timeout=min(0.25, remaining))


class FramingHandler(BaseHTTPRequestHandler):
    """One request per connection; the server supplies the upstream connection."""

    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.upstream_timeout())
        self.server.register_client(self.connection)

    def finish(self):
        try:
            super().finish()
        finally:
            self.server.release_client(self.connection)

    def log_message(self, format, *args):
        pass

    def base_record(self) -> dict:
        return {
            "method": self.command,
            "path": self.path,
            "received_unix_s": time.time(),
            "received_monotonic_s": time.monotonic(),
            "upstream_dispatched": False,
        }

    def respond(self, status: int, message: str):
        with suppress(OSError):
            self.send_error(status, message)

    def reject_route(self):
        self.close_connection = True
        record = self.base_record()
        record.update(
            error="lane transport accepts POST only",
            error_kind="route_rejection",
            error_direction="route_rejection",
            completed_unix_s=time.time(),
        )
        self.server.record(record)
        self.respond(405, "lane transport accepts POST only")

    do_GET = do_HEAD = reject_route
    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = reject_route

    def do_POST(self):
        self.close_connection = True
        record, connection = self.base_record(), None
        try:
            body = read_request_body(self, self.server.body_limit())
            record.update(
                body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest()
            )
            connection = self.server.upstream()
            relay(connection, self, body, record)
        except Exception as error:
            record.setdefault("error", f"{type(error).__name__}: {error}")
            record.setdefault("error_kind", type(error).__name__)
            record.setdefault("error_direction", "upstream_connect")
            if isinstance(error, FramingError):
                record["error_direction"] = error.direction
            if record.get("response_status") is None:
                self.respond(502, "lane transport forwarding failed")
        finally:
            if connection is not None:
                self.server.release_upstream(connection)
                with suppress(OSError):
                    connection.close()
            record["completed_unix_s"] = time.time()
            record["completed_monotonic_s"] = time.monotonic()
            self.server.record(record)


def loopback_server(
    port: int,
    socket_path: str,
    timeout: float,
    journal: HopJournal,
) -> TrackedServer:
    """Expose one guest loopback endpoint that forwards to a mounted socket."""
    server = TrackedServer(
        ("127.0.0.1", port),
        connect=lambda: UnixHTTPConnection(socket_path, timeout),
        timeout=timeout,
        journal=journal,
    )
    if server.server_address[0] != "127.0.0.1":
        server.server_close()
        raise ValueError("lane endpoint must bind guest loopback only")
    return server


def stop_server(server, bound: float = HANDLER_RELEASE_SECONDS) -> bool:
    """Attempt every listener owner inside the bound and never raise.

    The accept loop polls ``server.stop_event``, so stopping needs no new
    thread: under thread exhaustion the original start failure stays
    authoritative instead of being replaced by a cleanup allocation error.
    """
    complete = True
    server.serving = False
    server.stop_event.set()
    try:
        server.abort()
    except BaseException as error:
        complete = False
        server.errors.append(
            f"lane listener abort failed: {type(error).__name__}: {error}"
        )
    try:
        server.server_close()
    except BaseException as error:
        complete = False
        server.errors.append(
            f"lane listener close failed: {type(error).__name__}: {error}"
        )
    try:
        released = server.release(bound)
    except BaseException as error:
        released = False
        server.errors.append(
            f"lane handler release failed: {type(error).__name__}: {error}"
        )
    if not released:
        complete = False
        server.errors.append("lane handler release exceeded its bound")
    thread = server.thread
    if thread is not None and thread is not threading.current_thread():
        try:
            thread.join(timeout=bound)
        except BaseException as error:
            complete = False
            server.errors.append(
                f"lane listener join failed: {type(error).__name__}: {error}"
            )
        else:
            if thread.is_alive():
                complete = False
                server.errors.append("lane listener thread stayed alive")
    return complete


def serve(*servers) -> list[threading.Thread]:
    """Start every listener; only a real start counts as serving."""
    started = []
    for server in servers:
        thread = threading.Thread(target=server.run, daemon=True)
        try:
            thread.start()
        except BaseException as error:
            # An unstarted listener never accepts: its socket is closed without
            # any shutdown handshake, and every partial owner is still released.
            error.add_note(f"lane listener start failed: {server.server_address}")
            for candidate in servers:
                if not stop_server(candidate):
                    error.add_note(
                        f"lane listener cleanup stayed incomplete: "
                        f"{candidate.server_address}"
                    )
            raise
        server.serving, server.thread = True, thread
        started.append(thread)
    return started


def close_servers(
    servers, threads=None, *, bound: float = HANDLER_RELEASE_SECONDS
) -> bool:
    """Stop listeners, wake every socket, and join handlers inside the bound."""
    complete = True
    for server in servers:
        if not stop_server(server, bound):
            complete = False
    for thread in threads or []:
        if thread is threading.current_thread():
            continue
        thread.join(timeout=bound)
        if thread.is_alive():
            complete = False
    return complete


def child_exit(command: list[str], servers, threads, marker: Path | None) -> int:
    """Run the guest client with inherited streams and forward termination."""
    process = subprocess.Popen(command)

    def stop(_signum, _frame):
        if process.poll() is None:
            process.terminate()

    handlers = {
        number: signal.signal(number, stop)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        try:
            returncode = process.wait(timeout=None)
        except KeyboardInterrupt:
            process.terminate()
            returncode = process.wait(timeout=CONNECT_TIMEOUT)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    released = close_servers(servers, threads)
    if marker is not None:
        report = {
            "released": released and all(not server.errors for server in servers),
            "returncode": returncode,
            "servers": [
                {
                    "errors": server.errors,
                    "journal_entries": server.journal.count,
                    "journal_error": server.journal.error,
                }
                for server in servers
            ],
        }
        with suppress(OSError):
            marker.write_text(json.dumps(report) + "\n")
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-socket", required=True)
    parser.add_argument("--telemetry-socket")
    parser.add_argument("--model-port", type=int, default=18080)
    parser.add_argument("--telemetry-port", type=int, default=18081)
    parser.add_argument("--read-timeout", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = [item for item in args.command if item != "--"]
    if not command:
        parser.error("supply the guest client command after --")
    if not (args.read_timeout > 0):
        parser.error("guest read timeout must be positive")
    listeners = [(args.model_socket, args.model_port, "model")]
    if args.telemetry_socket:
        listeners.append((args.telemetry_socket, args.telemetry_port, "telemetry"))
    servers, threads = [], []
    try:
        for path, port, name in listeners:
            if not (0 < port <= 65535):
                parser.error(f"guest listener port is out of range: {port}")
            if not Path(path).is_socket():
                parser.error(f"mounted transport socket is missing: {path}")
            journal = HopJournal(
                Path(path).parent / f"guest-{name}.jsonl", f"guest-{name}"
            )
            servers.append(loopback_server(port, path, args.read_timeout, journal))
        threads = serve(*servers)
    except BaseException as error:
        if not close_servers(servers, threads):
            error.add_note("lane guest listener cleanup stayed incomplete")
        raise
    marker = Path(args.model_socket).parent / "guest-close.json"
    return child_exit(command, servers, threads, marker)


if __name__ == "__main__":
    sys.exit(main())
