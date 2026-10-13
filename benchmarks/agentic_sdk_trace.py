"""Collect genuine OMP SDK result spans without changing provider dispatch."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
import uuid
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MAX_EXPORT_BYTES = 8 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_EXPORTS = 8192
MAX_SPANS = 20000


def read_export(stream, headers) -> bytes:
    """The pinned Node OTLP exporter uses chunked HTTP with no content length."""
    transfer, length = headers.get("Transfer-Encoding"), headers.get("Content-Length")
    if transfer:
        if transfer.lower() != "chunked" or length is not None:
            raise ValueError("ambiguous SDK export framing")
        chunks, total = [], 0
        for _ in range(8192):
            line = stream.readline(128)
            if not line.endswith(b"\r\n") or len(line) >= 128:
                raise ValueError("invalid SDK export chunk header")
            size = int(line.strip(), 16)
            if size == 0:
                if stream.read(2) != b"\r\n" or not total:
                    raise ValueError("invalid SDK export end")
                return b"".join(chunks)
            total += size
            if size < 0 or total > MAX_EXPORT_BYTES:
                raise ValueError("SDK export size exceeds evidence limit")
            chunk = stream.read(size)
            if len(chunk) != size or stream.read(2) != b"\r\n":
                raise ValueError("truncated SDK export chunk")
            chunks.append(chunk)
        raise ValueError("SDK export chunk count exceeds evidence limit")
    size = int(length or "0")
    if not 0 < size <= MAX_EXPORT_BYTES:
        raise ValueError("SDK export size exceeds evidence limit")
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("truncated SDK export")
    return raw


def sdk_spans(raw: bytes) -> list[dict]:
    """Decode native OTLP fields. Input usage already includes cache tokens."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest.FromString(raw)
    results = []
    for resource in request.resource_spans:
        for scope in resource.scope_spans:
            for span in scope.spans:
                fields = {}
                for item in span.attributes:
                    if item.key in fields:
                        raise ValueError("duplicate SDK trace attribute")
                    value = MessageToDict(item.value)
                    if "stringValue" in value:
                        value = value["stringValue"]
                    elif "intValue" in value:
                        value = int(value["intValue"])
                    elif "arrayValue" in value:
                        value = [
                            item.get("stringValue")
                            for item in value["arrayValue"].get("values", [])
                        ]
                    fields[item.key] = value
                if fields.get("gen_ai.operation.name") != "chat":
                    continue
                if len(results) >= MAX_SPANS:
                    raise ValueError("SDK span count exceeds evidence limit")
                results.append(
                    {
                        "source": "omp_sdk_otlp",
                        "scope": scope.scope.name,
                        "trace_id": span.trace_id.hex(),
                        "span_id": span.span_id.hex(),
                        "started_unix_s": span.start_time_unix_nano / 1e9,
                        "completed_unix_s": span.end_time_unix_nano / 1e9,
                        "status_code": span.status.code,
                        "attributes": fields,
                    }
                )
    if len(results) > MAX_SPANS:
        raise ValueError("SDK span count exceeds evidence limit")
    return results


def read_sdk_spans(directory: Path) -> list[dict]:
    """Replay hashed original exports, with duplicate exporter retries retained."""
    results, identities, total = [], {}, 0
    closed = json.loads((directory / "closed.json").read_text())
    if (
        closed.get("closed") is not True
        or closed.get("error")
        or (directory / "error.json").exists()
    ):
        raise ValueError("SDK export collector did not close cleanly")
    records = (directory / "exports.jsonl").read_text().splitlines()
    if len(records) > MAX_EXPORTS:
        raise ValueError("SDK export count exceeds evidence limit")
    for index, line in enumerate(records):
        record = json.loads(line)
        filename = f"{index:04d}.pb"
        if record["file"] != filename:
            raise ValueError("SDK export path or sequence differs")
        path = directory / filename
        if path.stat().st_size > MAX_EXPORT_BYTES:
            raise ValueError("SDK export size exceeds evidence limit")
        raw = path.read_bytes()
        total += len(raw)
        if (
            len(raw) != record["bytes"]
            or len(raw) > MAX_EXPORT_BYTES
            or total > MAX_TOTAL_BYTES
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("SDK export size or hash differs")
        for span in sdk_spans(raw):
            identity = (span["trace_id"], span["span_id"])
            if identity in identities:
                if identities[identity] != span:
                    raise ValueError("SDK exporter retry changed a span")
                continue
            identities[identity] = span
            results.append(
                {**span, "export_file": filename, "export_sha256": record["sha256"]}
            )
            if len(results) > MAX_SPANS:
                raise ValueError("SDK span count exceeds evidence limit")
    if closed.get("exports") != len(records) or closed.get("bytes") != total:
        raise ValueError("SDK export closure counts differ")
    return results


class SdkTrace:
    """One owned loopback endpoint per fixed OMP process, bounded and fail closed."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(exist_ok=False)
        (directory / "exports.jsonl").touch(exist_ok=False)
        self.count, self.total, self.error = 0, 0, None
        self.closed = False
        self.received_ids = set()
        self.span_identities = {}
        self.condition = threading.Condition()
        route = "/" + uuid.uuid4().hex + "/v1/traces"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 5

            def setup(self):
                super().setup()

                def abort():
                    with suppress(OSError):
                        self.connection.shutdown(socket.SHUT_RDWR)

                self.deadline = threading.Timer(5, abort)
                self.deadline.daemon = True
                self.deadline.start()

            def finish(self):
                try:
                    super().finish()
                finally:
                    self.deadline.cancel()

            def log_message(self, *_):
                pass

            def do_POST(self):
                self.close_connection = True
                try:
                    self.connection.settimeout(5)
                    if (
                        owner.error
                        or self.path != route
                        or self.headers.get("Content-Type") != "application/x-protobuf"
                        or self.headers.get("Content-Encoding")
                        not in (None, "identity")
                        or owner.count >= MAX_EXPORTS
                    ):
                        raise ValueError(
                            "SDK export request exceeds protocol or evidence limits"
                        )
                    raw = read_export(self.rfile, self.headers)
                    size = len(raw)
                    if owner.total + size > MAX_TOTAL_BYTES:
                        raise ValueError("SDK export total exceeds evidence limit")
                    filename = f"{owner.count:04d}.pb"
                    (directory / filename).write_bytes(raw)
                    with (directory / "exports.jsonl").open("a") as journal:
                        journal.write(
                            json.dumps(
                                {
                                    "file": filename,
                                    "bytes": size,
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                    "received_unix_s": time.time(),
                                }
                            )
                            + "\n"
                        )
                    owner.count += 1
                    owner.total += size
                    spans = sdk_spans(raw)
                    with owner.condition:
                        for span in spans:
                            identity = (span["trace_id"], span["span_id"])
                            previous = owner.span_identities.get(identity)
                            if previous is not None and previous != span:
                                raise ValueError("SDK exporter retry changed a span")
                            if previous is None:
                                if len(owner.span_identities) >= MAX_SPANS:
                                    raise ValueError(
                                        "SDK span count exceeds evidence limit"
                                    )
                                owner.span_identities[identity] = span
                        owner.received_ids.update(
                            span["attributes"].get("gen_ai.response.id")
                            for span in spans
                            if span["attributes"].get("gen_ai.response.id")
                        )
                        owner.condition.notify_all()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-protobuf")
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                except Exception as error:
                    owner.error = f"{type(error).__name__}: {error}"
                    (directory / "error.json").write_text(
                        json.dumps(
                            {
                                "error": owner.error,
                                "headers": {
                                    key: self.headers.get(key)
                                    for key in (
                                        "Content-Type",
                                        "Content-Length",
                                        "Content-Encoding",
                                        "Transfer-Encoding",
                                    )
                                },
                                "expected_route": self.path == route,
                            }
                        )
                        + "\n"
                    )
                    with suppress(OSError):
                        self.send_error(400)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}{route}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def environment(self) -> dict[str, str]:
        return {
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": self.endpoint,
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION": "none",
            "OTEL_TRACES_EXPORTER": "otlp",
            "OTEL_TRACES_SAMPLER": "always_on",
            "OTEL_METRICS_EXPORTER": "none",
            "OTEL_LOGS_EXPORTER": "none",
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false",
            "OTEL_BSP_SCHEDULE_DELAY": "1000",
        }

    def wait_for(self, identities: set[str], timeout: float = 15):
        """Wait after active work, before RPC EOF can discard its export queue."""
        deadline = time.monotonic() + timeout
        with self.condition:
            while not identities <= self.received_ids:
                if self.error:
                    raise ValueError(self.error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("OMP did not export all completed SDK results")
                self.condition.wait(min(remaining, 0.1))

    def close(self, *, strict=True):
        if not self.closed:
            self.closed = True
            self.server.shutdown()
            self.thread.join(timeout=10)
            self.server.server_close()
            if self.thread.is_alive():
                raise ValueError("SDK trace collector did not stop")
            (self.directory / "closed.json").write_text(
                json.dumps(
                    {
                        "closed": True,
                        "exports": self.count,
                        "bytes": self.total,
                        "unique_sdk_spans": len(self.span_identities),
                        "endpoint": self.endpoint,
                        "error": self.error,
                    }
                )
                + "\n"
            )
        if strict and self.error:
            raise ValueError(self.error)
