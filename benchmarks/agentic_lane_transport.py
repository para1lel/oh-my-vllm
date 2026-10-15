"""Owned AF_UNIX bridges from a sealed sandbox lane to host loopback collectors.

Each lane mounts only its socket directory. The guest runs loopback endpoints
that forward original HTTP bodies to these sockets; the bridge forwards them to
the collector endpoints on host loopback. Every listener is AF_UNIX bound to a
task-owned path, both upstreams must be local plain HTTP origins, and closure
wakes every tracked socket, joins handlers inside a bound, and removes the
sockets only after verified release.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import socketserver
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from benchmarks.agentic_lane_forwarder import (
    FRAMING_DIRECTIONS,
    HANDLER_RELEASE_SECONDS,
    MAX_BODY_BYTES,
    HopJournal,
    TrackedServer,
    close_servers,
    journal_records,
    serve,
)

GUEST_MODEL_PORT = 18080
GUEST_TELEMETRY_PORT = 18081
MODEL_SOCKET = "model.sock"
TELEMETRY_SOCKET = "telemetry.sock"
GUEST_FORWARDER = "/usr/local/bin/lane-forward.py"
GUEST_PYTHON = "/usr/bin/python3"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
# Canonical client-disconnect errors of a deadline-aborted request. A hop may
# observe either one, and its attempt must also complete inside the window the
# recorded abort and disconnect prove, so one matching digest is never
# sufficient on its own.
DEADLINE_ATTEMPT_KINDS = ("BrokenPipeError", "ConnectionResetError")


def upstream_origin(url: str) -> tuple[str, int, str]:
    """Split one local plain HTTP collector base into origin and base path."""
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in LOOPBACK:
        raise ValueError("lane bridge upstream must be a local plain HTTP service")
    if not parts.port or parts.query or parts.fragment:
        raise ValueError("lane bridge upstream must name its port")
    return parts.hostname, parts.port, parts.path.rstrip("/")


class UnixBridgeServer(TrackedServer):
    """Serve HTTP framing on one mounted AF_UNIX socket, never on a network port."""

    address_family = socket.AF_UNIX
    allow_reuse_address = False

    def __init__(self, path: Path, host: str, port: int, timeout: float, journal):
        super().__init__(
            str(path),
            connect=lambda: http.client.HTTPConnection(host, port, timeout=timeout),
            timeout=timeout,
            journal=journal,
        )
        self.upstream_endpoint = (host, port)

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "lane-bridge", 0

    def body_limit(self) -> int:
        return MAX_BODY_BYTES


class LaneBridge:
    """Own one lane's socket directory, listeners and mounted guest ports."""

    def __init__(
        self,
        directory: Path,
        model_upstream: str,
        *,
        telemetry_upstream: str | None = None,
        read_timeout: float,
        model_port: int = GUEST_MODEL_PORT,
        telemetry_port: int = GUEST_TELEMETRY_PORT,
        release_seconds: float = HANDLER_RELEASE_SECONDS,
    ):
        if not (read_timeout > 0):
            raise ValueError("lane bridge read timeout must be positive")
        if not (0 < model_port <= 65535 and 0 < telemetry_port <= 65535):
            raise ValueError("lane bridge guest ports are out of range")
        self.directory, self.read_timeout = directory, read_timeout
        self.model_port, self.telemetry_port = model_port, telemetry_port
        self.release_seconds = release_seconds
        self.errors: list[str] = []
        self.closed = False
        self.threads: list = []
        self.servers: list[UnixBridgeServer] = []
        self.model = self.telemetry = None
        self.model_base_path = ""
        directory.mkdir(parents=True, exist_ok=False)
        self.model_socket = directory / MODEL_SOCKET
        self.telemetry_socket = directory / TELEMETRY_SOCKET
        try:
            self.model = self._serve(self.model_socket, model_upstream, "model")
            self.model_base_path = upstream_origin(model_upstream)[2]
            if telemetry_upstream is not None:
                self.telemetry = self._serve(
                    self.telemetry_socket, telemetry_upstream, "telemetry"
                )
            self.threads = serve(*self.servers)
        except BaseException as error:
            for note in self.rollback():
                error.add_note(note)
            raise

    def rollback(self) -> list[str]:
        """Attempt every owner cleanup and report what stayed incomplete."""
        try:
            self.close(raise_errors=False)
        except BaseException as error:  # pragma: no cover - defensive accounting
            self.errors.append(f"rollback raised: {type(error).__name__}: {error}")
        notes = [f"lane bridge rollback: {entry}" for entry in self.errors]
        for path in (self.model_socket, self.telemetry_socket):
            if path.exists():
                notes.append(f"lane bridge rollback left {path.name}")
        return notes

    def _serve(self, path: Path, upstream: str, name: str) -> UnixBridgeServer:
        host, port, _base = upstream_origin(upstream)
        journal = HopJournal(self.directory / f"{name}.jsonl", f"bridge-{name}")
        server = UnixBridgeServer(path, host, port, self.read_timeout, journal)
        self.servers.append(server)
        return server

    def guest_model_base(self) -> str:
        """Address the guest loopback listener that forwards to the model socket."""
        return f"http://127.0.0.1:{self.model_port}{self.model_base_path}"

    def guest_endpoint(self, url: str) -> str:
        """Move one collector endpoint path onto the guest telemetry listener."""
        if self.telemetry is None:
            raise ValueError("lane has no telemetry socket for this endpoint")
        parts = urlsplit(url)
        if not parts.port:
            raise ValueError("collector endpoint must name its port")
        return urlunsplit(parts._replace(netloc=f"127.0.0.1:{self.telemetry_port}"))

    def mounts(self) -> list[str]:
        """Mount only this socket directory; the lane root is mounted separately."""
        return [f"type=bind,source={self.directory},target={self.directory}"]

    def entry_point(self, argv: list[str]) -> list[str]:
        """Wrap the guest client so its loopback endpoints exist before it runs."""
        command = [
            GUEST_PYTHON,
            GUEST_FORWARDER,
            "--model-socket",
            str(self.model_socket),
            "--model-port",
            str(self.model_port),
        ]
        if self.telemetry is not None:
            command.extend(
                (
                    "--telemetry-socket",
                    str(self.telemetry_socket),
                    "--telemetry-port",
                    str(self.telemetry_port),
                )
            )
        command.extend(("--read-timeout", str(self.read_timeout), "--", *argv))
        return command

    def close(self, *, raise_errors: bool = True):
        if self.closed:
            return
        self.closed = True
        try:
            released = close_servers(
                self.servers, self.threads, bound=self.release_seconds
            )
        except BaseException as error:  # pragma: no cover - defensive accounting
            released = False
            self.errors.append(f"listener shutdown: {type(error).__name__}: {error}")
        for server in self.servers:
            self.errors.extend(server.errors)
            if server.journal.error:
                self.errors.append(server.journal.error)
        if not released:
            self.errors.append("lane bridge handler release exceeded its bound")
        removed = []
        if released:
            for path in (self.model_socket, self.telemetry_socket):
                if not path.exists():
                    continue
                try:
                    path.unlink()
                    removed.append(path.name)
                except OSError as error:
                    self.errors.append(
                        f"socket removal: {type(error).__name__}: {error}"
                    )
        (self.directory / "bridge.json").write_text(
            json.dumps(
                {
                    "directory": str(self.directory),
                    "model_upstream": (
                        self.model.upstream_endpoint if self.model else None
                    ),
                    "telemetry_upstream": (
                        self.telemetry.upstream_endpoint if self.telemetry else None
                    ),
                    "guest_model_port": self.model_port,
                    "guest_telemetry_port": (
                        self.telemetry_port if self.telemetry else None
                    ),
                    "read_timeout_s": self.read_timeout,
                    "handlers_released": released,
                    "removed_sockets": removed,
                    "remaining_sockets": [
                        path.name
                        for path in (self.model_socket, self.telemetry_socket)
                        if path.exists()
                    ],
                    "journals": {
                        server.journal.hop: {
                            "entries": server.journal.count,
                            "error": server.journal.error,
                        }
                        for server in self.servers
                    },
                    "closed": bool(released and not self.errors),
                    "errors": self.errors,
                    "at_unix_s": time.time(),
                }
            )
            + "\n"
        )
        if raise_errors and self.errors:
            raise ValueError("lane bridge cleanup failed: " + "; ".join(self.errors))


def recorded_bodies(paths) -> Counter:
    """Multiset of raw body digests the collector already recorded."""
    return Counter(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
    )


def original_bodies(directory: Path) -> dict[str, Counter]:
    """Original request and export digests per hop; a missing file means none."""
    wire = directory / "wire"
    actions = []
    if (wire / "http.jsonl").exists():
        actions = [
            json.loads(line)
            for line in (wire / "http.jsonl").read_text().splitlines()
            if line.strip()
        ]
    return {
        "model": recorded_bodies(wire / record["request_file"] for record in actions),
        "telemetry": recorded_bodies((directory / "sdk-trace").glob("*.pb")),
    }


def hop_entries(path: Path, reported: int) -> list[dict]:
    """Read one configured hop journal and require its recorded closure count."""
    entries = journal_records(path)
    if len(entries) != reported:
        raise ValueError(
            f"hop journal count differs from its closure record: {path.name}"
        )
    return entries


def deadline_attempt_bound(proof: dict) -> dict:
    """Canonical attempt bound of one recorded deadline abort.

    A hop attempt may only be explained by the original request it relays, so
    the bound keeps that request's own client-disconnect direction, its
    canonical disconnect errors, and the window its release must fall into. The
    window starts at the recorded abort action, not at an arbitrary time before
    the deadline, and ends at the recorded client disconnect, so a failure that
    completed before the abort or after the disconnect is a different attempt. A
    digest alone therefore explains nothing.
    """
    direction = proof.get("disconnect_direction")
    if direction != "downstream_write":
        raise ValueError("deadline attempt bound needs a canonical disconnect")
    deadline = float(proof["deadline_monotonic_s"])
    abort = float(proof["abort"]["start"]["monotonic_s"])
    disconnected = float(proof["disconnect_monotonic_s"])
    if not deadline <= abort <= disconnected:
        raise ValueError("deadline attempt bound has reversed abort clocks")
    return {
        "reason": f"collection deadline abort: {proof['kind']}",
        "response_id": proof["response_id"],
        "direction": direction,
        "kinds": list(DEADLINE_ATTEMPT_KINDS),
        "deadline_monotonic_s": deadline,
        "abort_monotonic_s": abort,
        "first_monotonic_s": abort,
        "last_monotonic_s": disconnected,
    }


def explanation_matches(entry: dict, bound: dict) -> bool:
    """Whether one failed attempt is the explained original's own attempt."""
    if entry.get("error_direction") != bound["direction"]:
        return False
    if entry.get("error_kind") not in bound["kinds"]:
        return False
    received, completed = (
        entry.get("received_monotonic_s"),
        entry.get("completed_monotonic_s"),
    )
    if type(received) not in (int, float) or type(completed) not in (int, float):
        return False
    return (
        received <= bound["deadline_monotonic_s"]
        and bound["first_monotonic_s"] <= completed <= bound["last_monotonic_s"]
    )


def bound_explanations(explained) -> dict[str, list[dict]]:
    """Validate one hop's explanations: one bound per recorded original."""
    bounds: dict[str, list[dict]] = {}
    for digest, values in (explained or {}).items():
        if not isinstance(digest, str) or not isinstance(values, list) or not values:
            raise ValueError("lane transport explanation needs a digest and a bound")
        for bound in values:
            if not isinstance(bound, dict) or not {
                "direction",
                "kinds",
                "deadline_monotonic_s",
                "first_monotonic_s",
                "last_monotonic_s",
                "reason",
            } <= set(bound):
                raise ValueError("lane transport explanation bound is incomplete")
            if bound["direction"] != "downstream_write":
                raise ValueError(
                    "lane transport explanation needs a canonical disconnect"
                )
        bounds[digest] = list(values)
    return bounds


def audit_hop(
    entries: list[dict],
    expected: Counter,
    explained: dict[str, list[dict]],
    *,
    hop: str,
) -> dict:
    """Bind every hop entry to one original body with exact multiplicity.

    Every attempt carrying a body digest must consume exactly one original
    request the collector recorded, and every failed attempt must additionally
    consume one not-yet-used explanation whose canonical direction, error and
    completion window match it. Two identical attempts therefore cannot hide
    behind one original, one success plus one explained failure cannot cover
    one request, and a failed retry stays visible until its own evidence
    explains it.
    """
    reached, aborted, rejections, unexplained = Counter(), Counter(), 0, []
    pending = {digest: list(bounds) for digest, bounds in explained.items()}
    used = Counter()
    for entry in entries:
        digest = entry.get("body_sha256")
        if entry.get("upstream_dispatched") and not entry.get("error") and digest:
            reached[digest] += 1
            continue
        if (
            not entry.get("upstream_dispatched")
            and entry.get("error_direction") in FRAMING_DIRECTIONS
        ):
            rejections += 1
            continue
        bound = next(
            (
                candidate
                for candidate in pending.get(digest, ())
                if explanation_matches(entry, candidate)
            ),
            None,
        )
        if bound is not None:
            pending[digest].remove(bound)
            used[digest] += 1
            aborted[digest] += 1
            continue
        unexplained.append(entry)
    if unexplained:
        first = unexplained[0]
        raise ValueError(
            f"unexplained {hop} transport failure: "
            f"{first.get('path')} {first.get('error')}"
        )
    unbound = sorted(digest for digest in reached if digest not in expected)
    if unbound:
        raise ValueError(
            f"{hop} hop relays a body that is not among the original requests"
        )
    inflated = sorted(
        digest for digest in reached if reached[digest] > expected[digest]
    )
    if inflated:
        raise ValueError(
            f"{hop} hop relays more attempts than the collector recorded for "
            f"{len(inflated)} request(s)"
        )
    excessive = sorted(
        digest
        for digest in aborted
        if reached[digest] + aborted[digest] > expected.get(digest, 0)
    )
    if excessive:
        raise ValueError(
            f"{hop} hop explains more attempts than the collector recorded for "
            f"{len(excessive)} request(s)"
        )
    missing = sorted(
        digest
        for digest in expected
        if reached[digest] + aborted[digest] < expected[digest]
    )
    if missing:
        raise ValueError(
            f"{hop} hop is missing a recorded attempt for {len(missing)} request(s)"
        )
    return {
        "hop": hop,
        "reached_records": sum(reached.values()),
        "explained_aborts": sum(aborted.values()),
        "explained_rejections": rejections,
        "consumed_explanations": dict(used),
    }


def bridge_closure(bridge: dict, configured: tuple[str, ...]) -> dict[str, int]:
    """Require a closed bridge with one journal record per configured hop."""
    if bridge.get("closed") is not True:
        raise ValueError("lane bridge did not close cleanly")
    if bridge.get("handlers_released") is not True:
        raise ValueError("lane bridge handlers were not released")
    if bridge.get("remaining_sockets"):
        raise ValueError("lane bridge left mounted sockets behind")
    if bridge.get("errors"):
        raise ValueError(f"lane bridge reported cleanup errors: {bridge['errors']}")
    journals = bridge.get("journals")
    if not isinstance(journals, dict) or set(journals) != {
        f"bridge-{name}" for name in configured
    }:
        raise ValueError("lane bridge journal records differ from its configuration")
    counts = {}
    for name in configured:
        record = journals[f"bridge-{name}"]
        if record.get("error"):
            raise ValueError(f"lane bridge {name} journal failed: {record['error']}")
        if type(record.get("entries")) is not int or record["entries"] < 0:
            raise ValueError(f"lane bridge {name} journal count is invalid")
        counts[name] = record["entries"]
    return counts


def guest_closure(guest: dict, configured: tuple[str, ...]) -> dict[str, int]:
    """Require released guest endpoints that match the configured listeners."""
    if guest.get("released") is not True:
        raise ValueError("guest lane endpoints were not released")
    servers = guest.get("servers")
    if not isinstance(servers, list) or len(servers) != len(configured):
        raise ValueError("guest lane endpoint count differs from its configuration")
    counts = {}
    for name, server in zip(configured, servers, strict=True):
        if server.get("errors"):
            raise ValueError(f"guest lane endpoint reported errors: {server['errors']}")
        if server.get("journal_error"):
            raise ValueError(f"guest lane journal failed: {server['journal_error']}")
        if type(server.get("journal_entries")) is not int:
            raise ValueError("guest lane journal count is invalid")
        counts[name] = server["journal_entries"]
    return counts


def lane_transport_evidence(directory: Path, *, explained: dict | None = None):
    """Audit one lane's mounted hops against the recorded collector evidence.

    ``explained`` maps one hop name to its recorded request digests and the
    canonical attempt bounds their originals produced, for example a native goal
    deadline cancellation on the model hop. Each failed attempt must consume one
    of those bounds, each bound is reported as unused when no hop recorded a
    matching failure, and a completed hop stays legitimate, so asymmetric hop
    outcomes are allowed while an unexplained failure and a hop that never
    relayed a recorded request are both rejected.
    """
    transport = directory / "transport"
    bridge = json.loads((transport / "bridge.json").read_text())
    configured = (
        ("model", "telemetry") if bridge.get("telemetry_upstream") else ("model",)
    )
    explained = {
        name: bound_explanations(values)
        for name, values in (explained or {}).items()
        if values
    }
    if set(explained) - set(configured):
        raise ValueError("lane transport explanation names an unconfigured hop")
    counts = bridge_closure(bridge, configured)
    guest_counts = guest_closure(
        json.loads((transport / "guest-close.json").read_text()), configured
    )
    expected = original_bodies(directory)
    result, consumed, unused = {}, {}, []
    for name in configured:
        for prefix, side, reported in (
            ("", "bridge", counts),
            ("guest-", "guest", guest_counts),
        ):
            entries = hop_entries(transport / f"{prefix}{name}.jsonl", reported[name])
            audit = audit_hop(
                entries,
                expected[name],
                explained.get(name, {}),
                hop=f"{side} {name}",
            )
            for digest, used in audit.pop("consumed_explanations").items():
                consumed[(name, digest)] = consumed.get((name, digest), 0) + used
            result[f"{side}-{name}"] = audit
    for name, values in explained.items():
        for digest, bounds in values.items():
            # A bound is consumed by the hop attempts that match it. It stays
            # legitimate for the hops to have relayed the request successfully
            # while the client aborted downstream of them, so an unused bound is
            # reported as evidence rather than rejected; a hop that never
            # relayed the request still fails the exact coverage check above.
            if consumed.get((name, digest), 0) < len(bounds):
                unused.append(f"{name}:{digest}")
    if "telemetry" not in configured and any(
        (transport / f"{name}.jsonl").exists()
        for name in ("telemetry", "guest-telemetry")
    ):
        raise ValueError("lane transport recorded an unconfigured telemetry hop")
    result["bridge_errors"] = bridge.get("errors")
    result["guest_errors"] = []
    result["unused_explanations"] = sorted(unused)
    result["explained_aborts"] = {
        name: sorted(values) for name, values in explained.items()
    }
    return result
