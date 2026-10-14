"""Own one service tree in a dedicated Linux child-subreaper process."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.common import (
    ProcessOwnership,
    _child_identities,
    _child_subreaper,
    _children,
    _process_record,
    _stop_engine,
)


class IsolatedEngineOwnership(ProcessOwnership):
    """Only the sole spawned engine can supply children to this subreaper.

    This class is used only by the dedicated CLI process. It creates no other
    subprocesses and requires an empty child inventory before spawning the root.
    Kernel reparenting therefore proves new direct-child provenance, including
    descendants whose intermediate parents exit before the first observation.
    """

    def __init__(self, supervisor):
        self.supervisor = supervisor
        super().__init__(supervisor["pid"], adopt_orphans=True, baseline=set())
        os.close(self.pidfds.pop(self.key(self.root)))
        self.records.clear()
        self.sessions.clear()
        self.root, self.root_pid = None, None

    def bind_engine(self, process):
        self.root_pid = process.pid
        self.root = _process_record(process.pid)
        if self.root is None:
            raise RuntimeError("engine disappeared before identity lookup")
        if self.root["ppid"] != self.supervisor["pid"]:
            raise RuntimeError("engine is not a direct child of its supervisor")
        if (
            self._bind(
                process.pid, parent=self.supervisor, expected=self.key(self.root)
            )
            is None
        ):
            raise RuntimeError("engine disappeared before pidfd binding")

    def refresh(self):
        if self.root is not None:
            super().refresh()
        with self.lock:
            for pid in _children(self.collector_pid):
                record = _process_record(pid)
                if record is None or self.key(record) in self.records:
                    continue
                if record["start_ticks"] < self.supervisor["start_ticks"]:
                    raise RuntimeError("isolated supervisor has an older child")
                bound = self._bind(
                    pid, parent=self.supervisor, expected=self.key(record)
                )
                if bound is not None:
                    self.actions.append(
                        {
                            "pid": pid,
                            "start_ticks": bound["start_ticks"],
                            "adopted_supervisor": self.supervisor,
                            "monotonic_s": time.monotonic(),
                        }
                    )
            if self.root is not None:
                super().refresh()

    def unknown_adoptees(self):
        if self.root is not None:
            return super().unknown_adoptees()
        self.unverified_adopted = [
            record
            for pid in _children(self.collector_pid)
            if (record := _process_record(pid)) is not None
            and self.key(record) not in self.records
        ]
        return self.unverified_adopted


def _capture_error(errors, operation, error):
    errors.append({"operation": operation, "error": f"{type(error).__name__}: {error}"})


def _stop_bound_processes(process, ownership, errors):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if ownership is not None:
            try:
                ownership.refresh()
            except Exception as error:
                _capture_error(errors, "cleanup_refresh", error)
            for record in reversed(ownership.live_records()):
                try:
                    ownership.signal(record, signum)
                except Exception as error:
                    _capture_error(errors, "cleanup_signal", error)
        if process.poll() is None and (
            ownership is None
            or ownership.root is None
            or ownership.key(ownership.root) not in ownership.pidfds
        ):
            process.send_signal(signum)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            process.poll()
            if ownership is not None:
                ownership.reap_known()
                if not ownership.live_records():
                    break
            elif process.returncode is not None:
                break
            time.sleep(0.02)
    process.wait(timeout=2)


def _cleanup_engine(process, ownership, summary):
    errors = summary["errors"]
    if process is not None:
        try:
            if (
                ownership is None
                or ownership.root is None
                or ownership.key(ownership.root) not in ownership.pidfds
            ):
                _stop_bound_processes(process, ownership, errors)
            else:
                _stop_engine(process, ownership)
        except Exception as error:
            _capture_error(errors, "stop_engine", error)
            try:
                _stop_bound_processes(process, ownership, errors)
            except Exception as fallback_error:
                _capture_error(errors, "stop_bound_processes", fallback_error)
        summary["engine_returncode"] = process.returncode
    if ownership is not None:
        try:
            ownership.unknown_adoptees()
            summary["cleanup"] = ownership.snapshot()
        except Exception as error:
            _capture_error(errors, "snapshot", error)
        finally:
            try:
                ownership.close()
            except Exception as error:
                _capture_error(errors, "close_pidfds", error)


def _watch_engine(process, ownership, observations, stop):
    previous, count, maximum_interval = None, 0, 0.0
    control = bytearray()
    while True:
        now = time.monotonic()
        ownership.refresh()
        interval = None if previous is None else now - previous
        maximum_interval = max(maximum_interval, interval or 0)
        count += 1
        observations.write(
            json.dumps({"monotonic_s": now, "interval_s": interval}) + "\n"
        )
        observations.flush()
        previous = now
        if stop["signal"] is not None:
            cause = "controller_signal"
            break
        if process.poll() is not None:
            cause = "engine_exited"
            break
        if select.select([sys.stdin], [], [], 0.05)[0]:
            chunk = os.read(sys.stdin.fileno(), 32)
            control.extend(chunk)
            if len(control) > 5 or not b"stop\n".startswith(control):
                raise ValueError("supervisor accepts only the stop command")
            if control == b"stop\n":
                cause = "controller_stop"
                break
            if not chunk:
                cause = "controller_eof"
                break
    return {
        "stop_cause": cause,
        "stop_signal": stop["signal"],
        "observations": count,
        "maximum_interval_s": maximum_interval,
        "last_observed_monotonic_s": previous,
    }


def supervise(command, log_path, audit_path, *, natural_completion=False):
    supervisor = _process_record(os.getpid())
    summary = {
        "schema": 1,
        "supervisor": supervisor,
        "command": command,
        "ownership_scope": "fresh subreaper with one spawned engine and no other roots",
        "errors": [],
        "passed": False,
        "completion_mode": "natural_exit" if natural_completion else "controller_stop",
    }
    process, ownership = None, None
    stop = {"signal": None}

    def request_stop(signum, _frame):
        stop["signal"] = signum

    old_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with _child_subreaper():
            if _child_identities():
                raise RuntimeError("supervisor must start without existing children")
            ownership = IsolatedEngineOwnership(supervisor)
            with (
                log_path.open("x") as output,
                audit_path.with_suffix(".observations.jsonl").open("x") as observations,
            ):
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    ownership.bind_engine(process)
                    print(
                        json.dumps(
                            {
                                "event": "started",
                                "engine": ownership.root,
                                "supervisor": supervisor,
                            }
                        ),
                        flush=True,
                    )
                    summary["monitor"] = _watch_engine(
                        process, ownership, observations, stop
                    )
                except Exception as error:
                    _capture_error(summary["errors"], "monitor", error)
                finally:
                    _cleanup_engine(process, ownership, summary)
    except Exception as error:
        _capture_error(summary["errors"], "supervisor_scope", error)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        cleanup = summary.get("cleanup", {})
        summary["passed"] = (
            not summary["errors"]
            and summary.get("monitor", {}).get("stop_cause")
            == ("engine_exited" if natural_completion else "controller_stop")
            and (not natural_completion or summary.get("engine_returncode") == 0)
            and bool(cleanup)
            and not any(
                cleanup[name]
                for name in ("live", "unverified_adopted", "invalid_sessions")
            )
        )
        audit_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps({"event": "finished", "passed": summary["passed"]}), flush=True
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--natural-completion", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an engine command is required")
    if args.audit.exists():
        parser.error("the process audit already exists")
    result = supervise(
        command, args.log, args.audit, natural_completion=args.natural_completion
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
