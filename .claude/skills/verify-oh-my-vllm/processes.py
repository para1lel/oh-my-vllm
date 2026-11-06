import contextlib
import ctypes
import os
import signal
import subprocess
import time
from pathlib import Path

from evidence import ROOT, digest, save_json, timestamp


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot become a child subreaper")


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record):
    current = process_identity(record["pid"])
    if current and current["start_ticks"] == record["start_ticks"]:
        return current
    return None


class Processes:
    def __init__(self, artifacts, environment):
        self.artifacts = artifacts
        self.environment = environment
        self.children = {}
        self.records = {}

    def observe(self):
        pending = [os.getpid()]
        seen = set()
        while pending:
            parent = pending.pop()
            if parent in seen:
                continue
            seen.add(parent)
            try:
                children = Path(f"/proc/{parent}/task/{parent}/children").read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            for child in map(int, children.split()):
                record = process_identity(child)
                if record is None:
                    continue
                key = f"{child}:{record['start_ticks']}"
                if key not in self.records:
                    self.records[key] = {**record, "first_seen_at": timestamp()}
                pending.append(child)
        return list(self.records.values())

    def start(self, label, command):
        directory = self.artifacts / "processes" / label
        directory.mkdir(parents=True)
        save_json(directory / "command.json", {"argv": command, "cwd": str(ROOT)})
        with (
            (directory / "stdout.txt").open("wb") as stdout,
            (directory / "stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        self.children[label] = process
        self.observe()
        identity = process_identity(process.pid)
        executable = Path(f"/proc/{process.pid}/exe")
        if identity is not None:
            try:
                identity["executable"] = os.readlink(executable)
                identity["executable_sha256"] = digest(executable)
            except (FileNotFoundError, ProcessLookupError):
                identity["executable_state"] = "exited before executable capture"
        save_json(directory / "identity.json", identity)
        return process

    def wait(self, label, timeout):
        process = self.children[label]
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            self.observe()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label} exceeded {timeout} seconds")
            time.sleep(0.05)
        self.observe()
        save_json(
            self.artifacts / "processes" / label / "exit.json",
            {"returncode": process.returncode, "observed_at": timestamp()},
        )
        if process.returncode:
            raise RuntimeError(f"{label} exited with {process.returncode}")
        return self.artifacts / "processes" / label

    def live(self):
        self.observe()
        return [
            current
            for record in self.records.values()
            if (current := same_process(record)) and current["state"] not in ("Z", "X")
        ]

    def reap(self):
        for process in self.children.values():
            process.poll()
        primary = {process.pid for process in self.children.values()}
        for record in self.records.values():
            if record["pid"] in primary:
                continue
            current = same_process(record)
            if current and current["state"] == "Z" and current["ppid"] == os.getpid():
                with contextlib.suppress(ChildProcessError):
                    pid, status = os.waitpid(record["pid"], os.WNOHANG)
                    if pid:
                        record["adopted_wait_status"] = status

    def drain(self, seconds):
        deadline = time.monotonic() + seconds
        while self.live() and time.monotonic() < deadline:
            self.reap()
            time.sleep(0.05)
        self.reap()

    def signal_owned(self, sig):
        self.observe()
        for record in self.records.values():
            current = same_process(record)
            if current and current["state"] not in ("Z", "X"):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(record["pid"], sig)

    def cleanup(self):
        self.observe()
        for process in self.children.values():
            if process.poll() is None:
                process.terminate()
        self.drain(6)
        if self.live():
            self.signal_owned(signal.SIGTERM)
            self.drain(2)
        if self.live():
            self.signal_owned(signal.SIGKILL)
            self.drain(2)
        for label, process in self.children.items():
            save_json(
                self.artifacts / "processes" / label / "exit.json",
                {"returncode": process.poll(), "observed_at": timestamp()},
            )
        records = [
            {**record, "final_identity": same_process(record)}
            for record in self.records.values()
        ]
        live = self.live()
        save_json(
            self.artifacts / "cleanup.json",
            {"processes": records, "live_processes": live, "finished_at": timestamp()},
        )
        if live:
            raise RuntimeError("owned processes are still alive; scratch was kept")
