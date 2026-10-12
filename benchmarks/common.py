"""Process ownership, runtime identity, and output parsing for GPU collectors."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_identity():
    """Describe the independent environment; reject accidental legacy execution."""
    from oh_my_vllm.worker.runtime import identity

    worker = Path(os.environ.get("OH_MY_VLLM_WORKER_PYTHON", sys.executable))
    if worker.resolve() != Path(sys.executable).resolve():
        raise RuntimeError("benchmark worker must use the current independent Python")
    return identity()


_PROCESS_RUN_LOCK = threading.RLock()


def _process_record(pid: int) -> dict | None:
    """A PID alone is not an identity. Keep its kernel birth time and ancestry."""
    try:
        data = Path("/proc", str(pid), "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = data[data.rfind(")") + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(f"cannot read process identity for {pid}")
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def _children(pid: int) -> set[int]:
    """Include children created by any thread, not just the thread-group leader."""
    result = set()
    for task in Path("/proc", str(pid), "task").glob("[0-9]*"):
        try:
            result.update(
                int(value) for value in (task / "children").read_text().split()
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def _open_pidfd(pid):
    """Some compatible Python builds omit the Linux os wrapper."""
    if callable(getattr(os, "pidfd_open", None)):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


def _child_identities():
    return {
        (record["pid"], record["start_ticks"])
        for pid in _children(os.getpid())
        if (record := _process_record(pid)) is not None
    }


class ProcessOwnership:
    """Track exact descendants across setsid, reparenting, and PID reuse.

    Ownership is established through live, revalidated parent links. A process
    never becomes owned just because it has the same UID or process group.
    pidfds address the recorded process even if its numeric PID is later reused.
    """

    def __init__(self, root, *, adopt_orphans=False, baseline=None):
        self.root_pid = root.pid if hasattr(root, "pid") else int(root)
        self.collector_pid = os.getpid()
        self.adopt_orphans = adopt_orphans
        self.sessions = {}
        self.invalid_sessions = set()
        self.baseline = _child_identities() if baseline is None else set(baseline)
        self.unverified_adopted = []
        self.lock = threading.RLock()
        self.records = {}
        self.pidfds = {}
        self.actions = []
        self.root = self._bind(self.root_pid)
        if self.root is None:
            raise RuntimeError("engine exited before its process identity was bound")

    @staticmethod
    def key(record):
        return record["pid"], record["start_ticks"]

    def _bind(self, pid, *, parent=None, adopted_session=None, expected=None):
        record = _process_record(pid)
        if record is None:
            return None
        key = self.key(record)
        if expected is not None and key != expected:
            return None
        existing = key in self.records
        try:
            fd = self.pidfds[key] if existing else _open_pidfd(pid)
        except ProcessLookupError:
            return None
        published = existing
        try:
            again = _process_record(pid)
            if (
                again is None
                or self.key(again) != key
                or (parent is not None and again["ppid"] != parent["pid"])
                or (
                    adopted_session is not None
                    and (
                        again["ppid"] != self.collector_pid
                        or again["session"] != adopted_session
                        or not self._session_owned(adopted_session)
                    )
                )
            ):
                return None
            if parent is not None:
                live_parent = _process_record(parent["pid"])
                if live_parent is None or self.key(live_parent) != self.key(parent):
                    return None
            self.records[key] = again
            self.pidfds[key] = fd
            if again["session"] == again["pid"]:
                self.sessions[again["session"]] = again
            published = True
            return again
        finally:
            if not published:
                os.close(fd)

    def _live(self, record):
        now = _process_record(record["pid"])
        return (
            now is not None
            and self.key(now) == self.key(record)
            and now["state"] not in ("Z", "X")
        )

    def _session_owned(self, session):
        leader = self.sessions.get(session)
        if leader is None or session in self.invalid_sessions:
            return False
        current = _process_record(leader["pid"])
        if current is not None and self.key(current) != self.key(leader):
            self.invalid_sessions.add(session)
            return False
        # A bound leader's pidfd and birth identity remain recorded after exit.
        # Fork inherits its SID; a new process cannot join that existing session.
        return self.key(leader) in self.pidfds

    def refresh(self):
        with self.lock:
            todo = list(self.records.values())
            seen = set()
            while todo:
                parent = todo.pop()
                key = self.key(parent)
                if key in seen:
                    continue
                seen.add(key)
                # A bound child may call setsid later. Refresh its exact birth
                # even while it is a zombie, before adopting its new-session
                # compiler children. Never bind a reused numeric PID here.
                parent = self._bind(parent["pid"], expected=key)
                if parent is None or not self._live(parent):
                    continue
                for pid in _children(parent["pid"]):
                    record = self._bind(pid, parent=parent)
                    if record is not None:
                        todo.append(record)

            if self.adopt_orphans:
                for pid in _children(self.collector_pid):
                    candidate = _process_record(pid)
                    if (
                        candidate is None
                        or self.key(candidate) in self.records
                        or self.key(candidate) in self.baseline
                    ):
                        continue
                    # Session membership is inherited by fork. A process cannot
                    # join somebody else's existing session. Our fresh setsid
                    # leaders bind that provenance before any child is adopted.
                    if (
                        candidate["ppid"] == self.collector_pid
                        and self._session_owned(candidate["session"])
                        and candidate["start_ticks"] >= self.root["start_ticks"]
                    ):
                        again = self._bind(pid, adopted_session=candidate["session"])
                        if again is not None:
                            self.actions.append(
                                {
                                    "pid": pid,
                                    "start_ticks": again["start_ticks"],
                                    "adopted_session": again["session"],
                                }
                            )

    def unknown_adoptees(self):
        if not self.adopt_orphans:
            return []
        unknown = []
        for pid in _children(self.collector_pid):
            record = _process_record(pid)
            if (
                record is not None
                and self.key(record) not in self.records
                and self.key(record) not in self.baseline
                and record["start_ticks"] >= self.root["start_ticks"]
            ):
                unknown.append(record)
        self.unverified_adopted = unknown
        return unknown

    def owns(self, pid: int) -> bool:
        with self.lock:
            candidate = _process_record(pid)
            if candidate is None:
                return False
            if self.key(candidate) in self.records:
                return True
            chain, seen = [], set()
            current = candidate
            while current is not None and current["pid"] not in seen:
                seen.add(current["pid"])
                if self.key(current) in self.records:
                    # Read each parent link again before publishing ownership.
                    for child in reversed(chain):
                        bound = self._bind(child["pid"], parent=current)
                        if bound is None or self.key(bound) != self.key(child):
                            return False
                        current = bound
                    return True
                chain.append(current)
                current = _process_record(current["ppid"])
            return False

    def signal(self, record, signum):
        with self.lock:
            key = self.key(record)
            fd = self.pidfds.get(key)
            if fd is None or not self._live(record):
                return False
            try:
                signal.pidfd_send_signal(fd, signum)
            except ProcessLookupError:
                return False
            self.actions.append(
                {
                    "pid": record["pid"],
                    "start_ticks": record["start_ticks"],
                    "signal": signum,
                    "monotonic_s": time.monotonic(),
                }
            )
            return True

    def reap_known(self):
        with self.lock:
            for record in self.records.values():
                if record["pid"] == self.root_pid:
                    continue  # Popen owns the root's return code.
                now = _process_record(record["pid"])
                if now is not None and self.key(now) != self.key(record):
                    continue
                with suppress(ChildProcessError):
                    os.waitid(
                        os.P_PIDFD,
                        self.pidfds[self.key(record)],
                        os.WEXITED | os.WNOHANG,
                    )

    def live_records(self):
        with self.lock:
            return [record for record in self.records.values() if self._live(record)]

    def close(self):
        with self.lock:
            for fd in self.pidfds.values():
                os.close(fd)
            self.pidfds.clear()

    def snapshot(self):
        with self.lock:
            return {
                "root": self.root,
                "processes": list(self.records.values()),
                "signals": list(self.actions),
                "live": self.live_records(),
                "unverified_adopted": self.unverified_adopted,
                "baseline": sorted(self.baseline),
                "invalid_sessions": sorted(self.invalid_sessions),
            }


@contextmanager
def _child_subreaper():
    """Adopt abandoned descendants so known compiler children can be reaped."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
    try:
        yield
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "cannot restore child subreaper")


def assert_gpu_exclusive(gpu, process_group, *, ownership=None):
    """Accept exact owned descendants; reject every unproved GPU process."""
    temporary = ownership is None
    if temporary:
        root = os.getpid() if process_group == os.getpgrp() else process_group
        ownership = ProcessOwnership(root)
    try:
        ownership.refresh()
        snapshot = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        )
        for line in snapshot.splitlines():
            uuid, pid = (part.strip() for part in line.split(",", 1))
            if uuid != gpu:
                continue
            try:
                record = _process_record(int(pid))
                if record is None:
                    continue  # The listed process exited before identity lookup.
                owned = ownership.owns(int(pid))
            except (OSError, ValueError):
                owned = False
            if not owned:
                raise RuntimeError(f"GPU contention on {gpu}: external process {pid}")
    finally:
        if temporary:
            ownership.close()


def run_engine(command, timeout=3600, stderr=None, log_path=None, cancelled=None):
    """Own the engine and verified descendants, including isolated worker groups."""
    with _PROCESS_RUN_LOCK, _child_subreaper(), ExitStack() as stack:
        output = stack.enter_context(Path(log_path).open("w")) if log_path else None
        return _run_engine(command, timeout, stderr, output, log_path, cancelled)


def _stop_engine(process, ownership):
    # Give Rust time to run its normal worker/process-group shutdown first.
    ownership.refresh()
    ownership.signal(ownership.root, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        ownership.refresh()
        time.sleep(0.02)
    # A dead root can leave known setsid descendants. Signal exact identities,
    # never a numeric process group that could contain an unproved client.
    for signum, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        ownership.refresh()
        for record in reversed(ownership.live_records()):
            ownership.signal(record, signum)
        deadline = time.monotonic() + grace
        while ownership.live_records() and time.monotonic() < deadline:
            ownership.refresh()
            ownership.reap_known()
            time.sleep(0.02)
    process.wait(timeout=2)
    ownership.reap_known()
    if ownership.live_records():
        raise RuntimeError("task-owned engine descendants did not exit")
    if ownership.unknown_adoptees():
        raise RuntimeError(
            "unproved adopted processes remain; no foreign process was signalled"
        )


def _run_engine(command, timeout, stderr, output, log_path, cancelled):
    baseline = _child_identities()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=output if output is not None else subprocess.PIPE,
        stderr=stderr,
        start_new_session=True,
    )
    ownership = None
    stop = threading.Event()
    monitor_errors = []
    monitor = None
    try:
        ownership = ProcessOwnership(process, adopt_orphans=True, baseline=baseline)

        def watch():
            while not stop.is_set():
                try:
                    ownership.refresh()
                except Exception as error:
                    monitor_errors.append(error)
                    return
                stop.wait(0.05)

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
        deadline = time.monotonic() + timeout
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("engine run was cancelled")
            if monitor_errors:
                raise RuntimeError(
                    "cannot establish engine descendant ownership"
                ) from monitor_errors[0]
            if gpu.startswith("GPU-"):
                assert_gpu_exclusive(gpu, process.pid, ownership=ownership)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                if output is not None:
                    process.wait(timeout=min(1, remaining))
                    stdout = Path(log_path).read_text()
                else:
                    stdout, _ = process.communicate(timeout=min(1, remaining))
                break
            except subprocess.TimeoutExpired:
                if output is None and process.poll() is not None:
                    # A compiler orphan may still hold the stdout pipe open.
                    # Do not wait for the benchmark timeout to notice root exit.
                    _stop_engine(process, ownership)
                    stdout, _ = process.communicate(timeout=2)
                    break
                continue
        ownership.refresh()
        if monitor_errors:
            raise RuntimeError(
                "cannot establish engine descendant ownership"
            ) from monitor_errors[0]
        if cancelled is not None and cancelled():
            raise InterruptedError("engine run was cancelled")
        if gpu.startswith("GPU-"):
            assert_gpu_exclusive(gpu, process.pid, ownership=ownership)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout)
        return stdout
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=2)
        try:
            if ownership is not None:
                _stop_engine(process, ownership)
            elif process.poll() is None:
                # The unreaped Popen object still owns this exact PID. Do not
                # guess ownership of any children when root binding failed.
                process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
                process.wait()
        finally:
            if ownership is not None:
                try:
                    if log_path is not None:
                        audit = Path(str(log_path) + ".processes.json")
                        audit.write_text(
                            json.dumps(ownership.snapshot(), indent=2) + "\n"
                        )
                finally:
                    ownership.close()
            if process.stdout is not None:
                process.stdout.close()


def rust_sources_sha256():
    """Mirror the binary build stamp without relying on file timestamps."""
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    paths.extend(
        path
        for path in (ROOT / "crates").rglob("*")
        if path.is_file() and (path.suffix == ".rs" or path.name == "Cargo.toml")
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(ROOT)).encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_identity(binary):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--porcelain").decode(),
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "rust_sources_sha256": rust_sources_sha256(),
        "python_sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "python").rglob("*.py"))
        },
    }


def parse_rows(stdout: str, marker: str):
    return [
        json.loads(line[len(marker) :])
        for line in stdout.splitlines()
        if line.startswith(marker)
    ]


def _parse_cuda_provenance(stdout: str) -> dict:
    rows = [
        json.loads(line.removeprefix("CUDA_BUILD_PROVENANCE "))
        for line in stdout.splitlines()
        if line.startswith("CUDA_BUILD_PROVENANCE ")
    ]
    if len(rows) != 1:
        raise ValueError("expected one loaded CUDA build provenance record")
    row = rows[0]
    if (
        not row.get("nvcc_path")
        or not row.get("nvcc_version")
        or str(row["nvcc_version"]).startswith("nvcc-unavailable:")
        or row["nvcc_version"] == "nvcc-changed-during-build"
        or row.get("compiler_identity_source") not in ("live", "manifest")
        or not row.get("so_path")
        or not isinstance(row.get("so_sha256"), str)
        or len(row["so_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in row["so_sha256"])
    ):
        raise ValueError("loaded CUDA build provenance is incomplete")
    return row


def _parse_bench_config(stdout: str, key: str, cast):
    """Parse a structured-log BENCH_CONFIG line emitted by the worker.

    The worker logs one line of the form:
        ... INFO BENCH_CONFIG key1=value1 key2=value2 ...
    This extracts the value for *key* and casts it with *cast*.  Raises
    ValueError if the line is absent or the key is missing, so a binary
    rebuilt with different defaults causes an immediate failure rather than
    a silent config mismatch.
    """
    import re

    for line in stdout.splitlines():
        if "BENCH_CONFIG" not in line:
            continue
        match = re.search(rf"\b{re.escape(key)}=(\S+)", line)
        if match:
            value = match.group(1)
            if value.startswith('"'):
                value = json.loads(value)
            return cast(value)
    raise ValueError(
        f"BENCH_CONFIG line not found or missing key '{key}' — "
        "worker must be rebuilt from this commit"
    )


def _observed_pool_capacities(stdout: str, expected_fa: int, expected_gdn: int):
    """Require the worker and scheduler to report the requested device slots."""
    worker = {
        "fa": _parse_bench_config(stdout, "worker_fa_pool_blocks", int),
        "mamba": _parse_bench_config(stdout, "worker_gdn_pool_blocks", int),
    }
    scheduler = {
        "fa": _parse_bench_config(stdout, "fa_pool_blocks", int),
        "mamba": _parse_bench_config(stdout, "gdn_pool_blocks", int),
    }
    expected = {"fa": expected_fa, "mamba": expected_gdn}
    if worker != expected or scheduler != expected:
        raise ValueError(
            f"observed cache capacities differ from requested: "
            f"worker={worker}, scheduler={scheduler}, requested={expected}"
        )
    return worker, scheduler
