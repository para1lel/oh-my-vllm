"""Process ownership, runtime identity, and output parsing for GPU collectors."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_identity():
    """Describe the independent environment; reject accidental legacy execution."""
    from oh_my_vllm.worker.runtime import identity

    worker = Path(os.environ.get("OH_MY_VLLM_WORKER_PYTHON", sys.executable))
    if worker.resolve() != Path(sys.executable).resolve():
        raise RuntimeError("benchmark worker must use the current independent Python")
    return identity()


def assert_gpu_exclusive(gpu, process_group):
    """Reject measurements contaminated by non-cooperating GPU clients."""
    snapshot = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    for line in snapshot.splitlines():
        uuid, pid = (part.strip() for part in line.split(",", 1))
        if uuid != gpu:
            continue
        try:
            group = os.getpgid(int(pid))
        except ProcessLookupError:
            continue  # The process exited between the snapshot and lookup.
        if group != process_group:
            raise RuntimeError(f"GPU contention on {gpu}: external process {pid}")


def run_engine(command, timeout=3600, stderr=None, log_path=None, cancelled=None):
    """Own the entire engine process group, including model-worker children."""
    with ExitStack() as stack:
        output = stack.enter_context(Path(log_path).open("w")) if log_path else None
        return _run_engine(command, timeout, stderr, output, log_path, cancelled)


def _run_engine(command, timeout, stderr, output, log_path, cancelled):
    process = subprocess.Popen(
        command,
        text=True,
        stdout=output if output is not None else subprocess.PIPE,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + timeout
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("engine run was cancelled")
            if gpu.startswith("GPU-"):
                assert_gpu_exclusive(gpu, process.pid)
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
                continue
        if cancelled is not None and cancelled():
            raise InterruptedError("engine run was cancelled")
        if gpu.startswith("GPU-"):
            assert_gpu_exclusive(gpu, process.pid)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout)
        return stdout
    finally:
        # A successful parent can also leave descendants behind.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
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
