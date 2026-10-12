"""Optional one-hour, four-agent serving collector with complete raw evidence.

The caller owns the dedicated HTTP service and its GPU worker. This collector
owns only OMP subprocesses and temporary loopback evidence proxies. It does not
load a model or leave a service running. Short diagnostics cannot pass formal
soak acceptance. All output must be outside the source repository.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

from benchmarks.agentic_soak_analysis import (
    device_used_bytes,
    file_identity,
    resource_checks,
)
from benchmarks.common import _process_record
from scripts.serving_evidence import stream_identity

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "soak_omp_acceptance", ROOT / "scripts/agentic-acceptance.py"
)
agentic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agentic)


def collector_environment(omp: str) -> dict:
    """Read package/executable metadata without loading a second GPU context."""
    executable = shutil.which(omp)
    return {
        "python_executable": sys.executable,
        "omp_executable": executable,
        "omp_file": file_identity(Path(executable)) if executable else None,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "transformers",
                "tilelang",
                "apache-tvm-ffi",
                "flashinfer-python",
                "pyzmq",
                "msgpack",
                "matplotlib",
                "opentelemetry-proto",
                "protobuf",
            )
        },
    }


class Evidence:
    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()

    def append(self, name: str, value: dict):
        with self.lock, (self.directory / name).open("a") as output:
            output.write(json.dumps(value, allow_nan=False) + "\n")

    def write(self, name: str, value: dict):
        temporary = self.directory / (name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, self.directory / name)


def run_omp(
    command: list[str],
    state: Path,
    directory: Path,
    stop: threading.Event,
    timeout: float,
) -> int:
    """Release the task-owned live process group on timeout or cancellation."""
    with (
        (directory / "events.jsonl").open("x") as output,
        (directory / "stderr.log").open("x") as errors,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ | {"PI_CODING_AGENT_DIR": str(state)},
            stdout=output,
            stderr=errors,
            start_new_session=True,
        )
        process_record = {
            "pid": process.pid,
            "pgid": process.pid,
            "started_unix_s": time.time(),
        }
        with suppress(FileNotFoundError):
            stat = Path("/proc", str(process.pid), "stat").read_text()
            process_record["start_ticks"] = int(stat[stat.rfind(")") + 2 :].split()[19])
        (directory / "process.json").write_text(json.dumps(process_record) + "\n")
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if stop.is_set() or time.monotonic() >= deadline:
                    return 124
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.2)
            return process.returncode
        finally:
            # Do not signal a reaped PID: it can have been reused by another job.
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process_record.update(
                returncode=process.returncode, stopped_unix_s=time.time(), reaped=True
            )
            (directory / "process.json").write_text(json.dumps(process_record) + "\n")


def diagnostics(base_url: str) -> dict:
    url = base_url.removesuffix("/v1").rstrip("/") + "/diagnostics"
    with urllib.request.urlopen(url, timeout=15) as response:
        if response.status != 200:
            raise ValueError("diagnostics endpoint is not ready")
        sample = json.load(response)
        if sample.get("worker"):
            sample["worker"]["gpu"]["device_used_bytes"] = device_used_bytes(sample)
        return sample


def gpu_processes(worker_pid: int) -> dict:
    """Keep foreign GPU use as evidence without stopping or rejecting its owner."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    rows = []
    for line in output.splitlines():
        uuid, pid, memory = (field.strip() for field in line.split(","))
        pid = int(pid)
        rows.append(
            {
                "gpu_uuid": uuid,
                "pid": pid,
                "used_memory_mib": int(memory) if memory.isdecimal() else None,
                "process": _process_record(pid),
            }
        )
    selected = {row["gpu_uuid"] for row in rows if row["pid"] == worker_pid}
    if len(selected) != 1:
        raise ValueError("serving worker has missing or ambiguous GPU identity")
    uuid = selected.pop()
    return {
        "gpu_uuid": uuid,
        "processes": [row for row in rows if row["gpu_uuid"] == uuid],
    }


def resource_sample(base_url: str, epoch: int, evidence: Evidence) -> dict:
    """Save failed worker observations before a GPU inventory can obscure them."""

    def observe():
        return {
            **diagnostics(base_url),
            "collected_unix_s": time.time(),
            "collected_monotonic_s": time.monotonic(),
            "phase": "normal",
            "observation": "active",
        }

    def failures(sample):
        checks = resource_checks(sample)
        failed = [check["name"] for check in checks if not check["passed"]]
        if sample.get("epoch") != epoch:
            failed.append(f"worker epoch {sample.get('epoch')} differs from {epoch}")
        return failed

    sample = observe()
    failed = failures(sample)
    if failed:
        sample["sampling_error"] = failed
        evidence.append("memory.jsonl", sample)
        raise ValueError(f"worker or resource checks failed: {failed}")
    try:
        sample["gpu_processes"] = gpu_processes(sample["worker_pid"])
    except Exception as error:
        sample["sampling_error"] = f"GPU process inventory: {error}"
        evidence.append("memory.jsonl", sample)
        # Recovery can start after the first HTTP snapshot and before NVML.
        try:
            fresh = observe()
        except Exception as recheck_error:
            evidence.append(
                "memory.jsonl",
                {
                    "phase": "normal",
                    "observation": "gpu_inventory_failure_recheck",
                    "collected_unix_s": time.time(),
                    "sampling_error": str(recheck_error),
                },
            )
            raise ValueError(
                f"GPU process inventory failed: {error}; "
                f"worker recheck failed: {recheck_error}"
            ) from error
        fresh["observation"] = "gpu_inventory_failure_recheck"
        fresh["sampling_error"] = failures(fresh)
        evidence.append("memory.jsonl", fresh)
        raise ValueError(
            f"GPU process inventory failed: {error}; worker recheck: "
            f"status={fresh.get('status')}, epoch={fresh.get('epoch')}, "
            f"restarts={fresh.get('restarts')}, failures={fresh['sampling_error']}"
        ) from error
    return sample


def wait_idle(
    args, evidence: Evidence, stop: threading.Event, after: float, phase: str = "normal"
) -> dict:
    deadline = time.monotonic() + args.drain_timeout
    last = None
    while not stop.is_set() and time.monotonic() < deadline:
        sample = diagnostics(args.base_url)
        sample["collected_unix_s"] = time.time()
        sample["collected_monotonic_s"] = time.monotonic()
        evidence.append("memory.jsonl", {"phase": "draining", **sample})
        checks = resource_checks(sample, require_idle=True)
        snapshot_time = sample.get("worker_snapshot_unix_s", 0)
        if all(check["passed"] for check in checks) and snapshot_time >= after:
            sample["phase"] = phase
            sample["observation"] = "drained"
            evidence.append("idle.jsonl", sample)
            return sample
        last = checks
        stop.wait(1)
    raise ValueError(
        f"request state did not drain with a fresh worker snapshot: {last}"
    )


def lifecycle_probe(args, evidence: Evidence, api: str, kind: str, epoch: int) -> dict:
    """Keep intentional cancellation/slow consumption separate from normal points."""
    endpoint = "/chat/completions" if api == "chat" else "/responses"
    body = {"model": "qwen3.8-27b-fp8", "stream": True}
    prompt = (
        "List consecutive integers starting at 1, one per line, as many as possible."
    )
    if api == "chat":
        body.update(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            reasoning_effort="off",
        )
    else:
        body.update(
            input=prompt,
            max_output_tokens=1024,
            reasoning={"effort": "off"},
            store=False,
        )
    record = {
        "api": api,
        "kind": kind,
        "epoch": epoch,
        "phase": "intentional_probe",
        "started_unix_s": time.time(),
        "request": body,
        "passed": False,
    }
    raw_path = evidence.directory / f"probe-{epoch:03d}-{api}-{kind}.sse"
    request = urllib.request.Request(
        args.base_url + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=180) as response,
            raw_path.open("xb") as output,
        ):
            record["http_status"] = response.status
            if kind == "slow":
                # A small receive window makes the pause visible to the server even
                # when an answer fits the normal socket buffer. Stay below its grace.
                connection = getattr(getattr(response.fp, "raw", None), "_sock", None)
                if connection is not None:
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
                time.sleep(args.slow_pause)
            identity, has_output = None, False
            for line in response:
                output.write(line)
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    event = json.loads(line[6:])
                    identity, content = stream_identity(event, api, identity)
                    has_output |= content
                    if kind == "cancel" and content:
                        break
            record.update(request_id=identity, has_output=has_output)
            if identity is None or not has_output:
                raise ValueError(
                    "lifecycle probe produced no identifiable model output"
                )
        record.update(
            passed=True, completed_unix_s=time.time(), raw=file_identity(raw_path)
        )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["completed_unix_s"] = time.time()
        if raw_path.exists():
            record["raw"] = file_identity(raw_path)
        evidence.append("probes.jsonl", record)
    return record


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
