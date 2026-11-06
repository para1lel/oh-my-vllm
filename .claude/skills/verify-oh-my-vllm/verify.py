#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
from http.client import HTTPException
from pathlib import Path

from drive import Http, drive, require
from evidence import ROOT, manifest, save_json, timestamp
from processes import Processes, enable_subreaper, process_identity


def artifact_directory(value):
    if not value:
        raise ValueError("set ARTIFACT_ROOT or use --artifact-root")
    root = Path(value).expanduser().resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("ARTIFACT_ROOT must be external to the repository")
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="verify-oh-my-vllm-", dir=root))


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def fixture_environment(model, artifacts):
    from tests.worker_fixture import install

    model.mkdir()
    environment = install(model)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    save_json(
        artifacts / "fixture.json",
        {
            "installer": "tests.worker_fixture.install",
            "worker_python": environment["OH_MY_VLLM_WORKER_PYTHON"],
            "substitutions": [
                "checkpoint config and identities",
                "token preparation and fixed replies",
                "worker readiness, execution, and cache messages",
            ],
            "config": json.loads((model / "config.json").read_text()),
            "replies": json.loads((model / "replies.json").read_text()),
        },
    )
    return environment


def doctor(http, service, processes, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f"service exited during launch: {service.returncode}")
        try:
            health = http.json("doctor-diagnostics", "GET", "/diagnostics")
            if health.get("worker_pid"):
                break
        except (OSError, HTTPException):
            pass
        time.sleep(0.1)
    else:
        raise TimeoutError("worker readiness deadline expired")
    owned = {record["pid"] for record in processes.observe()}
    require(health["server_pid"] == service.pid, "HTTP server PID is not owned")
    require(health["worker_pid"] in owned, "worker PID is not owned")
    worker = process_identity(health["worker_pid"])
    require(
        worker is not None and worker["ppid"] == service.pid, "worker parent differs"
    )
    models = http.json("doctor-models", "GET", "/v1/models")
    http.model = models["data"][0]["id"]
    save_json(http.artifacts / "doctor.json", {"diagnostics": health, "models": models})


def command_factory(binary, model, scratch):
    def command(label):
        return [
            str(binary),
            "--model",
            str(model),
            "--socket",
            str(scratch / f"{label}.ipc"),
            "--num-gpu-blocks",
            "8192",
        ]

    return command


def execute(args, artifacts, scratch, summary):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    binary = args.binary.expanduser().resolve(strict=True)
    save_json(artifacts / "manifest.json", manifest(binary, args.feature))
    model = scratch / "model"
    environment = fixture_environment(model, artifacts)
    environment["OH_MY_VLLM_RUN_ID"] = artifacts.name
    processes = Processes(artifacts, environment)
    command = command_factory(binary, model, scratch)
    port = free_port()
    save_json(
        artifacts / "launch.json",
        {"scratch": str(scratch), "port": port, "ipc": str(scratch / "service.ipc")},
    )
    try:
        processes.start("help", [str(binary), "--help"])
        processes.wait("help", 5)
        service = processes.start(
            "service",
            [
                *command("service"),
                "serve",
                "--listen",
                f"127.0.0.1:{port}",
                "--diagnostics-interval-seconds",
                "1",
                "--request-timeout-seconds",
                "5",
                "--shutdown-grace-seconds",
                "2",
            ],
        )
        http = Http(port, artifacts, processes)
        print("Doctor: wait for the owned service and worker", flush=True)
        doctor(http, service, processes, args.launch_timeout)
        print(f"Drive: {args.feature}", flush=True)
        summary["features"] = drive(args.feature, http, processes, command)
        http.json("final-diagnostics", "GET", "/diagnostics")
        summary["status"] = "completed"
    finally:
        print("Cleanup: stop owned processes", flush=True)
        processes.cleanup()
        summary["cleanup_completed"] = True


def interrupt(_number, _frame):
    raise KeyboardInterrupt


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("this helper uses Linux /proc process identities")
    os.umask(0o077)
    artifacts = artifact_directory(args.artifact_root)
    print(f"Evidence: {artifacts}", flush=True)
    enable_subreaper()
    scratch = Path(tempfile.mkdtemp(prefix="omv-verify-", dir="/tmp"))
    summary = {"status": "failed", "scope": "CPU service/transport fixture only"}
    signal.signal(signal.SIGTERM, interrupt)
    try:
        execute(args, artifacts, scratch, summary)
    except (Exception, KeyboardInterrupt) as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        (artifacts / "failure.txt").write_text(traceback.format_exc())
        print(summary["error"], file=sys.stderr)
    finally:
        if summary.get("cleanup_completed") or not (artifacts / "launch.json").exists():
            shutil.rmtree(scratch)
        summary["scratch_removed"] = not scratch.exists()
        summary["finished_at"] = timestamp()
        save_json(artifacts / "result.json", summary)
    print(f"{summary['status']}: {artifacts}", flush=True)
    return 0 if summary["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(
        description="Drive the Rust app with the CPU worker fixture and save evidence."
    )
    parser.add_argument("action", choices=["run"])
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/debug/oh-my-vllm-zmq-worker"
    )
    parser.add_argument("--artifact-root", default=os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument(
        "--feature",
        choices=["chat", "responses", "tools", "operations", "all"],
        default="all",
    )
    parser.add_argument("--launch-timeout", type=float, default=15)
    args = parser.parse_args()
    if args.launch_timeout <= 0:
        parser.error("--launch-timeout must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
