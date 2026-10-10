"""Optional one-hour, four-agent serving collector with complete raw evidence.

The caller owns the dedicated HTTP service and its GPU worker. This collector
owns only OMP subprocesses and temporary loopback evidence proxies. It does not
load a model or leave a service running. Short diagnostics cannot pass formal
soak acceptance. All output must be outside the source repository.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import math
import os
import re
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
    bind_points,
    device_used_bytes,
    file_identity,
    late_idle_samples,
    memory_audit,
    normal_run_complete,
    plot_memory,
    plot_points,
    probe_outcomes,
    read_metrics,
    resource_checks,
    source_citations,
    task_budget,
)
from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy, load_requests
from benchmarks.evidence import observed_configuration, runtime_source_identity
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


def task_prompt(index: int, lane: int) -> str:
    """A fixed reusable prefix, varied context length, and real source-read task."""
    context = (
        "A request owns its state until completion or cancellation. "
        "A shared prefix can be reused after its computation completes. "
    )
    repetitions = (0, 8, 32, 128)[(index + lane) % 4]
    return (
        "Read these exact source ranges using the read tool: "
        "crates/scheduler/src/lib.rs:240-330 and "
        "python/oh_my_vllm/worker/model_runner.py:85-215. "
        "Read both before answering. Explain the schedule method and worker "
        "initialization from the returned source, in about 120 words. "
        "Cite both complete source filenames. Do not modify files. "
        "Use the actual source, and do not replace the reads with directory listings. "
        + context * repetitions
        + f"\nReview instance {index}, independent reader {lane}."
    )


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


def run_task(
    args,
    evidence: Evidence,
    slots: ModelSlots,
    stop: threading.Event,
    epoch: int,
    lane: int,
    index: int,
    phase: str = "normal",
) -> dict:
    task_id = f"epoch-{epoch:03d}-lane-{lane}-task-{index:04d}"
    directory = evidence.directory / task_id
    directory.mkdir()
    state = directory / "state"
    state.mkdir()
    api = "chat" if (lane + index) % 2 == 0 else "responses"
    summary = {
        "task_id": task_id,
        "api": api,
        "lane": lane,
        "epoch": epoch,
        "phase": phase,
        "started_unix_s": time.time(),
        "passed": False,
        "requests": [],
        "task_budget_seconds": task_budget(args, phase),
    }
    prompt = task_prompt(index, lane)
    (directory / "task.txt").write_text(prompt)
    command = [
        args.omp,
        "--cwd",
        str(ROOT),
        "--model",
        f"oh-my-vllm-{api}/qwen3.8-27b-fp8",
        "--thinking",
        "off",
        "--tools",
        "read",
        "--no-lsp",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-title",
        "--no-session",
        "--mode",
        "json",
        "--max-time",
        f"{math.ceil(task_budget(args, phase))}s",
        "--print",
        prompt,
    ]
    try:
        with disk_proxy(
            args.base_url,
            directory / "wire",
            slots,
            request_timeout=task_budget(args, phase),
        ) as (proxy, _records):
            configuration = agentic.configuration(proxy)
            for provider in configuration["providers"].values():
                provider["models"][0]["maxTokens"] = 2048
            (state / "models.yml").write_text(json.dumps(configuration, indent=2))
            code = run_omp(command, state, directory, stop, task_budget(args, phase))
        summary["exit_code"] = code
        requests = load_requests(directory / "wire")
        summary["requests"] = [
            {key: value for key, value in record.items() if key != "request"}
            for record in requests
        ]
        if code:
            raise ValueError(f"normal OMP task exited {code}")
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        summary["tool_verification"] = agentic.validate_agent_evidence(
            events, requests, api, citation_check=source_citations
        )
        summary["passed"] = True
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        wire_log = directory / "wire/http.jsonl"
        if wire_log.exists() and not summary["requests"]:
            summary["requests"] = [
                json.loads(line) for line in wire_log.read_text().splitlines()
            ]
        stop.set()
    finally:
        summary["completed_unix_s"] = time.time()
        evidence.append("tasks.jsonl", summary)
    return summary


def diagnostics(base_url: str) -> dict:
    url = base_url.removesuffix("/v1").rstrip("/") + "/diagnostics"
    with urllib.request.urlopen(url, timeout=15) as response:
        if response.status != 200:
            raise ValueError("diagnostics endpoint is not ready")
        sample = json.load(response)
        if sample.get("worker"):
            sample["worker"]["gpu"]["device_used_bytes"] = device_used_bytes(sample)
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mtp", "dspark"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--min-requests", type=int, default=500)
    parser.add_argument("--agents", type=int, choices=(4,), default=4)
    parser.add_argument("--epoch-seconds", type=float, default=300)
    parser.add_argument("--warmup-epoch-seconds", type=float, default=30)
    parser.add_argument("--max-runtime", type=float, default=10800)
    parser.add_argument("--task-timeout", type=float, default=210)
    parser.add_argument("--warmup-task-timeout", type=float, default=900)
    parser.add_argument("--drain-timeout", type=float, default=90)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--slow-pause", type=float, default=5)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="shortened run, never formal acceptance",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="diagnostic-only when plotting tools are unavailable",
    )
    return parser


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def run(args):
    args.base_url = args.base_url.rstrip("/")
    directory = args.output_dir.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("raw soak evidence must be outside the source repository")
    if not args.diagnostic and (
        args.duration < 3600 or args.min_requests < 500 or args.skip_plots
    ):
        raise ValueError(
            "formal soak requires >=3600 seconds, >=500 requests, "
            "and real scatter figures"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            args.duration,
            args.min_requests,
            args.epoch_seconds,
            args.warmup_epoch_seconds,
            args.max_runtime,
            args.task_timeout,
            args.warmup_task_timeout,
            args.drain_timeout,
            args.sample_seconds,
        )
    ):
        raise ValueError("soak durations and counts must be positive")
    if args.max_runtime < args.duration or not 0 <= args.slow_pause < 20:
        raise ValueError("runtime limit or slow-client pause is invalid")
    directory.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(directory)
    evidence.write(
        "summary.json", {"passed": False, "phase": "preflight", "mode": args.mode}
    )
    source = runtime_source_identity(args.binary)
    if source["git_status"] and not args.diagnostic:
        raise ValueError("formal soak requires committed, clean source")
    startup_log = args.server_log.read_bytes()
    (directory / "server-startup.log").write_bytes(startup_log)
    observed = observed_configuration(startup_log.decode())
    if observed.get("speculative_mode") != args.mode:
        raise ValueError("observed service mode differs from the requested soak mode")
    log_identity = (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
    log_offset = args.server_log.stat().st_size
    stop, sampler_stop, slots = threading.Event(), threading.Event(), ModelSlots()
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.set())
    started, started_unix = time.monotonic(), time.time()
    summary = {
        "schema": 1,
        "kind": "agentic_soak",
        "mode": args.mode,
        "passed": False,
        "diagnostic": args.diagnostic,
        "source": source,
        "observed_service_configuration": observed,
        "collector_environment": collector_environment(args.omp),
        "measured_started_unix_s": started_unix,
        "protocol": {
            "duration_s": args.duration,
            "minimum_normal_requests": args.min_requests,
            "agents": 4,
            "model_concurrency_limit": 4,
            "epoch_seconds": args.epoch_seconds,
            "warmup_epoch_seconds": args.warmup_epoch_seconds,
            "warm_idle_epochs": 2,
            "minimum_late_idle_snapshots": 6,
            "normal_task_budget_seconds": args.task_timeout,
            "warmup_task_budget_seconds": args.warmup_task_timeout,
            "preconditioning_limit_seconds": 30
            + 3 * args.drain_timeout
            + 2 * (args.warmup_epoch_seconds + args.warmup_task_timeout),
            "performance_gate": None,
        },
        "server_log": str(args.server_log),
        "server_log_offset": log_offset,
        "limits": [
            "Finite-duration evidence does not prove the absence of every leak.",
            "Each actual source-read tool loop still requires semantic answer review.",
        ],
    }
    evidence.write("summary.json", summary)
    sample_errors = []
    epoch_identity = None
    phase = "warmup"

    def sample_memory():
        while not sampler_stop.is_set():
            try:
                sample = diagnostics(args.base_url)
                sample["collected_unix_s"] = time.time()
                sample["collected_monotonic_s"] = time.monotonic()
                evidence.append("memory.jsonl", {"phase": phase, **sample})
                if sample.get("epoch") != epoch_identity:
                    raise ValueError("worker epoch changed during normal soak")
                if not all(check["passed"] for check in resource_checks(sample)):
                    raise ValueError(
                        "resource cap, worker state, or restart check failed"
                    )
            except Exception as error:
                sample_errors.append(str(error))
                stop.set()
                return
            sampler_stop.wait(args.sample_seconds)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    try:
        baseline = wait_idle(args, evidence, stop, time.time(), phase="baseline")
        epoch_identity = baseline["epoch"]
        summary["initial_diagnostics"] = baseline
        summary["owned_service_processes"] = {}
        for key in ("server_pid", "worker_pid"):
            pid = baseline.get(key)
            if not isinstance(pid, int) or pid <= 0:
                if not args.diagnostic:
                    raise ValueError("diagnostics has no live service process identity")
                continue
            stat = Path("/proc", str(pid), "stat").read_text()
            executable = file_identity(Path("/proc", str(pid), "exe"))
            summary["owned_service_processes"][key] = {
                "pid": pid,
                "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
                "executable": executable,
            }
            if key == "server_pid" and executable["sha256"] != source["binary_sha256"]:
                raise ValueError("running HTTP server differs from the release binary")

        sampler.start()
        for warm_epoch in range(2):
            if (
                time.monotonic() - started
                >= summary["protocol"]["preconditioning_limit_seconds"]
            ):
                raise ValueError("warm preconditioning exceeded its bounded deadline")
            deadline = time.monotonic() + args.warmup_epoch_seconds

            def warm_lane(lane, until=deadline, epoch=warm_epoch):
                index = 0
                while not stop.is_set() and (index == 0 or time.monotonic() < until):
                    run_task(
                        args, evidence, slots, stop, epoch, lane, index, phase="warmup"
                    )
                    index += 1

            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                list(pool.map(warm_lane, range(4)))
            if stop.is_set():
                raise ValueError("a warmup OMP task failed or collection was cancelled")
            wait_idle(args, evidence, stop, time.time(), phase="warmup")
        summary["preconditioning_elapsed_seconds"] = time.monotonic() - started
        started, started_unix = time.monotonic(), time.time()
        summary["measured_started_unix_s"] = started_unix
        evidence.write("summary.json", summary)
        phase = "normal"
        normal_count, normal_span, epoch = 0, 0.0, 2
        normal_idle, normal_ended = [], started
        while not normal_run_complete(
            normal_span,
            normal_ended - started,
            normal_count,
            args.duration,
            args.min_requests,
            idle_samples=None if args.diagnostic else normal_idle,
            normal_started=started,
        ):
            if stop.is_set() or time.monotonic() - started >= args.max_runtime:
                raise ValueError(
                    "soak cancelled, exceeded deadline, "
                    "or failed its active resource check"
                )
            epoch_deadline = min(
                started + args.max_runtime, time.monotonic() + args.epoch_seconds
            )

            def lane_run(lane, deadline=epoch_deadline, current_epoch=epoch):
                completed, index = [], 0
                while not stop.is_set() and (index == 0 or time.monotonic() < deadline):
                    completed.append(
                        run_task(
                            args, evidence, slots, stop, current_epoch, lane, index
                        )
                    )
                    index += 1
                return completed

            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                list(pool.map(lane_run, range(4)))
            if stop.is_set():
                raise ValueError("a normal OMP task failed or collection was cancelled")
            for api in ("chat", "responses"):
                for kind in ("cancel", "slow"):
                    lifecycle_probe(args, evidence, api, kind, epoch)
            sample = wait_idle(args, evidence, stop, time.time())
            normal_idle.append(sample)
            normal_ended = time.monotonic()
            normal_tasks = [
                task
                for task in read_jsonl(directory / "tasks.jsonl")
                if task["phase"] == "normal"
            ]
            points = bind_points(
                read_metrics(args.server_log, log_offset), normal_tasks, started_unix
            )
            normal_count = len(points)
            normal_span = max(
                point["generation_completed_unix_s"] for point in points
            ) - min(point["processing_started_unix_s"] for point in points)
            progress = {
                "mode": args.mode,
                "epoch": epoch,
                "normal_requests": normal_count,
                "normal_request_span_s": normal_span,
                "wall_elapsed_s": normal_ended - started,
                "eligible_late_idle_snapshots": len(
                    late_idle_samples(normal_idle, started, normal_ended)
                ),
                "idle_sampled_unix_s": sample["sampled_unix_s"],
            }
            evidence.append("epochs.jsonl", progress)
            print(json.dumps(progress), flush=True)
            epoch += 1
        summary["measured_duration_s"] = normal_span
        summary["collector_elapsed_s"] = normal_ended - started
        metrics = read_metrics(args.server_log, log_offset)
        tasks = read_jsonl(directory / "tasks.jsonl")
        normal_tasks = [task for task in tasks if task["phase"] == "normal"]
        warm_tasks = [task for task in tasks if task["phase"] == "warmup"]
        points = bind_points(metrics, normal_tasks, started_unix)
        warm_points = bind_points(metrics, warm_tasks, started_unix)
        idle = read_jsonl(directory / "idle.jsonl")
        post_warm = [sample for sample in idle if sample["phase"] == "normal"]
        prefix_hits = post_warm[-1].get("prefix_hit_tokens", 0) - baseline.get(
            "prefix_hit_tokens", 0
        )
        if prefix_hits <= 0 and not args.diagnostic:
            raise ValueError("no real prefix cache hit was observed")
        summary["observed_prefix_hit_tokens"] = prefix_hits
        audit = memory_audit(post_warm, started, normal_ended)
        probe_records = read_jsonl(directory / "probes.jsonl")
        probe_ids = {record["request_id"] for record in probe_records}
        normal_ids = {point["request_id"] for point in points}
        actual_ids = {metric["request_id"] for metric in metrics}
        warm_ids = {point["request_id"] for point in warm_points}
        if actual_ids != normal_ids | probe_ids | warm_ids:
            raise ValueError(
                "server request metrics include unknown requests "
                "or omit intentional probes"
            )
        outcomes = probe_outcomes(probe_records, metrics)
        evidence.write("probe-outcomes.json", outcomes)
        if (
            args.server_log.stat().st_dev,
            args.server_log.stat().st_ino,
        ) != log_identity:
            raise ValueError("server log changed identity")
        with args.server_log.open("rb") as source_log:
            source_log.seek(log_offset)
            new_log = source_log.read()
        (directory / "server-window.log").write_bytes(new_log)
        if re.search(
            rb"CUDA out of memory|OutOfMemoryError|memory allocation failed with OOM",
            new_log,
        ):
            raise ValueError("unexpected OOM occurred during normal soak")
        if runtime_source_identity(args.binary) != source:
            raise ValueError("source or release binary changed during measurement")
        evidence.write("points.json", {"mode": args.mode, "points": points})
        if not args.skip_plots:
            plot_points(points, args.mode, directory)
            plot_memory(
                read_jsonl(directory / "memory.jsonl") + post_warm,
                args.mode,
                directory,
                started_unix,
            )
        summary.update(
            normal_completed_requests=len(points),
            completed_agent_tasks=len(normal_tasks),
            warmup_requests=len(warm_points),
            warmup_agent_tasks=len(warm_tasks),
            api_counts={
                api: sum(point["api"] == api for point in points)
                for api in ("chat", "responses")
            },
            lane_counts={
                str(lane): sum(point["lane"] == lane for point in points)
                for lane in range(4)
            },
            first_processing_started_unix_s=min(
                p["processing_started_unix_s"] for p in points
            ),
            last_generation_completed_unix_s=max(
                p["generation_completed_unix_s"] for p in points
            ),
            observed_model_concurrency=slots.peak,
            idle_epochs=len(idle),
            eligible_late_idle_snapshots=audit["gpu_allocated"]["late_samples"],
            memory_audit=audit,
            intentional_probes=len(probe_records),
            probe_outcomes=outcomes,
            sampler_errors=sample_errors,
            queue_seconds={
                "minimum": min(p["queue_seconds"] for p in points),
                "maximum": max(p["queue_seconds"] for p in points),
            },
            passed=not args.diagnostic
            and normal_run_complete(
                summary["measured_duration_s"],
                summary["collector_elapsed_s"],
                len(points),
                args.duration,
                args.min_requests,
                idle_samples=post_warm,
                normal_started=started,
            )
            and all(record["passed"] for record in audit.values())
            and not sample_errors,
        )
        if not all(record["passed"] for record in audit.values()):
            summary["investigation_required"] = (
                "Idle memory growth or insufficient idle samples needs review."
            )
    except Exception as error:
        summary.update(
            error=f"{type(error).__name__}: {error}",
            collector_elapsed_s=time.monotonic() - started,
            sampler_errors=sample_errors,
        )
    finally:
        stop.set()
        sampler_stop.set()
        if sampler.ident is not None:
            sampler.join(timeout=20)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if (
            args.server_log.stat().st_dev,
            args.server_log.stat().st_ino,
        ) == log_identity:
            with args.server_log.open("rb") as source_log:
                source_log.seek(log_offset)
                (directory / "server-window.log").write_bytes(source_log.read())
        summary["raw_files"] = [
            dict(path=path.relative_to(directory).as_posix(), **file_identity(path))
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "summary.json"
        ]
        evidence.write("summary.json", summary)
    return summary


def main():
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("source", "raw_files")
            },
            indent=2,
        )
    )
    raise SystemExit(
        0 if result["passed"] or (args.diagnostic and "error" not in result) else 1
    )
