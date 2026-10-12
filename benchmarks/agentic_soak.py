"""Opt-in continuous one-hour acceptance with four persistent OMP sessions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import threading
import time
from pathlib import Path

from benchmarks.agentic_context import ContextGuard
from benchmarks.agentic_session_evidence import audit_session
from benchmarks.agentic_sessions import RpcClient, curriculum, followup
from benchmarks.agentic_soak_analysis import (
    file_identity,
    plot_memory,
    plot_points,
    read_metrics,
)
from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy
from benchmarks.agentic_soak_support import (
    Evidence,
    collector_environment,
    read_jsonl,
    resource_sample,
    wait_idle,
)
from benchmarks.evidence import observed_configuration, runtime_source_identity

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mtp", "dspark"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--normalizer", type=Path, default=ROOT / "target/release/normalize-request"
    )
    parser.add_argument(
        "--model", type=Path, default=os.environ.get("OH_MY_VLLM_MODEL")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--agents", type=int, choices=(4,), default=4)
    parser.add_argument("--turn-timeout", type=float, default=1800)
    parser.add_argument("--drain-timeout", type=float, default=180)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument(
        "--diagnostic", action="store_true", help="short smoke, never formal acceptance"
    )
    parser.add_argument("--skip-plots", action="store_true", help="diagnostic only")
    return parser


def run(args):
    directory = args.output_dir.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("raw soak evidence must be outside the source repository")
    if not args.diagnostic and (args.duration < 3600 or args.skip_plots):
        raise ValueError("formal soak requires >=3600 seconds and real scatter figures")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            args.duration,
            args.turn_timeout,
            args.drain_timeout,
            args.sample_seconds,
        )
    ):
        raise ValueError("soak durations must be finite and positive")
    if not args.model:
        raise ValueError(
            "supply --model or OH_MY_VLLM_MODEL for exact CPU token checks"
        )
    directory.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(directory)
    source = runtime_source_identity(args.binary)
    if source["git_status"] and not args.diagnostic:
        raise ValueError("formal soak requires committed, clean source")
    startup = args.server_log.read_bytes()
    (directory / "server-startup.log").write_bytes(startup)
    observed = observed_configuration(startup.decode())
    if (
        observed.get("speculative_mode") != args.mode
        or int(observed.get("max_model_len", 0)) != 262144
    ):
        raise ValueError("service mode or maximum context differs")
    offset = args.server_log.stat().st_size
    log_identity = (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
    stop, sampler_stop, slots = threading.Event(), threading.Event(), ModelSlots()
    old_handlers = {
        sig: signal.signal(sig, lambda signum, frame: stop.set())
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    summary = {
        "schema": 2,
        "kind": "persistent_agentic_soak",
        "mode": args.mode,
        "passed": False,
        "diagnostic": args.diagnostic,
        "source": source,
        "observed_service_configuration": observed,
        "collector_environment": collector_environment(args.omp),
        "protocol": {
            "duration_s": args.duration,
            "agents": 4,
            "model_concurrency_limit": 4,
            "context_window": 262144,
            "thinking_level": "max",
            "reasoning_effort": "xhigh",
            "minimum_normal_requests": None,
            "warmup": None,
            "intermediate_drains": False,
            "performance_gate": None,
            "compaction_coverage_required": False,
        },
    }
    evidence.write("summary.json", summary)
    errors, clients, sessions = [], {}, {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    clock = {}
    guard, sampler = None, None
    initial_epoch = None
    normal_ended = None

    def sample_memory():
        while not sampler_stop.is_set():
            try:
                sample = resource_sample(args.base_url, initial_epoch, evidence)
                with lock:
                    sample["agents"] = [client.context() for client in clients.values()]
                evidence.append("memory.jsonl", sample)
            except Exception as error:
                errors.append(f"resource sampler: {error}")
                stop.set()
                return
            sampler_stop.wait(args.sample_seconds)

    def lane_run(lane):
        nonlocal normal_ended
        api = "chat" if lane < 2 else "responses"
        lane_directory = directory / f"lane-{lane}"
        lane_directory.mkdir()
        client = None
        try:
            with disk_proxy(
                args.base_url,
                lane_directory / "wire",
                slots,
                request_timeout=args.turn_timeout,
                preflight=guard.check,
                request_context=lambda: client.context() if client else {"lane": lane},
                normal_completion=lambda: not stop.is_set(),
                cancel_preflight=guard.close,
            ) as (proxy, _records):
                client = RpcClient(
                    args.omp, api, lane, lane_directory, proxy, stop, args.turn_timeout
                )
                client.initialize()
                with lock:
                    clients[lane] = client
                barrier.wait(timeout=120)
                with lock:
                    if not clock:
                        clock.update(monotonic=time.monotonic(), unix=time.time())
                        evidence.append("clock.jsonl", clock)
                paths, index = curriculum(lane), 0
                while not stop.is_set() and (
                    index == 0 or time.monotonic() - clock["monotonic"] < args.duration
                ):
                    client.prompt(followup(lane, index, paths), index)
                    evidence.append(
                        "progress.jsonl",
                        {
                            "lane": lane,
                            "turn": index,
                            "elapsed_s": time.monotonic() - clock["monotonic"],
                            "session_id": client.session_id,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "mode": args.mode,
                                "lane": lane,
                                "turn": index,
                                "elapsed_s": time.monotonic() - clock["monotonic"],
                            }
                        ),
                        flush=True,
                    )
                    index += 1
                if stop.is_set():
                    raise ValueError("persistent soak stopped before normal completion")
                active_ended = client.turns[-1]["completed_monotonic_s"]
                collected = client.collect()
                collected["active_seconds"] = active_ended - clock["monotonic"]
                with lock:
                    sessions[lane] = collected
                    normal_ended = max(normal_ended or active_ended, active_ended)
        except Exception as error:
            errors.append(f"lane {lane}: {type(error).__name__}: {error}")
            stop.set()
            barrier.abort()
            if guard:
                guard.close()
        finally:
            if client:
                client.close()

    try:
        baseline = wait_idle(args, evidence, stop, time.time(), phase="baseline")
        initial_epoch = baseline["epoch"]
        summary["initial_diagnostics"] = baseline
        summary["owned_service_processes"] = {}
        for name in ("server_pid", "worker_pid"):
            pid = baseline[name]
            stat = Path("/proc", str(pid), "stat").read_text()
            executable = file_identity(Path("/proc", str(pid), "exe"))
            if name == "server_pid" and executable["sha256"] != source["binary_sha256"]:
                raise ValueError("HTTP executable differs from recorded release binary")
            summary["owned_service_processes"][name] = {
                "pid": pid,
                "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
                "executable": executable,
            }
        guard = ContextGuard(args.model, args.normalizer, directory)
        summary["context_guard"] = guard.identity
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lane_run, range(4)))
        if any(
            session["active_seconds"] < args.duration for session in sessions.values()
        ):
            raise ValueError("a persistent lane ended before the required hour")
        sampler_stop.set()
        # Finish the HTTP/NVML observation before checking its errors.
        # The sampler must also stop writing evidence before final hashes.
        sampler.join()
        if errors:
            raise ValueError("; ".join(errors))
        final_idle = wait_idle(args, evidence, stop, time.time(), phase="post_run")
        for name, original in summary["owned_service_processes"].items():
            pid = final_idle[name]
            stat = Path("/proc", str(pid), "stat").read_text()
            ticks = int(stat[stat.rfind(")") + 2 :].split()[19])
            if pid != original["pid"] or ticks != original["start_ticks"]:
                raise ValueError("service process identity changed during the soak")
        summary["final_diagnostics"] = final_idle
        metrics = read_metrics(args.server_log, offset)
        by_response = {metric["response_id"]: metric for metric in metrics}
        points, audits = [], []
        for lane in range(4):
            selected, audit = audit_session(
                directory / f"lane-{lane}", sessions[lane], by_response, clock["unix"]
            )
            points.extend(selected)
            audits.append(audit)
        if {point["response_id"] for point in points} != set(by_response):
            raise ValueError("service and collector model request coverage differs")
        points.sort(key=lambda point: point["processing_started_unix_s"])
        for point in points:
            evidence.append("requests.jsonl", point)
        final_source = runtime_source_identity(args.binary)
        if final_source != source:
            raise ValueError("source or executable changed during collection")
        for marker in ("CUDA out of memory", "GPU worker restarted"):
            if marker in args.server_log.read_bytes()[offset:].decode():
                raise ValueError("unexpected OOM/restart during continuous service")
        if not args.skip_plots:
            plot_points(points, args.mode, directory)
            plot_memory(
                [*read_jsonl(directory / "memory.jsonl"), final_idle],
                args.mode,
                directory,
                clock["unix"],
            )
        summary.update(
            measured_started_unix_s=clock["unix"],
            collector_elapsed_s=normal_ended - clock["monotonic"],
            completed_normal_requests=len(points),
            sessions=audits,
            observed_model_concurrency=slots.peak,
            maximum_input_tokens=max(point["input_tokens"] for point in points),
            maximum_total_budget=max(
                point["input_tokens"] + point["output_budget"] for point in points
            ),
            compaction_requests=sum(point["purpose"] != "task" for point in points),
            memory_audit={
                "resource_bounds_passed": True,
                "final_release_passed": True,
                "continuous_context_growth_requires_review": True,
            },
            operational_passed=True,
            status="diagnostic_complete"
            if args.diagnostic
            else "awaiting_resource_review",
            passed=False,
        )
    except Exception as error:
        summary.update(error=f"{type(error).__name__}: {error}", errors=errors)
    finally:
        stop.set()
        sampler_stop.set()
        if sampler and sampler.ident:
            sampler.join()
        if guard:
            guard.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if (
            args.server_log.stat().st_dev,
            args.server_log.stat().st_ino,
        ) == log_identity:
            with args.server_log.open("rb") as log:
                log.seek(offset)
                (directory / "server-window.log").write_bytes(log.read())
        summary["raw_files"] = [
            {"path": path.relative_to(directory).as_posix(), **file_identity(path)}
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
    raise SystemExit(0 if result.get("operational_passed") else 1)
