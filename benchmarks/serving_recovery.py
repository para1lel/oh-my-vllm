"""Opt-in target-model allocator OOM and HTTP recovery acceptance.

Each fault stage owns a separate service and uses an external test-only worker
launcher. Failures are retained. This collector is separate from normal serving
soak and performance acceptance, and it leaves no service running.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from benchmarks.agentic_soak_analysis import (
    file_identity,
    read_metrics,
)
from benchmarks.common import (
    ProcessOwnership,
    _child_identities,
    _child_subreaper,
    _process_record,
    _stop_engine,
    assert_gpu_exclusive,
)
from benchmarks.evidence import (
    checkpoint_identity,
    loaded_draft_provenance,
    observed_configuration,
    runtime_source_identity,
)
from benchmarks.serving_recovery_analysis import (
    body,
    cached_tokens,
    fresh_snapshot,
    idle_checks,
    live_group,
    validate_arithmetic_answer,
    validate_configuration,
    validate_fault_stream,
    validate_oom,
    validate_replacement,
)
from scripts.serving_evidence import response_text, stream_identity

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("after_commit", "graph_capture", "sampling", "proposal")
FIXTURE = ROOT / "tests/fixtures/worker_faults.py"


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def launcher(directory: Path, stage: str, marker: Path) -> Path:
    """Import the real fault fixture by filename, then run the real ZMQ bridge."""
    if stage not in STAGES:
        raise ValueError("unknown recovery stage")
    bootstrap = directory / "worker-launcher.py"
    bootstrap.write_text(
        "import importlib.util, runpy, sys\n"
        "spec = importlib.util.spec_from_file_location(\n"
        f"    'owned_worker_faults', {str(FIXTURE)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"module.install_fault({stage!r}, 'real_oom', {str(marker)!r})\n"
        "if sys.argv[1:3] != ['-m', 'oh_my_vllm.worker.zmq_bridge']:\n"
        "    raise RuntimeError('unexpected owned-worker arguments')\n"
        "sys.argv = [sys.argv[2], *sys.argv[3:]]\n"
        "runpy.run_module('oh_my_vllm.worker.zmq_bridge',\n"
        "                 run_name='__main__', alter_sys=True)\n"
    )
    executable = directory / "worker-python"
    executable.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(bootstrap))
        + ' "$@"\n'
    )
    executable.chmod(0o700)
    return executable


class Wire:
    """Persist every request and exact response bytes before assertions."""

    def __init__(self, directory: Path, base_url: str, timeout: float):
        self.directory, self.base_url, self.timeout = directory, base_url, timeout
        self.records, self.lock = [], threading.Lock()

    def request(self, name: str, path: str, value=None):
        record = {
            "name": name,
            "path": path,
            "request": value,
            "started_unix_s": time.time(),
        }
        write_json(self.directory / (name + "-request.json"), record)
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(value).encode() if value is not None else None,
            headers={"Content-Type": "application/json"},
        )
        raw = b""
        try:
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                record["status"] = response.status
                raw = response.read()
            parsed = json.loads(raw)
            record["response"] = parsed
            return response.status, parsed
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            path = self.directory / (name + "-response.bin")
            path.write_bytes(raw)
            record.update(completed_unix_s=time.time(), raw=file_identity(path))
            with self.lock:
                self.records.append(record)
                write_json(self.directory / (name + "-result.json"), record)

    def stream(self, name: str, value: dict, marker: Path, stage: str, check):
        record = {
            "name": name,
            "request": value,
            "stage": stage,
            "started_unix_s": time.time(),
        }
        write_json(self.directory / (name + "-request.json"), record)
        path = self.directory / (name + ".sse")
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"},
        )
        output_seen, failures, successes, armed, rid = False, 0, 0, False, None
        response_id = None
        if stage == "graph_capture":
            marker.unlink()
            armed = True
            record["armed_unix_s"] = time.time()
        try:
            with path.open("xb") as output:
                try:
                    response = urllib.request.urlopen(request, timeout=self.timeout)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    record["status"] = response.status
                    for line in response:
                        output.write(line)
                        output.flush()
                        check()
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:].strip()
                        if data == b"[DONE]":
                            successes += 1
                            continue
                        event = json.loads(data)
                        incoming = event.get("id")
                        if incoming:
                            if response_id is not None and incoming != response_id:
                                raise ValueError(
                                    "fault SSE changed its response identity"
                                )
                            response_id = incoming
                        rid, visible = stream_identity(event, "chat", rid)
                        output_seen |= visible
                        if visible and not armed:
                            marker.unlink()
                            armed = True
                            record["armed_unix_s"] = time.time()
                        failures += int(
                            bool(event.get("error")) or event.get("type") == "error"
                        )
                        successes += sum(
                            choice.get("finish_reason") is not None
                            for choice in event.get("choices", [])
                        )
            record.update(
                request_id=rid,
                response_id=response_id,
                visible_output=output_seen,
                armed=armed,
                explicit_errors=failures,
                success_events=successes,
                marker_recreated=marker.is_file(),
            )
            validate_fault_stream(record)
            return record
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record.update(
                request_id=rid,
                response_id=response_id,
                visible_output=output_seen,
                armed=armed,
                explicit_errors=failures,
                success_events=successes,
                marker_recreated=marker.is_file(),
            )
            record["completed_unix_s"] = time.time()
            if path.exists():
                record["raw"] = file_identity(path)
            with self.lock:
                self.records.append(record)
                write_json(self.directory / (name + "-result.json"), record)


class OwnedService:
    """Track exact service/worker births and release every verified descendant."""

    def __init__(self, command, environment, directory: Path, gpu: str, timeout: float):
        self.command, self.environment, self.directory = command, environment, directory
        self.gpu, self.timeout = gpu, timeout
        self.stop, self.errors = threading.Event(), []
        self.process = self.ownership = self.monitor = self.log = None

    def __enter__(self):
        self.log = (self.directory / "server.log").open("x")
        try:
            baseline = _child_identities()
            self.process = subprocess.Popen(
                self.command,
                env=self.environment,
                cwd=ROOT,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.ownership = ProcessOwnership(
                self.process, adopt_orphans=True, baseline=baseline
            )
            write_json(self.directory / "command.json", self.command)
            started = time.monotonic()

            def watch():
                next_gpu = 0.0
                while not self.stop.is_set():
                    try:
                        self.ownership.refresh()
                        if time.monotonic() - started >= self.timeout:
                            raise TimeoutError("owned recovery stage deadline exceeded")
                        if time.monotonic() >= next_gpu:
                            assert_gpu_exclusive(
                                self.gpu, self.process.pid, ownership=self.ownership
                            )
                            next_gpu = time.monotonic() + 1
                    except Exception as error:
                        self.errors.append(error)
                        write_json(
                            self.directory / "monitor-error.json",
                            {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        self.ownership.signal(self.ownership.root, signal.SIGTERM)
                        return
                    self.stop.wait(0.05)

            self.monitor = threading.Thread(
                target=watch, name="recovery-process-monitor", daemon=True
            )
            self.monitor.start()
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.errors:
            raise RuntimeError(
                f"owned recovery monitor failed: {self.errors[0]}"
            ) from self.errors[0]
        if self.process.poll() is not None:
            raise RuntimeError(f"owned HTTP service exited {self.process.returncode}")

    def close(self):
        self.stop.set()
        monitor_error = None
        if self.monitor is not None:
            self.monitor.join(timeout=15)
            if self.monitor.is_alive():
                monitor_error = RuntimeError("owned process monitor did not stop")
        try:
            if self.process is not None:
                if self.ownership is not None:
                    _stop_engine(self.process, self.ownership)
                    write_json(
                        self.directory / "processes.json", self.ownership.snapshot()
                    )
                else:
                    # This birth is still unreaped and belongs to this Popen.
                    self.process.kill()
                    self.process.wait(timeout=30)
        finally:
            if self.ownership is not None:
                self.ownership.close()
            if self.log is not None:
                self.log.close()
        if monitor_error:
            raise monitor_error

    def __exit__(self, kind, error, traceback):
        self.close()


def snapshot(wire: Wire, name: str, service: OwnedService) -> dict:
    service.check()
    status, value = wire.request(name, "/diagnostics")
    if status != 200 or value.get("server_pid") != service.process.pid:
        raise ValueError("HTTP process identity changed or diagnostics failed")
    with (wire.directory / "diagnostics.jsonl").open("a") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")
    return value


def wait_snapshot(wire, service, timeout, label, predicate):
    deadline, index = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        try:
            value = snapshot(wire, f"{label}-{index:04d}", service)
        except urllib.error.URLError:
            value = None
        if value is not None and predicate(value):
            return value
        index += 1
        time.sleep(0.2)
    raise TimeoutError(f"recovery diagnostics did not reach {label}")


def wait_idle(wire, service, timeout, label, after, restarts):
    return wait_snapshot(
        wire,
        service,
        timeout,
        label,
        lambda sample: (
            fresh_snapshot(sample, after)
            and all(check["passed"] for check in idle_checks(sample, restarts))
        ),
    )


def command(args, directory, port):
    value = [
        str(args.binary),
        "--model",
        str(args.model),
        "--socket",
        str(directory / "worker.ipc"),
        "--num-gpu-blocks",
        "4200",
        "--mamba-blocks",
        "128",
        "--max-model-len",
        "262144",
        "--speculative-mode",
        args.mode,
        "--num-speculative-tokens",
        "4" if args.mode == "mtp" else "7",
    ]
    if args.mode == "dspark":
        value.extend(["--draft-model", str(args.draft_model)])
    value.extend(
        [
            "serve",
            "--listen",
            f"127.0.0.1:{port}",
            "--diagnostics-interval-seconds",
            "1",
        ]
    )
    return value


def run_stage(args, directory: Path, stage: str, gpu: str) -> dict:
    directory.mkdir()
    marker = directory / "fault-once.marker"
    marker.write_text("warmup disables fault injection\n")
    executable = launcher(directory, stage, marker)
    summary = {"stage": stage, "kind": "real_allocator_oom", "passed": False}
    summary["launcher"] = {
        name: file_identity(directory / name)
        for name in ("worker-launcher.py", "worker-python")
    }
    write_json(directory / "summary.json", summary)
    shared = (
        "Shared serving recovery context, with completed reusable cache pages. " * 800
    )
    prompt = (
        shared + "\nList consecutive integers starting at 1, one per line, "
        "as many as possible."
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    wire = Wire(directory, f"http://127.0.0.1:{port}", args.request_timeout)
    environment = os.environ | {
        "OH_MY_VLLM_WORKER_PYTHON": str(executable),
        "RUST_LOG": "info,oh_my_vllm_zmq_worker::serving=debug",
    }
    try:
        with (
            tempfile.TemporaryDirectory(prefix="ohmv-recovery-", dir="/tmp") as ipc,
            OwnedService(
                command(args, Path(ipc), port),
                environment,
                directory,
                gpu,
                args.stage_timeout,
            ) as service,
        ):
            ready = wait_snapshot(
                wire,
                service,
                args.startup_timeout,
                "ready",
                lambda s: s.get("status") == "running" and s.get("worker"),
            )
            if ready["epoch"] != 1 or ready["restarts"] != 0:
                raise ValueError("fresh service restarted during initial loading")
            for index in range(2):
                status, result = wire.request(
                    f"warm-{index}", "/v1/chat/completions", body("chat", prompt)
                )
                if status != 200 or not response_text(result, "chat"):
                    raise ValueError("shared-prefix warmup did not complete")
                if result["usage"]["prompt_tokens"] < 1568 or (
                    index == 1 and cached_tokens(result) < 1568
                ):
                    raise ValueError(
                        "warmup did not prove reuse of at least 1568 prefix tokens"
                    )
            before = wait_idle(
                wire, service, args.drain_timeout, "warm-idle", time.time(), 0
            )
            old_process = _process_record(before["worker_pid"])
            if old_process is None or old_process["pgid"] != old_process["pid"]:
                raise ValueError("owned worker has no isolated process-group identity")
            recovery_seen, observer_errors, stop = [], [], threading.Event()

            def observe():
                try:
                    index = 0
                    while not stop.is_set():
                        sample = snapshot(wire, f"fault-watch-{index:04d}", service)
                        if sample["status"] == "recovering":
                            status, rejection = wire.request(
                                "recovering-rejection",
                                "/v1/chat/completions",
                                body("chat", "Reply OK."),
                            )
                            recovery_seen.append(
                                {
                                    "snapshot": sample,
                                    "http_status": status,
                                    "response": rejection,
                                }
                            )
                            return
                        index += 1
                        stop.wait(0.05)
                except BaseException as error:
                    observer_errors.append(error)

            observer = threading.Thread(
                target=observe, name="fault-recovery-observer", daemon=True
            )
            observer.start()
            try:
                fault_prompt = (
                    prompt
                    if stage != "graph_capture"
                    else shared * 2
                    + "\nList consecutive integers starting at 1, one per line."
                )
                fault = wire.stream(
                    "fault",
                    body("chat", fault_prompt, streaming=True, maximum=2048),
                    marker,
                    stage,
                    service.check,
                )
                deadline = time.monotonic() + 10
                while (
                    not recovery_seen
                    and not observer_errors
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                if observer_errors:
                    raise RuntimeError("recovery observer failed") from observer_errors[
                        0
                    ]
                if len(recovery_seen) != 1 or recovery_seen[0]["http_status"] != 503:
                    raise ValueError(
                        "recovery did not preserve HTTP and reject generation with 503"
                    )
            finally:
                stop.set()
                observer.join(timeout=args.request_timeout + 5)
                if observer.is_alive():
                    raise RuntimeError("recovery HTTP observer did not exit")
            after = wait_idle(
                wire, service, args.recovery_timeout, "recovered-idle", time.time(), 1
            )
            old_live = live_group(old_process["pgid"])
            validate_replacement(before, after, old_process, old_live)
            checks = []
            for name, expected in (("cold-prefix", 0), ("hot-prefix", None)):
                status, result = wire.request(
                    name, "/v1/chat/completions", body("chat", prompt)
                )
                hit = cached_tokens(result)
                if (
                    status != 200
                    or not response_text(result, "chat")
                    or (expected == 0 and hit != 0)
                    or (expected is None and hit < 1568)
                ):
                    raise ValueError(
                        "post-recovery shared prefix did not progress from cold to hot"
                    )
                checks.append(
                    {"name": name, "cached_tokens": hit, "response_id": result["id"]}
                )
            for api, path in (
                ("chat", "/v1/chat/completions"),
                ("responses", "/v1/responses"),
            ):
                status, result = wire.request(
                    "normal-" + api,
                    path,
                    body(api, "What is 12 plus 13? Reply with only the integer."),
                )
                if status != 200:
                    raise ValueError("post-recovery API request failed")
                validate_arithmetic_answer(response_text(result, api))
            final = wait_idle(
                wire, service, args.drain_timeout, "final-idle", time.time(), 1
            )
            if (
                final["epoch"] != after["epoch"]
                or final["worker_pid"] != after["worker_pid"]
            ):
                raise ValueError("worker changed again after expected recovery")
            log = (directory / "server.log").read_text()
            validate_oom(log)
            observed = observed_configuration(log)
            validate_configuration(observed, args.mode, args.source)
            if args.mode == "dspark":
                lines = [
                    line
                    for line in log.splitlines()
                    if "DSPARK_CHECKPOINT_PROVENANCE " in line
                ]
                if len(lines) != 2:
                    raise ValueError(
                        "expected DSpark provenance for exactly two workers"
                    )
                summary["loaded_draft"] = [
                    loaded_draft_provenance(line, args.draft_identity) for line in lines
                ]
            metrics = read_metrics(directory / "server.log")
            failed = [record for record in metrics if record["status"] != "completed"]
            if (
                len(failed) != 1
                or failed[0]["request_id"] != fault["request_id"]
                or failed[0]["response_id"] != fault["response_id"]
                or failed[0]["status"] != "failed"
            ):
                raise ValueError(
                    "fault was replayed, discarded, or another request failed"
                )
            expected_ids = {
                int(record["response"]["id"].rsplit("_", 1)[1])
                for record in wire.records
                if record.get("status") == 200
                and isinstance(record.get("response"), dict)
                and "id" in record["response"]
            }
            completed_ids = {
                record["request_id"]
                for record in metrics
                if record["status"] == "completed"
            }
            if completed_ids != expected_ids or len(completed_ids) != 6:
                raise ValueError(
                    "completed metric identities do not match all six client requests"
                )
            summary.update(
                passed=True,
                before=before,
                after=after,
                final=final,
                old_worker=old_process,
                old_group_live=old_live,
                fault=fault,
                recovery_observation=recovery_seen[0],
                prefix_checks=checks,
                metrics=metrics,
                observed=observed,
            )
            if any(
                file_identity(directory / name) != identity
                for name, identity in summary["launcher"].items()
            ):
                raise ValueError("owned fault launcher changed during execution")
    except BaseException as error:
        summary["passed"] = False
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        summary["files"] = {
            str(path.relative_to(directory)): file_identity(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "summary.json"
        }
        write_json(directory / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mtp", "dspark"), required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--model", type=Path, default=os.environ.get("OH_MY_VLLM_MODEL")
    )
    parser.add_argument(
        "--draft-model", type=Path, default=os.environ.get("OH_MY_VLLM_DRAFT_MODEL")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--recovery-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--drain-timeout", type=float, default=90)
    parser.add_argument("--stage-timeout", type=float, default=3600)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if args.model is None or (args.mode == "dspark" and args.draft_model is None):
        raise ValueError("target model and DSpark draft paths must be explicit")
    if len(set(args.stages)) != len(args.stages):
        raise ValueError("fault stages must be unique")
    if any(
        not 0 < getattr(args, name) <= 7200
        for name in (
            "startup_timeout",
            "recovery_timeout",
            "request_timeout",
            "drain_timeout",
            "stage_timeout",
        )
    ):
        raise ValueError("recovery timeouts must be positive and bounded")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == ROOT or ROOT in args.output_dir.parents:
        raise ValueError("raw recovery evidence must be outside the repository")
    args.binary = args.binary.resolve(strict=True)
    return args


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "kind": "target_allocator_oom_recovery",
        "mode": args.mode,
        "diagnostic": args.diagnostic,
        "passed": False,
        "stages": [],
    }
    write_json(args.output_dir / "summary.json", summary)
    previous = {}
    cancelled = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)

        def terminate(signum, frame):
            if not cancelled:
                cancelled.append(signum)
                raise InterruptedError("recovery collector was cancelled")

        signal.signal(signum, terminate)
    try:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not gpu.startswith("GPU-") or "," in gpu:
            raise ValueError(
                "run recovery acceptance under the idle UUID-pinned with-gpu wrapper"
            )
        source = runtime_source_identity(args.binary)
        if source["git_status"] and not args.diagnostic:
            raise ValueError(
                "formal recovery acceptance requires committed, unchanged source"
            )
        summary["source"] = source
        args.source = source
        summary["fixture"] = file_identity(FIXTURE)
        summary["model"] = checkpoint_identity(args.model)
        if args.mode == "dspark":
            summary["draft_model"] = checkpoint_identity(args.draft_model)
            args.draft_identity = summary["draft_model"]
        summary["hardware"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + gpu,
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
        if "B200" not in summary["hardware"]:
            raise ValueError("recovery acceptance requires one B200")
        summary["runtime"] = {
            "python": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "pyzmq", "msgpack")
            },
        }
        with _child_subreaper():
            for stage in args.stages:
                result = run_stage(args, args.output_dir / stage, stage, gpu)
                summary["stages"].append(result)
                write_json(args.output_dir / "summary.json", summary)
                if cancelled:
                    raise InterruptedError("recovery collector was cancelled")
        if (
            runtime_source_identity(args.binary) != source
            or file_identity(FIXTURE) != summary["fixture"]
            or checkpoint_identity(args.model) != summary["model"]
            or (
                args.mode == "dspark"
                and checkpoint_identity(args.draft_model) != summary["draft_model"]
            )
        ):
            raise ValueError(
                "source, binary, fixture, or checkpoint changed during run"
            )
        summary["all_stages_passed"] = all(
            stage["passed"] for stage in summary["stages"]
        )
        summary["coverage_complete"] = set(args.stages) == set(STAGES)
        summary["passed"] = (
            summary["all_stages_passed"]
            and summary["coverage_complete"]
            and not args.diagnostic
        )
    except BaseException as error:
        summary["passed"] = False
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        summary["files"] = {
            str(path.relative_to(args.output_dir)): file_identity(path)
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file() and path != args.output_dir / "summary.json"
        }
        write_json(args.output_dir / "summary.json", summary)
    return summary


def main():
    summary = run(parse_args())
    print(
        json.dumps(
            {key: summary[key] for key in ("mode", "passed", "diagnostic")},
            allow_nan=False,
        )
    )
    if not summary["passed"] and not (
        summary.get("all_stages_passed") and summary["diagnostic"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
