"""Late sampler failures must invalidate acceptance before evidence hashing."""

import argparse
import hashlib
import inspect
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchmarks import agentic_soak as soak


@pytest.mark.parametrize(
    "sampler_error,external_cancel", [(True, False), (False, False), (False, True)]
)
def test_sampler_exits_before_acceptance_and_raw_hashing(
    monkeypatch, tmp_path, sampler_error, external_cancel
):
    release_sample = threading.Event()
    sample_entered = threading.Event()
    sampler_exited = threading.Event()
    sampler_stop_holder = {}
    cancelled = threading.Event()
    original_thread = threading.Thread

    class ControlledThread(original_thread):
        """Model a slow observation without spending 20 seconds in this test."""

        def __init__(self, *args, **kwargs):
            target = kwargs.get("target")
            self.is_sampler = getattr(target, "__name__", "") == "sample_memory"
            if self.is_sampler:
                sampler_stop_holder["event"] = inspect.getclosurevars(target).nonlocals[
                    "sampler_stop"
                ]
            super().__init__(*args, **kwargs)

        def run(self):
            try:
                super().run()
            finally:
                if self.is_sampler:
                    sampler_exited.set()

        def join(self, timeout=None):
            if self.is_sampler:
                if timeout is not None:
                    # The old collector lets the sampler outlive this bounded join.
                    return
                # A correct unbounded join permits the pending observation to end.
                release_sample.set()
            super().join(timeout)

    class Client:
        def __init__(self, omp, api, lane, directory, proxy, stop, timeout, **kwargs):
            self.lane = lane
            self.api = api
            self.turns = []
            self.session_id = str(lane)

        def initialize(self):
            assert sample_entered.wait(2)
            if external_cancel:
                cancelled.set()

        def context(self):
            return {"lane": self.lane}

        def prompt(self, text, index):
            time.sleep(0.005)
            self.turns.append({"completed_monotonic_s": time.monotonic()})

        def collect(self):
            return {"lane": self.lane}

        def close(self):
            pass

    class Guard:
        def __init__(self, *args):
            self.identity = {}

        def check(self, *args):
            pass

        def close(self):
            pass

    @contextmanager
    def proxy(*args, **kwargs):
        yield "stub", []

    def resource_sample(url, epoch, evidence):
        sample_entered.set()
        assert sampler_stop_holder["event"].wait(2)
        assert release_sample.wait(2)
        if sampler_error:
            evidence.append("memory.jsonl", {"sampling_error": "late sampler failure"})
            raise ValueError("late sampler failure")
        return {"final_sample": True}

    def idle(args, evidence, stop, after, phase):
        if stop.is_set():
            raise ValueError("sampler failed before drain")
        if phase == "post_run":
            # On the old path, fail after its stop check and final idle observation.
            release_sample.set()
        return {"epoch": 0, "server_pid": os.getpid(), "worker_pid": os.getpid()}

    def identity(path):
        if str(path).startswith("/proc/"):
            return {"sha256": "stub"}
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def audit(directory, session, *args):
        assert sampler_exited.wait(2)
        metric = metrics[session["lane"]]
        return [metric], {
            "response_ids": [metric["response_id"]],
            "cancelled_requests": 0,
        }

    metrics = [
        {
            "response_id": f"resp_{lane}",
            "input_tokens": 10,
            "output_budget": 20,
            "processing_started_unix_s": time.time(),
            "purpose": "task",
            "epoch": 0,
        }
        for lane in range(4)
    ]
    monkeypatch.setattr(soak.threading, "Thread", ControlledThread)
    for name, value in {
        "RpcClient": Client,
        "ContextGuard": Guard,
        "disk_proxy": proxy,
        "resource_sample": resource_sample,
        "wait_idle": idle,
        "curriculum": lambda lane: [],
        "followup": lambda *args: "stub",
        "runtime_source_identity": lambda path: {
            "git_status": "",
            "binary_sha256": "stub",
        },
        "file_identity": identity,
        "observed_configuration": lambda log: {
            "speculative_mode": "mtp",
            "max_model_len": 262144,
        },
        "collector_environment": lambda omp: {},
        "server_identity": lambda *args: {},
        "read_metrics": lambda *args: metrics,
        "audit_session": audit,
        "audit_framework_session": audit,
    }.items():
        monkeypatch.setattr(soak, name, value)
    server_log = tmp_path / "server.log"
    server_log.write_text("stub\n")
    directory = tmp_path / "evidence"
    args = argparse.Namespace(
        output_dir=directory,
        diagnostic=True,
        duration=0.001,
        skip_plots=True,
        turn_timeout=10.0,
        drain_timeout=10.0,
        sample_seconds=30.0,
        model=Path("/stub"),
        binary=Path("/stub"),
        normalizer=Path("/stub"),
        server_log=server_log,
        mode="mtp",
        omp="stub",
        base_url="stub",
        workload="audit",
    )
    result = soak.run(args, cancelled=cancelled) if external_cancel else soak.run(args)
    assert sampler_exited.is_set()
    memory = directory / "memory.jsonl"
    assert memory.exists()
    recorded_hash = next(
        row["sha256"] for row in result["raw_files"] if row["path"] == "memory.jsonl"
    )
    assert recorded_hash == hashlib.sha256(memory.read_bytes()).hexdigest()
    if external_cancel:
        assert cancelled.is_set()
        assert not result.get("operational_passed"), result
        assert any(
            "stopped before normal completion" in error for error in result["errors"]
        )
        assert all(
            "stopped before normal completion" in error or "BrokenBarrierError" in error
            for error in result["errors"]
        )
    elif sampler_error:
        assert not result.get("operational_passed"), result
        assert "resource sampler: late sampler failure" in result["error"]
        assert result["errors"] == ["resource sampler: late sampler failure"]
    else:
        assert result.get("operational_passed"), result
        assert '"final_sample": true' in memory.read_text()
