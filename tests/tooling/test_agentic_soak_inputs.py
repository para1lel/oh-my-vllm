"""Changed staged read-only inputs must fail a translation collection."""

import argparse
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchmarks import agentic_soak as soak
from benchmarks import agentic_translation as translation
from benchmarks.agentic_corpus import MANIFEST_NAME, STAGED_DIRECTORY, Corpus
from tests.support.corpus import synthetic_book


class Blocker(Exception):
    """End this stub lane after the collector observed its staged inputs."""


def collector_result(tmp_path, monkeypatch, *, mutate: str | None):
    directory, identity, _manifest = synthetic_book(tmp_path)
    corpus = Corpus(directory, identity)
    metrics = []
    transport_calls = []

    def lane_transport(lane, *, explained=None):
        transport_calls.append({"lane": lane.name, "explained": explained})
        return {"stub": True}

    class Client:
        def __init__(
            self, omp, api, lane, lane_directory, proxy, stop, timeout, **kwargs
        ):
            self.lane, self.api, self.session_id = lane, api, f"session-{lane}"
            self.workspace = lane_directory / "workspace"
            self.workspace.mkdir()
            self.workspace_manifest = translation.prepare_workspace(
                self.workspace, kwargs["corpus"]
            )
            self.turns = [{"completed_monotonic_s": time.monotonic()}]
            self.process = threading.current_thread()
            if mutate == "book":
                staged = self.workspace / STAGED_DIRECTORY / "book.txt"
                staged.write_bytes(b"x" * staged.stat().st_size)
            elif mutate == "manifest":
                staged = self.workspace / STAGED_DIRECTORY / MANIFEST_NAME
                staged.write_text("{}")
            elif mutate == "specification":
                (self.workspace / "TASK.md").write_text("changed")

        def initialize(self):
            pass

        def context(self):
            return {"lane": self.lane}

        def collect(self):
            return {
                "lane": self.lane,
                "api": self.api,
                "workload": "translation",
                "session_id": self.session_id,
                "turns": 1,
                "process": {"pid": os.getpid()},
                "entry_count": 0,
                "workspace_path": str(self.workspace),
                "workspace": self.workspace_manifest,
            }

        def close(self):
            pass

    class Guard:
        def __init__(self, *args):
            self.identity = {}

        def check(self, *args):
            pass

        def close(self):
            pass

    def run_goal(client, clock, duration, objective):
        assert "corpus" in objective(0)
        return {
            "collection_deadline_monotonic_s": clock["monotonic"] + duration,
            "collection_deadline_unix_s": clock["unix"] + duration,
            "controller_ended_monotonic_s": clock["monotonic"] + duration,
            "final_paused_tail_seconds": 0.0,
        }

    @contextmanager
    def proxy(*args, **kwargs):
        yield "stub", []

    def audit(directory, session, *args):
        return [], {"response_ids": [], "cancelled_requests": 0, "outcomes": []}

    def identity_of(path):
        if str(path).startswith("/proc/"):
            return {"sha256": "stub"}
        return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    for name, value in {
        "RpcClient": Client,
        "ContextGuard": Guard,
        "disk_proxy": proxy,
        "run_goal": run_goal,
        "resource_sample": lambda *args, **kwargs: {"final_sample": True},
        "wait_idle": lambda *args, **kwargs: {
            "epoch": 0,
            "server_pid": os.getpid(),
            "worker_pid": os.getpid(),
        },
        "file_identity": identity_of,
        "runtime_source_identity": lambda path: {
            "git_status": "",
            "binary_sha256": "stub",
        },
        "observed_configuration": lambda log: {
            "speculative_mode": "mtp",
            "max_model_len": 262144,
        },
        "collector_environment": lambda omp: {},
        "server_identity": lambda *args: {},
        "read_metrics": lambda *args: metrics,
        "audit_session": audit,
        "audit_framework_session": audit,
        "Corpus": lambda path: corpus,
        "lane_transport_evidence": lane_transport,
    }.items():
        monkeypatch.setattr(soak, name, value)
    monkeypatch.setattr(
        "benchmarks.agentic_sandbox.SandboxImage", lambda *a, **k: Guard()
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("stub\n")
    args = argparse.Namespace(
        output_dir=tmp_path / "evidence",
        diagnostic=True,
        duration=0.5,
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
        workload="translation",
        corpus=directory,
    )
    return soak.run(args), identity, transport_calls


def test_intact_staged_inputs_allow_an_operational_pass(tmp_path, monkeypatch):
    result, _identity, transport_calls = collector_result(
        tmp_path, monkeypatch, mutate=None
    )
    assert result.get("operational_passed") is True, result
    assert result["workload"] == "translation"
    assert result["corpus"]["manifest_sha256"]
    assert result["corpus_unchanged"] is True
    assert result["workload_specification_unchanged"] is True
    assert result["transport_audit"] == [
        {"lane": lane, "stub": True} for lane in range(4)
    ]
    assert transport_calls == [
        {"lane": f"lane-{lane}", "explained": {"model": {}}} for lane in range(4)
    ]


@pytest.mark.parametrize("mutate", ["book", "manifest", "specification"])
def test_changed_staged_inputs_cannot_pass_the_collection(
    tmp_path, monkeypatch, mutate
):
    result, _identity, _transport = collector_result(
        tmp_path, monkeypatch, mutate=mutate
    )
    assert not result.get("operational_passed"), result
    assert "operational_passed" not in result or result["operational_passed"] is False
    expected = (
        "lane specification changed during collection"
        if mutate == "specification"
        else "corpus manifest differs from the retained download record"
        if mutate == "manifest"
        else "accepted book bytes differ from the retained identity"
    )
    assert any(expected in error for error in result["errors"]), result["errors"]
    assert result["passed"] is False
