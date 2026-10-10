"""CPU checks for the explicit target allocator OOM recovery collector."""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import serving_recovery as recovery
from benchmarks import serving_recovery_analysis as analysis
from benchmarks.common import _child_subreaper, _process_record


def sample():
    return {
        "status": "running",
        "server_pid": 11,
        "worker_pid": 22,
        "epoch": 2,
        "restarts": 1,
        "worker_snapshot_unix_s": 200,
        "active_requests": 0,
        "pending_admissions": 0,
        "scheduler_running": 0,
        "scheduler_waiting": 0,
        "queued_reservations": 0,
        "live_reservations": 0,
        "completed_stream_drains": 0,
        "request_state_bytes": 0,
        "request_state_max_bytes": 536870912,
        "stored_responses": 0,
        "stored_response_bytes": 0,
        "limits": {
            "live": 64,
            "queued": 64,
            "response_capacity": 1000,
            "response_max_bytes": 268435456,
        },
        "kv": {
            "fa_blocks": 1400,
            "fa_free": 1399,
            "fa_cached": 0,
            "gdn_blocks": 128,
            "gdn_free": 127,
            "gdn_cached": 0,
        },
        "worker": {
            "requests": {"histories": 0, "samplers": 0},
            "preparation": {"pending": 0, "cancelled": 0},
            "graphs": {
                "target": {
                    "resident": 1,
                    "capacity": 32,
                    "probation": 0,
                    "probation_capacity": 256,
                    "capture_history": 1,
                    "capture_history_capacity": 4096,
                }
            },
            "compile": {
                "selection_entries": 0,
                "selection_capacity": 4,
                "rewrite_entries": 0,
                "rewrite_capacity": 4,
                "code_cache_entries": 0,
                "code_cache_capacity": 4,
                "module_entries": 0,
                "module_capacity": 4,
                "code_objects": 0,
                "code_object_capacity": 4,
                "compiled_graph_counts": {},
                "unit_capacity": 4,
                **{
                    f"{name}_{suffix}": value
                    for name in ("fx_source", "fake_tensor", "fx_profiler")
                    for suffix, value in (
                        ("entries", 0),
                        ("capacity", 4),
                        ("bytes", 0),
                        ("byte_capacity", 4096),
                    )
                },
            },
            "gpu": {
                "allocated_bytes": 1,
                "reserved_bytes": 2,
                "total_bytes": 3,
                "free_bytes": 1,
            },
        },
    }


def fault(stage="after_commit"):
    return {
        "status": 200,
        "request_id": 5,
        "armed": True,
        "stage": stage,
        "visible_output": True,
        "marker_recreated": True,
        "explicit_errors": 1,
        "success_events": 0,
    }


@pytest.mark.parametrize("stage", recovery.STAGES)
def test_actual_fault_stage_requires_explicit_failure_without_replay(stage):
    recovery.validate_fault_stream(fault(stage))
    row = fault(stage)
    row["visible_output"] = False
    if stage == "graph_capture":
        recovery.validate_fault_stream(row)
    else:
        with pytest.raises(ValueError):
            recovery.validate_fault_stream(row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", 503),
        ("request_id", True),
        ("request_id", 0),
        ("explicit_errors", 0),
        ("explicit_errors", 2),
        ("success_events", 1),
        ("marker_recreated", False),
        ("armed", False),
    ],
)
def test_silent_eof_duplicated_failure_success_and_untriggered_marker_fail(key, value):
    row = fault()
    row[key] = value
    with pytest.raises(ValueError):
        recovery.validate_fault_stream(row)


def test_only_actual_allocator_traceback_passes():
    trace = (
        'Traceback (most recent call last):\nFile "worker_faults.py"\n'
        'torch.empty(free + cached + (1 << 30), device="cuda")\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100 GiB"
    )
    recovery.validate_oom(trace)
    for changed in (
        trace.replace("worker_faults.py", "other.py"),
        trace.replace("torch.empty(free + cached", "fake_allocator("),
        trace.replace("Traceback (most recent call last)", ""),
        trace + "test-owned OOM",
    ):
        with pytest.raises(ValueError):
            recovery.validate_oom(changed)


@pytest.mark.parametrize("text", ["25", " 25\n", "25.", "\n25.\n"])
def test_post_recovery_arithmetic_accepts_only_the_expected_integer(text):
    recovery.validate_arithmetic_answer(text)


@pytest.mark.parametrize("text", ["125", "250", "25 25", "25..", "Answer: 25", ""])
def test_post_recovery_arithmetic_rejects_substrings_and_mixed_output(text):
    with pytest.raises(ValueError, match="exactly 25"):
        recovery.validate_arithmetic_answer(text)


def test_replacement_requires_exactly_one_new_epoch_and_empty_logical_cache(
    monkeypatch,
):
    before = sample()
    before.update(worker_pid=20, epoch=1, restarts=0)
    old = {"pid": 20, "pgid": 20, "start_ticks": 100}
    monkeypatch.setattr(
        analysis, "_process_record", lambda pid: {"pid": pid, "start_ticks": 200}
    )
    recovery.validate_replacement(before, sample(), old, [])
    for key, value in (
        ("epoch", 3),
        ("restarts", 2),
        ("server_pid", 12),
        ("worker_pid", 20),
    ):
        row = sample()
        row[key] = value
        with pytest.raises(ValueError):
            recovery.validate_replacement(before, row, old, [])
    row = sample()
    row["kv"]["fa_cached"] = 1
    with pytest.raises(ValueError, match="not empty"):
        recovery.validate_replacement(before, row, old, [])
    with pytest.raises(ValueError):
        recovery.validate_replacement(before, sample(), old, [{"pid": 21}])


def test_expected_recovery_never_relaxes_other_idle_bounds():
    assert all(check["passed"] for check in recovery.idle_checks(sample(), 1))
    for owner in ("request_state_bytes", "live_reservations", "scheduler_waiting"):
        row = sample()
        row[owner] = 1
        assert not all(check["passed"] for check in recovery.idle_checks(row, 1))
    row = sample()
    row["worker"]["requests"]["histories"] = 1
    assert not all(check["passed"] for check in recovery.idle_checks(row, 1))
    assert not all(check["passed"] for check in recovery.idle_checks(sample(), 2))


def test_fresh_worker_snapshot_is_required_after_drain(monkeypatch):
    rows = [sample(), sample(), sample()]
    rows[0]["status"] = "recovering"
    rows[0]["worker_snapshot_unix_s"] = None
    rows[1]["worker_snapshot_unix_s"] = 99
    rows[1]["sampled_unix_s"] = 1000
    predicates = []

    def wait(wire, service, timeout, label, predicate):
        predicates.extend(predicate(row) for row in rows)
        return rows[2]

    monkeypatch.setattr(recovery, "wait_snapshot", wait)
    recovery.wait_idle(None, None, 90, "idle", 100, 1)
    assert predicates == [False, False, True]


def test_external_launcher_preserves_bridge_argv_and_real_oom_api(
    tmp_path, monkeypatch
):
    fixture = tmp_path / "test fixture.py"
    calls = tmp_path / "calls.json"
    fixture.write_text(
        "import json\nfrom pathlib import Path\ndef install_fault(*args):\n"
        f"    Path({str(calls)!r}).write_text(json.dumps(args))\n"
    )
    monkeypatch.setattr(recovery, "FIXTURE", fixture)
    package = tmp_path / "package/oh_my_vllm/worker"
    package.mkdir(parents=True)
    for path in (package.parent / "__init__.py", package / "__init__.py"):
        path.write_text("")
    args_file = tmp_path / "args.json"
    (package / "zmq_bridge.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        f"Path({str(args_file)!r}).write_text(json.dumps(sys.argv))\n"
    )
    owned = tmp_path / "external evidence"
    owned.mkdir()
    marker = owned / "fault marker"
    executable = recovery.launcher(owned, "sampling", marker)
    result = subprocess.run(
        [
            str(executable),
            "-m",
            "oh_my_vllm.worker.zmq_bridge",
            "--socket",
            "ipc://a path",
        ],
        env=os.environ | {"PYTHONPATH": str(package.parent.parent)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(calls.read_text()) == ["sampling", "real_oom", str(marker)]
    assert json.loads(args_file.read_text()) == [
        str(package / "zmq_bridge.py"),
        "--socket",
        "ipc://a path",
    ]


@contextmanager
def stream_server(marker, stage, *, explicit_error=True, delta_id="chatcmpl_owned_5"):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(marker.exists())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"id":"chatcmpl_owned_5","choices":['
                b'{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            )
            self.wfile.flush()
            if stage != "graph_capture":
                self.wfile.write(
                    b"data: "
                    + json.dumps(
                        {
                            "id": delta_id,
                            "choices": [
                                {"delta": {"content": "1"}, "finish_reason": None}
                            ],
                        }
                    ).encode()
                    + b"\n\n"
                )
                self.wfile.flush()
            deadline = time.monotonic() + 2
            while marker.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            if not marker.exists():
                marker.touch()
                if explicit_error:
                    self.wfile.write(b'data: {"error":{"message":"worker failed"}}\n\n')
                    self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.mark.parametrize("stage", recovery.STAGES)
def test_fault_stream_arms_at_the_required_boundary_and_keeps_raw_sse(tmp_path, stage):
    marker = tmp_path / "fault.marker"
    marker.touch()
    with stream_server(marker, stage) as (url, seen):
        wire = recovery.Wire(tmp_path, url, 3)
        result = wire.stream(
            "fault",
            recovery.body("chat", "prompt", streaming=True),
            marker,
            stage,
            lambda: None,
        )
    assert seen == [stage != "graph_capture"]
    assert result["explicit_errors"] == 1
    assert marker.exists()
    assert b'"worker failed"' in (tmp_path / "fault.sse").read_bytes()
    assert (
        result["raw"]["sha256"]
        == recovery.file_identity(tmp_path / "fault.sse")["sha256"]
    )


def test_silent_stream_eof_retains_failed_raw_evidence(tmp_path):
    marker = tmp_path / "fault.marker"
    marker.touch()
    with stream_server(marker, "sampling", explicit_error=False) as (url, _):
        wire = recovery.Wire(tmp_path, url, 3)
        with pytest.raises(ValueError):
            wire.stream(
                "fault",
                recovery.body("chat", "prompt", streaming=True),
                marker,
                "sampling",
                lambda: None,
            )
    record = json.loads((tmp_path / "fault-result.json").read_text())
    assert "error" in record
    assert record["visible_output"] and record["armed"]
    assert record["explicit_errors"] == 0
    assert record["raw"]["bytes"] > 0


def test_changed_stream_identity_fails_and_keeps_the_offending_raw_frame(tmp_path):
    marker = tmp_path / "fault.marker"
    marker.touch()
    with stream_server(marker, "sampling", delta_id="chatcmpl_replayed_6") as (url, _):
        wire = recovery.Wire(tmp_path, url, 3)
        with pytest.raises(ValueError, match="changed its response identity"):
            wire.stream(
                "fault",
                recovery.body("chat", "prompt", streaming=True),
                marker,
                "sampling",
                lambda: None,
            )
    record = json.loads((tmp_path / "fault-result.json").read_text())
    assert record["response_id"] == "chatcmpl_owned_5"
    assert record["request_id"] == 5
    assert not record["armed"]
    assert b"chatcmpl_replayed_6" in (tmp_path / "fault.sse").read_bytes()


def test_owned_service_cleanup_stops_detached_child_and_preserves_unrelated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(recovery, "assert_gpu_exclusive", lambda *a, **k: None)
    child_file = tmp_path / "child.pid"
    script = tmp_path / "server.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'],\n"
        "                       start_new_session=True)\n"
        f"Path({str(child_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(300)\n"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(300)"], start_new_session=True
    )
    try:
        with (
            _child_subreaper(),
            recovery.OwnedService(
                [sys.executable, str(script)],
                os.environ.copy(),
                tmp_path,
                "GPU-test",
                30,
            ) as service,
        ):
            deadline = time.monotonic() + 3
            while not child_file.exists():
                if time.monotonic() > deadline:
                    pytest.fail("owned child did not start")
                time.sleep(0.01)
            child_pid = int(child_file.read_text())
            service.ownership.refresh()
            assert service.ownership.owns(child_pid)
            assert not service.ownership.owns(unrelated.pid)
        assert service.process.returncode is not None
        assert _process_record(child_pid) is None
        assert unrelated.poll() is None
        records = json.loads((tmp_path / "processes.json").read_text())
        assert records["live"] == []
        assert unrelated.pid not in {record["pid"] for record in records["processes"]}
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_failed_attempts_all_stages_and_exclusive_output_are_retained(
    tmp_path, monkeypatch
):
    binary = tmp_path / "binary"
    binary.write_text("binary")
    args = SimpleNamespace(
        output_dir=tmp_path / "raw",
        mode="mtp",
        diagnostic=False,
        stages=list(recovery.STAGES),
        binary=binary,
        model=tmp_path,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-owned")
    monkeypatch.setattr(
        recovery,
        "runtime_source_identity",
        lambda p: {"git_status": "", "git_commit": "head"},
    )
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda *a, **k: "GPU-owned, NVIDIA B200, 180000 MiB, driver",
    )
    monkeypatch.setattr(recovery.importlib.metadata, "version", lambda name: "pinned")
    monkeypatch.setattr(
        recovery, "checkpoint_identity", lambda p: {"tree_sha256": "fixed"}
    )
    calls = []

    def stage(args, directory, name, gpu):
        calls.append(name)
        directory.mkdir()
        (directory / "retained.log").write_text("actual raw failure or success")
        return {"stage": name, "passed": name != "sampling"}

    monkeypatch.setattr(recovery, "run_stage", stage)
    result = recovery.run(args)
    assert calls == list(recovery.STAGES)
    assert result["coverage_complete"]
    assert not result["passed"]
    assert "sampling/retained.log" in result["files"]
    assert (
        json.loads((args.output_dir / "summary.json").read_text())["stages"][2][
            "passed"
        ]
        is False
    )
    with pytest.raises(FileExistsError):
        recovery.run(args)


def test_formal_source_freeze_diagnostic_and_partial_coverage_never_pass(
    tmp_path, monkeypatch
):
    binary = tmp_path / "binary"
    binary.write_text("binary")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-owned")
    monkeypatch.setattr(
        recovery, "runtime_source_identity", lambda p: {"git_status": " M source.py"}
    )
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda *a, **k: "GPU-owned, NVIDIA B200, memory, driver",
    )
    monkeypatch.setattr(recovery.importlib.metadata, "version", lambda name: "pinned")
    monkeypatch.setattr(
        recovery, "checkpoint_identity", lambda p: {"tree_sha256": "fixed"}
    )
    monkeypatch.setattr(recovery, "run_stage", lambda *a: {"passed": True})
    for diagnostic, stages in ((False, list(recovery.STAGES)), (True, ["sampling"])):
        args = SimpleNamespace(
            output_dir=tmp_path / str(diagnostic),
            mode="mtp",
            diagnostic=diagnostic,
            stages=stages,
            binary=binary,
            model=tmp_path,
        )
        result = recovery.run(args)
        assert result["passed"] is False
        if diagnostic:
            assert result["all_stages_passed"] is True
            assert result["coverage_complete"] is False
        else:
            assert "committed" in result["error"]["message"]


def test_parse_requires_models_external_output_and_bounded_timeouts(tmp_path):
    binary = tmp_path / "binary"
    binary.touch()
    arguments = [
        "--mode",
        "mtp",
        "--model",
        str(tmp_path),
        "--binary",
        str(binary),
        "--output-dir",
        str(tmp_path / "raw"),
    ]
    args = recovery.parse_args(arguments)
    assert args.stages == list(recovery.STAGES)
    for extra in (
        ("--stages", "sampling", "sampling"),
        ("--stage-timeout", "nan"),
        ("--startup-timeout", "7201"),
        ("--output-dir", str(recovery.ROOT / ".local/raw")),
    ):
        with pytest.raises(ValueError):
            recovery.parse_args(arguments + list(extra))


def test_command_keeps_full_capacity_and_speculative_mode(tmp_path):
    args = SimpleNamespace(
        binary=Path("binary"),
        model=Path("model"),
        mode="dspark",
        draft_model=Path("draft"),
    )
    command = recovery.command(args, tmp_path, 1234)
    for option, value in (
        ("--num-gpu-blocks", "4200"),
        ("--mamba-blocks", "128"),
        ("--max-model-len", "262144"),
        ("--num-speculative-tokens", "7"),
        ("--draft-model", "draft"),
    ):
        assert command[command.index(option) + 1] == value


def test_observed_bench_config_uses_actual_logical_fa_capacity():
    log = (
        'INFO BENCH_CONFIG speculative_mode="mtp" block_size=784 '
        "max_model_len=262144 num_speculative_tokens=4 "
        "worker_fa_pool_blocks=1400 fa_pool_blocks=1400 "
        'worker_gdn_pool_blocks=128 gdn_pool_blocks=128 rust_sources_sha256="build"'
    )
    observed = recovery.observed_configuration(log)
    recovery.validate_configuration(observed, "mtp", {"rust_sources_sha256": "build"})
    observed["fa_pool_blocks"] = "4200"
    with pytest.raises(ValueError):
        recovery.validate_configuration(
            observed, "mtp", {"rust_sources_sha256": "build"}
        )


def test_ipc_path_is_short_and_removed_independently_of_external_evidence_path():
    args = SimpleNamespace(binary=Path("binary"), model=Path("model"), mode="mtp")
    with recovery.tempfile.TemporaryDirectory(
        prefix="ohmv-recovery-", dir="/tmp"
    ) as ipc:
        command = recovery.command(args, Path(ipc), 1234)
        socket = Path(command[command.index("--socket") + 1])
        assert len(str(socket).encode()) < 80
        socket.touch()
        Path(str(socket) + ".recovery-1").touch()
    assert not Path(ipc).exists()
