"""CPU checks for heavy-soak evidence; these do not execute real OMP or GPUs."""

import json
import math
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks.agentic_soak import build_parser, run
from benchmarks.agentic_soak_analysis import (
    bind_points,
    device_used_bytes,
    plateau,
    read_metrics,
    request_metric,
    resource_checks,
)
from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy, load_requests
from benchmarks.agentic_soak_support import Evidence, run_omp


def metric(rid=7, *, status="completed", api="chat", tokens=20):
    return {
        "request_id": rid,
        "response_id": f"{'chatcmpl' if api == 'chat' else 'resp'}_test_{rid}",
        "api": api,
        "status": status,
        "received_unix_s": 100.0,
        "processing_started_unix_s": 102.0,
        "generation_completed_unix_s": 105.0,
        "output_tokens": tokens,
        "queue_seconds": 2.0,
        "processing_seconds": 3.0,
    }


def task(rid=7, *, api="chat"):
    return {
        "task_id": "task",
        "lane": 3,
        "api": api,
        "passed": True,
        "requests": [
            {
                "method": "POST",
                "response_status": 200,
                "response_ids": [metric(rid, api=api)["response_id"]],
            }
        ],
    }


def sample(now=100.0):
    return {
        "status": "running",
        "restarts": 0,
        "epoch": 0,
        "sampled_unix_s": now,
        "worker_snapshot_unix_s": now,
        "collected_monotonic_s": now,
        "phase": "normal",
        "observation": "drained",
        "active_requests": 0,
        "pending_admissions": 0,
        "scheduler_running": 0,
        "scheduler_waiting": 0,
        "queued_reservations": 0,
        "live_reservations": 0,
        "completed_stream_drains": 0,
        "stored_responses": 5,
        "stored_response_bytes": 1000,
        "request_state_bytes": 0,
        "request_state_max_bytes": 100_000,
        "limits": {
            "live": 4,
            "queued": 4,
            "response_capacity": 10,
            "response_max_bytes": 10000,
        },
        "cpu": {"rss_bytes": 20_000_000},
        "kv": {
            "fa_blocks": 100,
            "fa_free": 99,
            "fa_cached": 10,
            "gdn_blocks": 10,
            "gdn_free": 9,
            "gdn_cached": 0,
        },
        "worker": {
            "requests": {"histories": 0, "samplers": 0},
            "graphs": {
                "target": {
                    "resident": 4,
                    "capacity": 16,
                    "probation": 0,
                    "probation_capacity": 32,
                    "capture_history": 3,
                    "capture_history_capacity": 16,
                },
                "dspark": None,
            },
            "compile": {
                "selection_entries": 0,
                "selection_capacity": 4,
                "rewrite_entries": 2,
                "rewrite_capacity": 4,
                "code_cache_entries": 3,
                "code_cache_capacity": 4,
                "code_objects": 2,
                "code_object_capacity": 4,
                "module_entries": 3,
                "module_capacity": 4,
                "compiled_graph_counts": {"unit": 2},
                "unit_capacity": 4,
                **{
                    f"{name}_{suffix}": value
                    for name in ("fx_source", "fake_tensor", "fx_profiler")
                    for suffix, value in (
                        ("entries", 2),
                        ("capacity", 4),
                        ("bytes", 2048),
                        ("byte_capacity", 4096),
                    )
                },
            },
            "gpu": {
                "allocated_bytes": 10_000_000,
                "reserved_bytes": 20_000_000,
                "total_bytes": 100_000_000,
                "free_bytes": 70_000_000,
                "device_used_bytes": 30_000_000,
            },
            "cpu": {"rss_bytes": 200_000_000},
            "preparation": {"pending": 0, "cancelled": 0},
        },
    }


def test_processing_metric_excludes_queue_and_binds_every_request():
    result = bind_points([request_metric(metric())], [task()], 100.0)
    assert result[0]["elapsed_minutes"] == pytest.approx(2 / 60)
    assert result[0]["output_tps"] == pytest.approx(20 / 3)
    assert result[0]["queue_seconds"] == 2
    assert result[0]["lane"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_tokens", True),
        ("output_tokens", -1),
        ("queue_seconds", math.nan),
        ("processing_seconds", 0),
        ("processing_started_unix_s", None),
        ("received_unix_s", math.inf),
        ("request_id", True),
        ("response_id", "resp_test_7"),
    ],
)
def test_invalid_completed_metric_rejected(field, value):
    value = metric() | {field: value}
    with pytest.raises(ValueError):
        request_metric(value)


def test_wall_clock_adjustment_does_not_replace_monotonic_duration():
    # NTP can change UTC. The server-provided monotonic duration stays authoritative.
    value = metric() | {"generation_completed_unix_s": 99.0}
    point = bind_points([request_metric(value)], [task()], 100.0)[0]
    assert point["output_tps"] == pytest.approx(20 / 3)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "cancelled", "failed_task", "http_error"]
)
def test_failed_or_missing_normal_requests_are_not_discarded(mutation):
    metrics, tasks = [metric()], [task()]
    if mutation == "missing":
        metrics = []
    elif mutation == "duplicate":
        tasks.append(task())
    elif mutation == "cancelled":
        metrics[0]["status"] = "cancelled"
    elif mutation == "failed_task":
        tasks[0]["passed"] = False
    elif mutation == "http_error":
        tasks[0]["requests"][0]["response_status"] = 503
    with pytest.raises(ValueError):
        bind_points(metrics, tasks, 100)


def test_metric_log_duplicate_and_truncation_fail(tmp_path):
    path = tmp_path / "server.log"
    line = "INFO SERVING_REQUEST_METRIC " + json.dumps(metric()) + "\n"
    path.write_text("old log\n" + line)
    assert read_metrics(path, len("old log\n"))[0]["request_id"] == 7
    path.write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_metrics(path)
    with pytest.raises(ValueError, match="truncated"):
        read_metrics(path, path.stat().st_size + 1)


@pytest.mark.parametrize("name", ["fx_source", "fake_tensor", "fx_profiler"])
@pytest.mark.parametrize(
    "suffix,cap", [("entries", "capacity"), ("bytes", "byte_capacity")]
)
@pytest.mark.parametrize(
    "mutation",
    ["missing", "over_capacity", "negative", "boolean", "fraction", "zero_budget"],
)
def test_compiler_metadata_requires_complete_and_bounded_evidence(
    name, suffix, cap, mutation
):
    healthy = sample()
    compiled = healthy["worker"]["compile"]
    key, cap_key = f"{name}_{suffix}", f"{name}_{cap}"
    if mutation == "missing":
        del compiled[key]
    elif mutation == "over_capacity":
        compiled[key] = compiled[cap_key] + 1
    elif mutation == "zero_budget":
        compiled[cap_key] = 0
    else:
        compiled[key] = {"negative": -1, "boolean": True, "fraction": 1.5}[mutation]
    checks = {check["name"]: check for check in resource_checks(healthy)}
    assert not checks[f"compile_{name}_{suffix}"]["passed"]


def test_all_caps_and_idle_resources_checked():
    healthy = sample()
    assert all(check["passed"] for check in resource_checks(healthy, require_idle=True))
    healthy["worker"]["requests"]["histories"] = 1
    assert any(
        check["name"] == "idle_worker_histories" and not check["passed"]
        for check in resource_checks(healthy, require_idle=True)
    )
    healthy["worker"]["graphs"]["target"]["resident"] = 17
    assert any(
        check["name"] == "graph_target_resident" and not check["passed"]
        for check in resource_checks(healthy)
    )
    healthy["stored_response_bytes"] = 10001
    assert any(
        check["name"] == "stored_response_bytes" and not check["passed"]
        for check in resource_checks(healthy)
    )
    healthy["restarts"] = 1
    assert any(
        check["name"] == "no_worker_restart" and not check["passed"]
        for check in resource_checks(healthy)
    )


@pytest.mark.parametrize("pool", ["fa", "gdn"])
def test_idle_kv_references_released_without_discarding_cached_prefixes(pool):
    row = sample()
    assert all(check["passed"] for check in resource_checks(row, require_idle=True))
    row["kv"][pool + "_free"] -= 1
    assert all(check["passed"] for check in resource_checks(row))
    assert any(
        check["name"] == "idle_kv_" + pool + "_references" and not check["passed"]
        for check in resource_checks(row, require_idle=True)
    )


def test_plateau_uses_post_warm_first_and_last_half():
    samples = []
    for index in range(16):
        row = sample(index * 300)
        row["worker"]["gpu"]["allocated_bytes"] = 10_000_000 + index * 100
        samples.append(row)
    options = {"absolute_bytes": 1000, "normal_started": 0, "normal_ended": 4500}
    result = plateau(samples, ("worker", "gpu", "allocated_bytes"), **options)
    assert result["passed"]
    assert result["initial_late_median_bytes"] == 10_000_950
    assert result["final_late_median_bytes"] == 10_001_350
    assert result["theil_sen_bytes_per_minute"] == 20
    for row in samples[12:]:
        row["worker"]["gpu"]["allocated_bytes"] += 1_000_000
    assert not plateau(samples, ("worker", "gpu", "allocated_bytes"), **options)[
        "passed"
    ]
    assert not plateau(samples[:3], ("worker", "gpu", "allocated_bytes"), **options)[
        "passed"
    ]


def test_short_diagnostic_cannot_be_formal(tmp_path):
    args = build_parser().parse_args(
        [
            "--mode",
            "mtp",
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--server-log",
            str(tmp_path / "server.log"),
            "--output-dir",
            str(tmp_path / "raw"),
            "--duration",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="formal soak"):
        run(args)
    assert not (tmp_path / "raw").exists()


def test_four_slots_bound_simultaneous_upstream_calls():
    slots = ModelSlots()
    active, peak = 0, 0
    lock = threading.Lock()

    def work():
        nonlocal active, peak
        with slots.slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == slots.peak == 4
    assert slots.active == 0
    with pytest.raises(ValueError):
        ModelSlots(8)


def test_disk_proxy_preserves_wire_bodies_and_stream_identity(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.server.body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"id":"chatcmpl_test_7","choices":'
                b'[{"index":0,"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        directory = tmp_path / "wire"
        body = b'{"messages":[{"role":"user","content":"read real code"}]}'
        with disk_proxy(
            f"http://127.0.0.1:{server.server_port}/v1", directory, ModelSlots()
        ) as (url, _):
            request = urllib.request.Request(
                url + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                content = response.read()
        records = load_requests(directory)
        assert server.body == body
        assert (directory / records[0]["request_file"]).read_bytes() == body
        assert (directory / records[0]["response_file"]).read_bytes() == content
        assert records[0]["response_ids"] == ["chatcmpl_test_7"]
        assert records[0]["request"]["messages"][0]["content"] == "read real code"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_omp_timeout_stops_its_process(tmp_path):
    output = tmp_path / "pid.txt"
    command = [
        sys.executable,
        "-c",
        "import os,time,pathlib; "
        f"pathlib.Path({str(output)!r}).write_text(str(os.getpid())); time.sleep(60)",
    ]
    state = tmp_path / "state"
    state.mkdir()
    code = run_omp(command, state, tmp_path, threading.Event(), 0.1)
    assert code == 124
    assert not Path("/proc", output.read_text()).exists()
    process = json.loads((tmp_path / "process.json").read_text())
    assert process["pid"] == int(output.read_text())
    assert process["reaped"] is True
    assert process["returncode"] < 0


def test_raw_evidence_append_and_atomic_summary(tmp_path):
    evidence = Evidence(tmp_path)
    evidence.append("events.jsonl", {"passed": False})
    evidence.append("events.jsonl", {"passed": True})
    evidence.write("summary.json", {"passed": False})
    evidence.write("summary.json", {"passed": True})
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "summary.json").read_text())["passed"] is True
    assert not (tmp_path / "summary.json.tmp").exists()


def test_real_scatter_export_preserves_all_input_points(tmp_path):
    from benchmarks.agentic_soak_analysis import plot_points

    points = bind_points(
        [metric(7), metric(8, api="responses")],
        [task(7), task(8, api="responses")],
        100.0,
    )
    plot_points(points, "mtp", tmp_path)
    svg = (tmp_path / "mtp-agentic-soak.svg").read_text()
    assert "2 completed requests" in svg
    assert "processing start" in svg
    assert (tmp_path / "mtp-agentic-soak.png").stat().st_size > 1000


@pytest.mark.parametrize("free", [-1, 100_000_001, True, math.nan])
def test_device_usage_rejects_invalid_free_memory(free):
    row = sample()
    row["worker"]["gpu"]["free_bytes"] = free
    with pytest.raises(ValueError, match="HBM"):
        device_used_bytes(row)
    checks = {check["name"]: check for check in resource_checks(row)}
    assert checks["gpu_device_usage"]["passed"] is False


def test_vendor_memory_growth_is_visible_with_stable_torch_counters():
    samples = [sample(float(i)) for i in range(12)]
    for i, row in enumerate(samples):
        gpu = row["worker"]["gpu"]
        gpu["free_bytes"] -= i * 4_000_000
        gpu["device_used_bytes"] = device_used_bytes(row)
    options = {"absolute_bytes": 1_000_000, "normal_started": 0, "normal_ended": 11}
    assert plateau(samples, ("worker", "gpu", "allocated_bytes"), **options)["passed"]
    assert not plateau(samples, ("worker", "gpu", "device_used_bytes"), **options)[
        "passed"
    ]


def test_memory_plot_labels_separate_device_and_torch_scopes(tmp_path):
    from benchmarks.agentic_soak_analysis import plot_memory

    rows = [
        sample(101 + i) | {"phase": "normal", "observation": "drained"}
        for i in range(6)
    ]
    plot_memory(rows, "dspark", tmp_path, 100)
    svg = (tmp_path / "dspark-agentic-memory.svg").read_text()
    for label in (
        "Device used (total - free)",
        "Torch allocated",
        "Torch reserved",
        "Drained snapshot",
        "GPU worker RSS",
        "HTTP server RSS",
    ):
        assert label in svg
    assert (tmp_path / "dspark-agentic-memory.png").stat().st_size > 1000


def test_idle_check_waits_for_zero_state_and_a_fresh_worker_snapshot(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import benchmarks.agentic_soak_support as soak

    snapshots = [sample(99.0), sample(100.0), sample(101.0)]
    snapshots[0]["sampled_unix_s"] = 200.0
    snapshots[1]["worker"]["requests"]["histories"] = 1
    monkeypatch.setattr(soak, "diagnostics", lambda _url: snapshots.pop(0))

    class ImmediateStop:
        def is_set(self):
            return False

        def wait(self, _duration):
            return False

    args = SimpleNamespace(base_url="unused", drain_timeout=1)
    final = soak.wait_idle(args, Evidence(tmp_path), ImmediateStop(), 100.0)
    assert final["worker_snapshot_unix_s"] == 101
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "idle.jsonl").read_text().splitlines()) == 1


def test_probe_failures_remain_in_raw_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import benchmarks.agentic_soak_support as soak

    def broken(*_args, **_kwargs):
        raise ConnectionError("test service unavailable")

    monkeypatch.setattr(soak.urllib.request, "urlopen", broken)
    args = SimpleNamespace(base_url="http://127.0.0.1:1/v1", slow_pause=1)
    with pytest.raises(ConnectionError):
        soak.lifecycle_probe(args, Evidence(tmp_path), "chat", "cancel", 0)
    record = json.loads((tmp_path / "probes.jsonl").read_text())
    assert record["passed"] is False
    assert "service unavailable" in record["error"]


def test_cumulative_compile_counts_have_a_label_count_budget():
    row = sample()
    row["worker"]["compile"]["compiled_graph_counts"] = {"unit": 1000000}
    assert all(check["passed"] for check in resource_checks(row))
    row["worker"]["compile"]["compiled_graph_counts"] = {
        str(key): 1 for key in range(5)
    }
    assert any(
        check["name"] == "compile_unit_label_capacity" and not check["passed"]
        for check in resource_checks(row)
    )


def test_proxy_close_cancels_slot_waiter_and_joins_before_evidence_read(tmp_path):
    slots = ModelSlots()
    for _ in range(4):
        slots.semaphore.acquire()
    errors = []
    client = None
    directory = tmp_path / "wire"
    started = time.monotonic()
    try:
        with disk_proxy("http://127.0.0.1:9/v1", directory, slots) as (url, records):

            def request():
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(
                            url + "/chat/completions",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=5,
                    ) as response:
                        response.read()
                except Exception as error:
                    errors.append(type(error).__name__)

            client = threading.Thread(target=request)
            client.start()
            deadline = time.monotonic() + 3
            while not (directory / "0000-request.json").exists():
                if time.monotonic() > deadline:
                    pytest.fail("client did not reach the blocked proxy slot")
                time.sleep(0.01)
        client.join(timeout=2)
        assert not client.is_alive()
        assert errors
        assert len(records) == 1
        assert "InterruptedError" in records[0]["error"]
        assert "upstream_started_unix_s" not in records[0]
        before = (directory / "http.jsonl").read_bytes()
        for _ in range(4):
            slots.semaphore.release()
        time.sleep(0.15)
        assert (directory / "http.jsonl").read_bytes() == before
        assert slots.active == 0
        assert time.monotonic() - started < 3
    finally:
        if client is not None:
            client.join(timeout=6)


@pytest.mark.parametrize("headers_sent", [False, True])
def test_proxy_close_wakes_upstream_header_and_body_reads(tmp_path, headers_sent):
    received, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if headers_sent:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
            received.set()
            release.wait(6)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    client = None
    try:
        with disk_proxy(
            f"http://127.0.0.1:{server.server_port}/v1", tmp_path / "wire", ModelSlots()
        ) as (url, records):

            def request():
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(
                            url + "/chat/completions",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=5,
                    ) as response:
                        response.read()
                except Exception as error:
                    errors.append(str(error))

            client = threading.Thread(target=request)
            client.start()
            assert received.wait(3)
            started = time.monotonic()
        assert time.monotonic() - started < 3
        client.join(timeout=2)
        assert not client.is_alive()
        assert len(records) == 1
        # An interrupted body can be a clean EOF, but it cannot claim a complete
        # response identity and the normal-task binder therefore rejects it.
        assert not records[0]["response_ids"]
        assert (tmp_path / "wire/http.jsonl").exists()
    finally:
        release.set()
        if client is not None:
            client.join(timeout=6)
        server.shutdown()
        server.server_close()
        thread.join()
