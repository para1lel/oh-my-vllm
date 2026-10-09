"""REQ-CONTEXT-001: six real B200 boundary rows and fail-closed evidence checks."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import context_boundary as boundary


def _log(batch_size=1, mode="mtp4", **changes):
    row = {
        "batch_size": batch_size,
        "input_len": boundary.INPUT_LEN,
        "output_len": boundary.OUTPUT_LEN,
        "output_tokens": batch_size * boundary.OUTPUT_LEN,
        "preemptions": 0,
        "proposed_draft_tokens": 4 if mode != "ordinary" else 0,
        "accepted_draft_tokens": 2 if mode != "ordinary" else 0,
        "verified_draft_tokens": 3 if mode != "ordinary" else 0,
        "speculative_mode": "dspark"
        if mode == "dspark"
        else "mtp"
        if mode == "mtp4"
        else "none",
        "elapsed_s": 1.0,
        "output_tps": 123.0,
        "steps": 100,
    }
    row.update(changes)
    memory = {
        "message": "GPU memory high water",
        "max_allocated_bytes": 64,
        "max_reserved_bytes": 128,
    }
    return "BENCH_RESULT " + json.dumps(row) + "\n" + json.dumps(memory) + "\n"


@pytest.mark.parametrize("mode", boundary.MODES)
@pytest.mark.parametrize("batch_size", boundary.BATCH_SIZES)
def test_boundary_parser_accepts_complete_rows(mode, batch_size):
    result = boundary.validate_row(_log(batch_size, mode), batch_size, mode, 256)
    assert result["output_tokens"] == batch_size * boundary.OUTPUT_LEN
    assert result["preemptions"] == 0
    assert result["max_reserved_bytes"] == 128
    assert (result["proposed_draft_tokens"] > 0) == (mode != "ordinary")


@pytest.mark.parametrize(
    ("log", "mode", "error"),
    [
        ("", "mtp4", "one BENCH_RESULT"),
        (_log(input_len=131072), "mtp4", "input_len"),
        (_log(output_tokens=1), "mtp4", "output_tokens"),
        (_log(preemptions=1), "mtp4", "preemptions"),
        (_log(proposed_draft_tokens=0), "mtp4", "proposed_draft_tokens"),
        (_log().split("\n", 1)[0], "mtp4", "GPU memory record"),
        (_log() + "CUDA error: out of memory\n", "mtp4", "GPU OOM"),
        (
            _log(mode="ordinary", accepted_draft_tokens=1),
            "ordinary",
            "accepted MTP drafts",
        ),
    ],
)
def test_boundary_parser_rejects_missing_or_false_evidence(log, mode, error):
    with pytest.raises(ValueError, match=error):
        boundary.validate_row(log, 1, mode, 256)


def test_boundary_parser_rejects_impossible_peak_memory():
    with pytest.raises(ValueError, match="peak GPU memory"):
        boundary.validate_row(_log(), 1, "mtp4", 100)


@pytest.mark.parametrize(
    "changes",
    [
        {"verified_draft_tokens": 0},
        {"verified_draft_tokens": 1},
        {"verified_draft_tokens": None},
        {"speculative_mode": "mtp"},
    ],
)
def test_dspark_boundary_rejects_unverified_or_wrong_method(changes):
    with pytest.raises(ValueError):
        boundary.validate_row(_log(mode="dspark", **changes), 1, "dspark", 256)


def test_dspark_boundary_passes_explicit_draft_checkpoint(monkeypatch):
    seen = []
    monkeypatch.setattr(boundary, "hardware_identity", lambda: {"gpu": "GPU-test"})
    monkeypatch.setattr(
        boundary,
        "run_engine",
        lambda command, **_: seen.extend(command) or _log(mode="dspark"),
    )
    row = boundary.run_row(
        Path("/unused/worker"),
        1,
        "dspark",
        model="target",
        draft_model="draft",
        gpu_total_bytes=256,
    )
    assert row["verified_draft_tokens"] == 3
    assert seen[seen.index("--speculative-mode") + 1] == "dspark"
    assert seen[seen.index("--draft-model") + 1] == "draft"


def test_boundary_direct_script_import_smoke():
    result = subprocess.run(
        [
            sys.executable,
            str(boundary.ROOT / "benchmarks/context_boundary.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_boundary_module_import_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.context_boundary", "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=boundary.ROOT,
    )
    assert "--output" in result.stdout


def test_boundary_run_row_inherits_pin_and_uses_shared_process_owner(monkeypatch):
    seen = {}

    def fake_run_engine(command, **kwargs):
        seen.update(command=command, **kwargs)
        return _log(mode="ordinary")

    monkeypatch.setattr(boundary, "run_engine", fake_run_engine)
    monkeypatch.setattr(
        boundary,
        "hardware_identity",
        lambda: {"gpu": "GPU-test", "gpu_info": "GPU-test, B200, 1000 MiB, 1"},
    )
    row = boundary.run_row(
        Path("/unused/binary"),
        1,
        "ordinary",
        model="fixture-model",
        gpu_total_bytes=256,
    )
    assert row["gpu_uuid"] == "GPU-test"
    assert seen["command"][0] == "/unused/binary"
    assert seen["command"][-2:] == ["--repetitions", "1"]
    assert seen["stderr"] == subprocess.STDOUT
    assert seen["log_path"].name == "bench.log"


@pytest.mark.parametrize("termination_signal", [signal.SIGINT, signal.SIGTERM])
def test_boundary_termination_reaches_owned_cleanup(monkeypatch, termination_signal):
    cleanup = []
    previous = signal.getsignal(termination_signal)

    def fake_run_engine(command, **kwargs):
        try:
            os.kill(os.getpid(), termination_signal)
        finally:
            cleanup.append("reaped")

    monkeypatch.setattr(boundary, "run_engine", fake_run_engine)
    monkeypatch.setattr(
        boundary,
        "hardware_identity",
        lambda: {"gpu": "GPU-test", "gpu_info": "GPU-test, B200, 1000 MiB, 1"},
    )
    with pytest.raises(InterruptedError, match="terminated"):
        boundary.run_row(
            Path("/unused/binary"),
            1,
            "ordinary",
            model="fixture-model",
            gpu_total_bytes=256,
        )
    assert cleanup == ["reaped"]
    assert signal.getsignal(termination_signal) is previous


@pytest.mark.parametrize("termination_signal", [signal.SIGINT, signal.SIGTERM])
def test_boundary_signal_during_spawn_reaps_process_group(
    monkeypatch, tmp_path, termination_signal
):
    from benchmarks import common

    real_popen = subprocess.Popen
    started = []

    def signal_during_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        os.kill(os.getpid(), termination_signal)
        return process

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(common.subprocess, "Popen", signal_during_spawn)
    with (
        pytest.raises(InterruptedError, match="cancelled"),
        boundary._reap_on_termination() as cancelled,
    ):
        common.run_engine(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            log_path=tmp_path / "spawn.log",
            cancelled=cancelled,
        )
    assert len(started) == 1
    assert started[0].poll() is not None


@pytest.mark.parametrize(
    ("source_changed", "gpu_changed"),
    [(False, False), (True, False), (False, True)],
)
def test_boundary_artifact_rejects_source_or_gpu_drift(
    monkeypatch, tmp_path, source_changed, gpu_changed
):
    binary = tmp_path / "worker"
    binary.write_bytes(b"binary")
    output = tmp_path / "summary.json"
    first = {"git_status": "", "binary_sha256": "first"}
    last = {"git_status": "", "binary_sha256": "last"} if source_changed else first
    sources = iter((first, last))
    monkeypatch.setattr(boundary, "source_identity", lambda _binary: next(sources))
    monkeypatch.setattr(
        boundary,
        "hardware_identity",
        lambda: {"gpu": "GPU-one", "gpu_info": "GPU-one, B200, 1000 MiB, 1"},
    )
    monkeypatch.setattr(
        boundary,
        "run_row",
        lambda _binary, batch_size, mode, *, model, raw_dir=None: {
            "output_tokens": batch_size * 4096,
            "max_reserved_bytes": 128,
            "gpu_uuid": "GPU-other" if gpu_changed else "GPU-one",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_boundary.py",
            "--model",
            "fixture-model",
            "--binary",
            str(binary),
            "--output",
            str(output),
            "--raw-dir",
            str(tmp_path),
        ],
    )
    if source_changed or gpu_changed:
        with pytest.raises(ValueError, match="clean source"):
            boundary.main()
    else:
        boundary.main()
    artifact = json.loads(output.read_text())
    assert artifact["passed"] == (not source_changed and not gpu_changed)
    assert artifact["source_end_matches_start"] == (not source_changed)
    assert artifact["gpu_uuid_matches_start"] == (not gpu_changed)


def test_boundary_collector_rejects_raw_log_dir_inside_repo(monkeypatch, tmp_path):
    binary = tmp_path / "worker"
    binary.write_bytes(b"binary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_boundary.py",
            "--model",
            "fixture-model",
            "--binary",
            str(binary),
            "--output",
            str(tmp_path / "summary.json"),
            "--raw-dir",
            str(boundary.ROOT / "raw-boundary-traces"),
        ],
    )
    with pytest.raises(ValueError, match="outside the repository"):
        boundary.main()


@pytest.mark.gpu
@pytest.mark.parametrize("mode", boundary.MODES)
@pytest.mark.parametrize("batch_size", boundary.BATCH_SIZES)
def test_max_context_length_262144(mode, batch_size, tmp_path):
    """Run ordinary/MTP4 258048+4096 on the outer UUID-pinned B200."""
    binary_env = os.environ.get("OH_MY_VLLM_TEST_BINARY")
    assert binary_env, "run GPU boundary cases through scripts/test.sh full"
    binary = Path(binary_env)
    assert binary.is_file(), "scripts/test.sh must build the debug bench binary"
    options = {}
    if mode == "dspark":
        draft = os.environ.get("OH_MY_VLLM_DRAFT_MODEL")
        if not draft:
            pytest.skip("DSpark boundary needs OH_MY_VLLM_DRAFT_MODEL")
        options["draft_model"] = draft
    row = boundary.run_row(
        binary,
        batch_size,
        mode,
        model=os.environ["OH_MY_VLLM_MODEL"],
        raw_dir=tmp_path,
        **options,
    )
    assert row["output_tokens"] == batch_size * 4096
    assert row["preemptions"] == 0
    assert row["max_reserved_bytes"] > 0
    if mode != "ordinary":
        assert row["proposed_draft_tokens"] > 0
    if mode == "dspark":
        assert row["verified_draft_tokens"] > 0
