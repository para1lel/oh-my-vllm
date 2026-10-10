"""CPU regressions for separate bounded warmup and normal agent budgets."""

import threading

import pytest

from benchmarks.agentic_soak import Evidence, build_parser, run
from benchmarks.agentic_soak_proxy import ModelSlots


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_warmup_budget_must_be_positive_and_finite(tmp_path, budget):
    args = build_parser().parse_args(
        [
            "--mode",
            "mtp",
            "--base-url",
            "unused",
            "--server-log",
            "unused",
            "--output-dir",
            str(tmp_path / "raw"),
            "--diagnostic",
            "--warmup-task-timeout",
            budget,
        ]
    )
    with pytest.raises(ValueError, match="durations"):
        run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("phase,expected", [("warmup", 900), ("normal", 210)])
def test_actual_agent_and_proxy_receive_phase_budget(
    tmp_path, monkeypatch, phase, expected
):
    from contextlib import contextmanager

    import benchmarks.agentic_soak as soak

    args = build_parser().parse_args(
        [
            "--mode",
            "mtp",
            "--base-url",
            "unused",
            "--server-log",
            "unused",
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = {}

    @contextmanager
    def proxy(_url, _directory, _slots, *, request_timeout):
        captured["http_timeout"] = request_timeout
        yield "http://local/v1", []

    def omp(command, _state, _directory, _stop, timeout):
        captured["omp_duration"] = command[command.index("--max-time") + 1]
        captured["task_timeout"] = timeout
        return 1  # Failed fixture prevents unearned semantic-evidence success.

    monkeypatch.setattr(soak, "disk_proxy", proxy)
    monkeypatch.setattr(soak, "run_omp", omp)
    monkeypatch.setattr(soak, "load_requests", lambda _path: [])
    result = soak.run_task(
        args, Evidence(tmp_path), ModelSlots(), threading.Event(), 0, 0, 0, phase=phase
    )
    assert not result["passed"]
    assert result["task_budget_seconds"] == expected
    assert captured == {
        "http_timeout": expected,
        "task_timeout": expected,
        "omp_duration": f"{expected}s",
    }
