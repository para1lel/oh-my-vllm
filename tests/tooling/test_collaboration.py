"""Pricing boundaries and Issue scope cannot silently change at dispatch."""

import datetime as dt
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from development import collaboration
from development.collaboration import dispatch_allowed, issue_branch, peak_pricing


@pytest.mark.parametrize(
    "clock,expected",
    [
        ("08:59:59", False),
        ("09:00:00", True),
        ("11:59:59", True),
        ("12:00:00", False),
        ("13:59:59", False),
        ("14:00:00", True),
        ("17:59:59", True),
        ("18:00:00", False),
    ],
)
def test_weekday_pricing_boundaries(clock, expected):
    moment = dt.datetime.fromisoformat(f"2026-10-14T{clock}+08:00")
    assert peak_pricing(moment) is expected
    assert peak_pricing(moment.astimezone(dt.UTC)) is expected
    if expected:
        with pytest.raises(ValueError, match="peak pricing"):
            dispatch_allowed(moment)
    else:
        dispatch_allowed(moment)


def test_weekends_and_verified_holidays_are_off_peak():
    weekend = dt.datetime.fromisoformat("2026-10-17T10:00:00+08:00")
    assert not peak_pricing(weekend)
    weekday = dt.datetime.fromisoformat("2026-10-14T10:00:00+08:00")
    assert peak_pricing(weekday)
    assert not peak_pricing(weekday, frozenset({weekday.date()}))


def test_host_timezone_cannot_determine_pricing():
    with pytest.raises(ValueError, match="explicit timezone"):
        peak_pricing(dt.datetime(2026, 10, 14, 10))


@pytest.mark.parametrize("branch", ["main", "omp/issue-40-other", "omp/issue-3-wrong"])
def test_other_branches_cannot_receive_an_issue_task(branch):
    with pytest.raises(ValueError, match="assigned Issue"):
        issue_branch(branch, 4)


def test_assigned_branch_is_accepted():
    issue_branch("omp/issue-4-translation-soak", 4)


def dispatch_fixture(monkeypatch, checkout, times, exit_code=0):
    moments = iter(dt.datetime.fromisoformat(value) for value in times)

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments)

    monkeypatch.setattr(collaboration.dt, "datetime", Clock)
    monkeypatch.setattr(
        collaboration, "validate_checkout", lambda *_: "omp/issue-4-test"
    )
    calls, spawned, processes = [], [], []

    def command(arguments, *_):
        calls.append(arguments)
        if arguments[:2] == ["git", "status"]:
            return ""
        if arguments[:3] == ["gh", "issue", "view"]:
            return json.dumps({"state": "OPEN", "title": "Assigned task"})
        raise AssertionError(arguments)

    class Process:
        pid, returncode = 12345, None
        stdin = io.BytesIO()

        def wait(self, timeout=None):
            self.returncode = exit_code
            arguments = spawned[-1]
            audit = arguments[arguments.index("--audit") + 1]
            from pathlib import Path

            Path(audit).write_text(json.dumps({"passed": exit_code == 0}))
            return exit_code

        def poll(self):
            return self.returncode

    def popen(arguments, **_):
        spawned.append(arguments)
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr(collaboration, "command", command)
    monkeypatch.setattr(collaboration.subprocess, "Popen", popen)
    monkeypatch.setattr(
        collaboration,
        "process_birth",
        lambda _: 50 if not processes or processes[-1].returncode is None else None,
    )
    return calls, spawned


@pytest.mark.parametrize(
    "times,read_issue",
    [
        (["2026-10-14T09:00:00+08:00"], False),
        (["2026-10-14T08:59:59+08:00", "2026-10-14T09:00:00+08:00"], True),
    ],
)
def test_peak_blocks_dispatch_before_and_after_github_read(
    monkeypatch, tmp_path, times, read_issue
):
    calls, spawned = dispatch_fixture(monkeypatch, tmp_path, times)
    with pytest.raises(ValueError, match="peak pricing"):
        collaboration.dispatch(tmp_path, 4, "high", None)
    assert bool(calls) is read_issue
    assert not spawned


def test_failed_omp_turn_keeps_failure_and_returns_an_error(monkeypatch, tmp_path):
    _, spawned = dispatch_fixture(
        monkeypatch,
        tmp_path,
        ["2026-10-14T18:00:00+08:00", "2026-10-14T18:00:01+08:00"],
        exit_code=9,
    )
    with pytest.raises(ValueError, match="OMP task failed"):
        collaboration.dispatch(tmp_path, 4, "high", None)
    assert spawned
    record = json.loads((tmp_path / ".local/omp/issue-4.json").read_text())
    assert record["status"] == "failed"
    assert record["exit_code"] == 9


def test_existing_issue_can_continue_its_native_session_during_peak(
    monkeypatch, tmp_path
):
    _, spawned = dispatch_fixture(monkeypatch, tmp_path, ["2026-10-14T09:00:00+08:00"])
    state = tmp_path / ".local/omp"
    sessions = state / "issue-4-sessions"
    sessions.mkdir(parents=True)
    saved = sessions / "native.jsonl"
    saved.write_text(
        json.dumps({"type": "title", "title": "Native session"})
        + "\n"
        + json.dumps({"type": "session", "cwd": str(tmp_path)})
        + "\n"
    )
    (state / "issue-4.json").write_text(
        json.dumps(
            {
                "issue": 4,
                "branch": "omp/issue-4-test",
                "pid": 99,
                "start_ticks": 1,
                "session_directory": str(sessions),
                "status": "awaiting_review",
                "started_at": "2026-10-13T18:00:00+08:00",
            }
        )
    )
    collaboration.dispatch(tmp_path, 4, "high", "Apply review corrections")
    [arguments] = spawned
    assert arguments[arguments.index("--resume") + 1] == str(saved)


def test_branch_change_during_github_read_cannot_start_omp(monkeypatch, tmp_path):
    _, spawned = dispatch_fixture(monkeypatch, tmp_path, ["2026-10-14T18:00:00+08:00"])
    branches = iter(("omp/issue-4-test", "main"))
    monkeypatch.setattr(collaboration, "validate_checkout", lambda *_: next(branches))
    with pytest.raises(ValueError, match="branch changed"):
        collaboration.dispatch(tmp_path, 4, "high", None)
    assert not spawned


def test_record_write_failure_closes_supervisor_control(monkeypatch, tmp_path):
    from pathlib import Path

    _, spawned = dispatch_fixture(
        monkeypatch,
        tmp_path,
        ["2026-10-14T18:00:00+08:00", "2026-10-14T18:00:01+08:00"],
    )
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "issue-4.json":
            raise OSError("record volume full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="record volume full"):
        collaboration.dispatch(tmp_path, 4, "high", None)
    assert spawned
    audit = tmp_path / ".local/omp/issue-4-1.processes.json"
    assert audit.exists()  # Cleanup waited on the supervisor even without a record.


def test_cleanup_failure_preserves_the_primary_record_error(monkeypatch, tmp_path):
    dispatch_fixture(
        monkeypatch,
        tmp_path,
        ["2026-10-14T18:00:00+08:00", "2026-10-14T18:00:01+08:00"],
    )

    class Process:
        pid, returncode = 12345, None
        stdin = io.BytesIO()

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("supervisor", timeout)

    monkeypatch.setattr(collaboration.subprocess, "Popen", lambda *_, **__: Process())
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "issue-4.json":
            raise OSError("record volume full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="record volume full") as failure:
        collaboration.dispatch(tmp_path, 4, "high", None)
    assert any("TimeoutExpired" in note for note in failure.value.__notes__)


def test_old_checkout_uses_the_launchers_reviewed_supervisor(monkeypatch, tmp_path):
    checkout = tmp_path / "old-checkout"
    checkout.mkdir()
    old_module = checkout / "benchmarks"
    old_module.mkdir()
    (old_module / "__init__.py").touch()
    (old_module / "engine_supervisor.py").write_text(
        "raise RuntimeError('old checkout supervisor must not execute')\n"
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "omp"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys,time\n"
        "from pathlib import Path\n"
        "os.chdir(sys.argv[sys.argv.index('--cwd')+1])\n"
        "Path('task-started').write_text('cpu fixture')\n"
        "time.sleep(.1)\n"
    )
    fake.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(collaboration, "github_environment", lambda: environment)
    monkeypatch.setattr(
        collaboration, "validate_checkout", lambda *_: "omp/issue-4-test"
    )
    monkeypatch.setattr(collaboration, "dispatch_allowed", lambda *_: None)
    monkeypatch.setattr(
        collaboration,
        "command",
        lambda args, *_: json.dumps({"state": "OPEN"}) if args[0] == "gh" else "",
    )
    collaboration.dispatch(checkout, 4, "high", None)
    assert (checkout / "task-started").read_text() == "cpu fixture"
    record = json.loads((checkout / ".local/omp/issue-4.json").read_text())
    assert record["status"] == "awaiting_review"
    assert record["cleanup_passed"] and record["process_exited"]
