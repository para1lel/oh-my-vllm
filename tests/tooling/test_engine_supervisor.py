"""Verify isolated ownership and cleanup through real Linux process faults."""

import json
import select
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmarks import engine_supervisor


@contextmanager
def supervisor(tmp_path, engine, *, natural_completion=False):
    log, audit = tmp_path / "engine.log", tmp_path / "processes.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.engine_supervisor",
            "--log",
            str(log),
            "--audit",
            str(audit),
            *(["--natural-completion"] if natural_completion else []),
            "--",
            sys.executable,
            "-c",
            engine,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 5)[0], "supervisor never started"
        event = json.loads(process.stdout.readline())
        assert event["event"] == "started"
        yield process, log, audit, event
    finally:
        if process.poll() is None:
            try:
                process.stdin.write("stop\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
        process.wait(timeout=20)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def wait_log(log):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log.exists() and log.read_text().strip():
            return int(log.read_text().strip())
        time.sleep(0.01)
    pytest.fail("engine child identity was not recorded")


def released(audit, child):
    result = json.loads(audit.read_text())
    assert result["cleanup"]["live"] == []
    assert result["cleanup"]["unverified_adopted"] == []
    assert result["cleanup"]["invalid_sessions"] == []
    assert not Path(f"/proc/{child}").exists()
    assert child in {record["pid"] for record in result["cleanup"]["processes"]}
    observations = [
        json.loads(line)
        for line in audit.with_suffix(".observations.jsonl").read_text().splitlines()
    ]
    if "monitor" in result:
        assert result["monitor"]["observations"] == len(observations)
        assert result["monitor"]["maximum_interval_s"] == max(
            record["interval_s"] or 0 for record in observations
        )
    return result


ORPHAN = """import os,time
intermediate = os.fork()
if intermediate == 0:
    os.setsid()
    if os.fork() == 0:
        print(os.getpid(),flush=True)
        time.sleep(60)
        os._exit(0)
    os._exit(0)
os.waitpid(intermediate,0)
time.sleep(60)
"""


def test_new_session_descendant_is_released_by_isolated_scope(tmp_path):
    with supervisor(tmp_path, ORPHAN) as (process, log, audit, _event):
        child = wait_log(log)
        process.stdin.write("stop\n")
        process.stdin.flush()
        assert process.wait(timeout=20) == 0
        result = released(audit, child)
    assert result["passed"] is True


@pytest.mark.parametrize("exit_code", [0, 2])
def test_natural_completion_releases_a_stubborn_orphan(tmp_path, exit_code):
    engine = """import os,signal,time
child = os.fork()
if child == 0:
    os.setsid()
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    print(os.getpid(),flush=True)
    time.sleep(60)
    os._exit(0)
time.sleep(.2)
os._exit(EXIT_CODE)
""".replace("EXIT_CODE", str(exit_code))
    with supervisor(tmp_path, engine, natural_completion=True) as (
        process,
        log,
        audit,
        _event,
    ):
        child = wait_log(log)
        assert process.wait(timeout=20) == (0 if exit_code == 0 else 1)
        result = released(audit, child)
    assert result["completion_mode"] == "natural_exit"
    assert result["engine_returncode"] == exit_code
    assert result["passed"] is (exit_code == 0)


def test_delayed_first_observation_owns_orphan_without_session_leader(tmp_path):
    log = tmp_path / "orphan.log"
    code = """import json,os,subprocess,sys,time
from pathlib import Path
from benchmarks.common import (
    _child_identities,_child_subreaper,_process_record,_stop_engine
)
from benchmarks.engine_supervisor import IsolatedEngineOwnership
with _child_subreaper():
    assert not _child_identities()
    owner = IsolatedEngineOwnership(_process_record(os.getpid()))
    with Path(sys.argv[1]).open('x') as output:
        root = subprocess.Popen(
            [sys.executable,'-c',sys.argv[2]],stdout=output,start_new_session=True
        )
        owner.bind_engine(root)
        try:
            deadline = time.monotonic()+5
            while not Path(sys.argv[1]).read_text().strip():
                assert time.monotonic()<deadline
                time.sleep(.01)
            child = int(Path(sys.argv[1]).read_text())
            while (
                _process_record(child)['ppid'] != os.getpid()
                or _process_record(_process_record(child)['session']) is not None
            ):
                assert time.monotonic()<deadline
                time.sleep(.01)
            assert not owner.owns(child)
            owner.refresh()
            assert owner.owns(child)
            _stop_engine(root,owner)
            print(json.dumps(owner.snapshot()),flush=True)
        finally:
            _stop_engine(root,owner)
            owner.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(log), ORPHAN],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit["live"] == []
    assert audit["unverified_adopted"] == []
    assert any(action.get("adopted_supervisor") for action in audit["signals"])
    assert not Path(f"/proc/{int(log.read_text())}").exists()


@pytest.mark.parametrize("control", ["eof", "invalid", "root_exit"])
def test_controller_or_engine_failure_releases_descendants(tmp_path, control):
    engine = ORPHAN
    if control == "root_exit":
        engine = ORPHAN.rsplit("time.sleep(60)", 1)[0] + "time.sleep(.1)\nos._exit(2)\n"
    with supervisor(tmp_path, engine) as (process, log, audit, _event):
        child = wait_log(log)
        if control == "eof":
            process.stdin.close()
        elif control == "invalid":
            process.stdin.write("unknown\n")
            process.stdin.flush()
        assert process.wait(timeout=20) == 1
        result = released(audit, child)
    assert result["passed"] is False
    if control == "invalid":
        assert result["errors"][0]["operation"] == "monitor"
    else:
        assert (
            result["monitor"]["stop_cause"]
            == {"eof": "controller_eof", "root_exit": "engine_exited"}[control]
        )


def test_unrelated_collector_child_is_never_part_of_engine_scope(tmp_path):
    foreign = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        with supervisor(tmp_path, ORPHAN) as (process, log, audit, _event):
            child = wait_log(log)
            process.stdin.write("stop\n")
            process.stdin.flush()
            assert process.wait(timeout=20) == 0
            result = released(audit, child)
        assert foreign.poll() is None
        assert foreign.pid not in {
            record["pid"] for record in result["cleanup"]["processes"]
        }
        assert foreign.pid not in {
            record["pid"] for record in result["cleanup"]["signals"]
        }
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_partial_control_command_does_not_block_signal_cleanup(tmp_path):
    with supervisor(tmp_path, ORPHAN) as (process, log, audit, _event):
        child = wait_log(log)
        process.stdin.write("st")
        process.stdin.flush()
        time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 1
        result = released(audit, child)
    assert result["monitor"]["stop_cause"] == "controller_signal"
    assert result["monitor"]["observations"] > 1


def test_engine_cannot_read_controller_pipe(tmp_path):
    engine = "import sys,time;print(len(sys.stdin.read()),flush=True);time.sleep(60)"
    with supervisor(tmp_path, engine) as (process, log, audit, event):
        assert wait_log(log) == 0
        process.stdin.write("stop\n")
        process.stdin.flush()
        assert process.wait(timeout=20) == 0
        result = released(audit, event["engine"]["pid"])
    assert result["passed"] is True


def test_root_pidfd_failure_still_releases_already_adopted_orphan(tmp_path):
    log, audit = tmp_path / "bind.log", tmp_path / "bind-audit.json"
    code = """import os,sys,time
from pathlib import Path
from benchmarks import common,engine_supervisor
original = common._open_pidfd
failed = False
def open_pidfd(pid):
    global failed
    if pid != os.getpid() and not failed:
        failed = True
        deadline = time.monotonic()+5
        path = Path(sys.argv[1])
        while not path.read_text().strip():
            assert time.monotonic()<deadline
            time.sleep(.01)
        child = int(path.read_text())
        while (
            common._process_record(child)['ppid'] != os.getpid()
            or common._process_record(common._process_record(child)['session'])
        ):
            assert time.monotonic()<deadline
            time.sleep(.01)
        raise OSError('injected root pidfd failure')
    return original(pid)
common._open_pidfd = open_pidfd
result = engine_supervisor.supervise(
    [sys.executable,'-c',sys.argv[3]],Path(sys.argv[1]),Path(sys.argv[2])
)
raise SystemExit(0 if result['passed'] else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(log), str(audit), ORPHAN],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stderr
    summary = released(audit, int(log.read_text()))
    assert summary["passed"] is False
    assert "injected root pidfd failure" in summary["errors"][0]["error"]
    assert any(
        action.get("adopted_supervisor") for action in summary["cleanup"]["signals"]
    )


def test_cleanup_keeps_primary_snapshot_and_close_errors():
    process, ownership = Mock(), Mock()
    process.returncode = -15
    ownership.snapshot.side_effect = RuntimeError("snapshot failed")
    ownership.close.side_effect = RuntimeError("close failed")
    summary = {"errors": [{"operation": "monitor", "error": "primary failure"}]}
    with (
        patch.object(
            engine_supervisor, "_stop_engine", side_effect=RuntimeError("stop failed")
        ),
        patch.object(engine_supervisor, "_stop_bound_processes") as fallback,
    ):
        engine_supervisor._cleanup_engine(process, ownership, summary)
    fallback.assert_called_once()
    assert [record["operation"] for record in summary["errors"]] == [
        "monitor",
        "stop_engine",
        "snapshot",
        "close_pidfds",
    ]
    assert summary["engine_returncode"] == -15


def test_existing_audit_is_preserved_without_starting_engine(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text("original\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.engine_supervisor",
            "--log",
            str(tmp_path / "unused.log"),
            "--audit",
            str(audit),
            "--",
            "missing-executable",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert audit.read_text() == "original\n"
    assert not (tmp_path / "unused.log").exists()
