"""Persistent RPC identity, context budgeting, isolation and cleanup regressions."""

import io
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from benchmarks.agentic_context import (
    COMPACTION_THRESHOLD,
    CONTEXT_WINDOW,
    OUTPUT_BUDGET,
    RESERVE_TOKENS,
    isolated_environment,
    session_configuration,
)
from benchmarks.agentic_sessions import (
    PROCESS_REAP_SECONDS,
    RpcClient,
    curriculum,
    followup,
)
from benchmarks.agentic_soak import build_parser, run


def test_window_headroom_and_maximum_effort_apply_to_task_and_summary_model():
    for api in ("chat", "responses"):
        models, settings = session_configuration("http://127.0.0.1:1/v1", api)
        assert list(models["providers"]) == [f"oh-my-vllm-{api}"]
        model = next(iter(models["providers"].values()))["models"][0]
        assert model["contextWindow"] == CONTEXT_WINDOW == 262144
        assert model["tokenizer"] == "qwen3"
        assert COMPACTION_THRESHOLD + RESERVE_TOKENS == CONTEXT_WINDOW
        assert OUTPUT_BUDGET <= RESERVE_TOKENS
        assert model["thinking"]["defaultLevel"] == "max"
        assert model["compat"]["reasoningEffortMap"]["max"] == "xhigh"
        assert len(settings["modelRoles"]) == 10
        assert set(settings["modelRoles"].values()) == {model["compactionModel"]}
        assert settings["compaction"]["methodOrder"] == ["soft"]
        assert settings["compaction"]["midTurnEnabled"] is True
        assert settings["compaction"]["asyncEnabled"] is False


def test_provider_credentials_and_global_config_do_not_enter_client_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("PI_CONFIG_FILES", "user-settings.yml")
    environment = isolated_environment(tmp_path)
    assert "OPENAI_API_KEY" not in environment
    assert "ANTHROPIC_BASE_URL" not in environment
    assert environment["HOME"] == os.environ["HOME"]
    assert environment["PI_CODING_AGENT_DIR"] == str(tmp_path)
    assert environment["PI_CONFIG_FILES"] == str(tmp_path / "settings.yml")


def test_four_audit_lanes_have_real_different_sources_without_prompt_padding():
    for lane in range(4):
        paths = curriculum(lane)
        assert paths
        assert all(Path(path).exists() for path in paths)
        first, second = followup(lane, 0, paths), followup(lane, 1, paths)
        assert first != second
        assert paths[0] in first
        assert "notes/progress.md" in first
        assert "compaction" in first


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_persistent_turn_deadline_must_be_finite_and_positive(tmp_path, value):
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
            "--turn-timeout",
            value,
        ]
    )
    with pytest.raises(ValueError, match="durations"):
        run(args)
    assert not args.output_dir.exists()


class Owner:
    """One lane owner that records its close and can fail like a real cleanup."""

    def __init__(self, name, calls, error=None):
        self.name, self.calls, self.error = name, calls, error

    def close(self, **kwargs):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error


def test_constructor_rollback_attempts_every_owner_and_keeps_the_original_error(
    tmp_path, monkeypatch
):
    """A cleanup fault must not mask the initialization failure or skip owners."""
    calls = []
    owners = {
        "sandbox": Owner(
            "sandbox", calls, error=RuntimeError("sandbox cleanup failed")
        ),
        "transport": Owner("transport", calls),
        "sdk_trace": Owner("sdk-trace", calls),
        "stderr": Owner("stderr", calls),
    }

    def initialize(self, *args, **kwargs):
        for name, owner in owners.items():
            setattr(self, name, owner)
        raise ValueError("original initialization failure")

    monkeypatch.setattr(RpcClient, "_initialize", initialize)
    with pytest.raises(ValueError, match="original initialization failure") as caught:
        RpcClient(
            "omp",
            "chat",
            0,
            tmp_path,
            "http://127.0.0.1:1",
            threading.Event(),
            5,
        )
    assert calls == ["sandbox", "transport", "sdk-trace", "stderr"]
    notes = list(getattr(caught.value, "__notes__", []))
    assert any("lane sandbox cleanup failed" in note for note in notes), notes
    assert caught.value.args == ("original initialization failure",)


def test_inner_launch_cleanup_keeps_the_original_error(tmp_path, monkeypatch):
    """Both inner failure points reap the guest, close streams and try every owner.

    The launch and the post-launch reader start are separate failure points
    that used to close owners one by one, so a failing SDK release replaced the
    original error and skipped the remaining owners.
    """
    trace_closes = []
    opened = []
    real_open = Path.open

    class FailingTrace:
        def __init__(self, directory):
            self.directory = directory

        def close(self, *, strict=True):
            trace_closes.append(strict)
            raise RuntimeError("sdk cleanup failed")

        def environment(self):
            return {}

    def recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self.name == "stderr.log":
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr("benchmarks.agentic_sessions.SdkTrace", FailingTrace)
    monkeypatch.setattr(
        "benchmarks.agentic_sessions.prepare_workspace",
        lambda directory: {"files": [], "source_root": str(directory)},
    )

    def launch(command, **kwargs):
        raise FileNotFoundError("guest executable disappeared")

    monkeypatch.setattr("benchmarks.agentic_sessions.subprocess.Popen", launch)
    with pytest.raises(
        FileNotFoundError, match="guest executable disappeared"
    ) as caught:
        RpcClient(
            "omp", "chat", 0, tmp_path, "http://127.0.0.1:1", threading.Event(), 5
        )
    notes = list(getattr(caught.value, "__notes__", []))
    assert any("lane sdk-trace cleanup failed" in note for note in notes), notes
    assert set(trace_closes) == {False}, trace_closes
    assert opened and opened[0].closed
    assert caught.value.args == ("guest executable disappeared",)


def test_post_launch_reader_failure_reaps_the_guest(tmp_path, monkeypatch):
    """A reader start failure must not leave a running guest or open pipes."""
    launched = []
    trace_closes = []
    real_popen = subprocess.Popen

    class FailingTrace:
        def __init__(self, directory):
            self.directory = directory

        def close(self, *, strict=True):
            trace_closes.append(strict)
            raise RuntimeError("sdk cleanup failed")

        def environment(self):
            return {}

    def launch(command, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        launched.append(process)
        return process

    def refuse_start(self):
        raise RuntimeError("cannot start a new thread")

    monkeypatch.setattr("benchmarks.agentic_sessions.SdkTrace", FailingTrace)
    monkeypatch.setattr(
        "benchmarks.agentic_sessions.prepare_workspace",
        lambda directory: {"files": [], "source_root": str(directory)},
    )
    monkeypatch.setattr("benchmarks.agentic_sessions.subprocess.Popen", launch)
    monkeypatch.setattr(threading.Thread, "start", refuse_start)
    try:
        with pytest.raises(RuntimeError, match="cannot start a new thread") as caught:
            RpcClient(
                "omp", "chat", 0, tmp_path, "http://127.0.0.1:1", threading.Event(), 5
            )
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
            process.wait()
    notes = list(getattr(caught.value, "__notes__", []))
    assert any("lane sdk-trace cleanup failed" in note for note in notes), notes
    assert set(trace_closes) == {False}, trace_closes
    assert launched and launched[0].poll() is not None
    assert launched[0].stdin.closed and launched[0].stdout.closed
    assert caught.value.args == ("cannot start a new thread",)


class StuckProcess:
    """A launched guest that ignores termination and never exits."""

    pid = 4321
    returncode = None

    def __init__(self):
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        self.timeouts = []

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        raise subprocess.TimeoutExpired("omp", timeout)


def test_bounded_reap_reports_survival_and_still_releases_pipes(monkeypatch):
    """A guest that survives its bounded reap must be reported, not hidden."""
    signals = []
    monkeypatch.setattr(
        "benchmarks.agentic_sessions.os.killpg",
        lambda pgid, number: signals.append((pgid, number)),
    )
    process = StuckProcess()
    client = RpcClient.__new__(RpcClient)
    error = ValueError("original startup failure")
    client._release_owner("process", process, error, lambda: client._reap(process))
    # The wait stayed bounded and asked the launched group to stop.
    assert process.timeouts == [PROCESS_REAP_SECONDS]
    assert signals == [(StuckProcess.pid, signal.SIGKILL)]
    # Survival travels as a cleanup note on the original failure.
    notes = list(getattr(error, "__notes__", []))
    assert any(
        "lane process cleanup failed" in note and "survived its" in note
        for note in notes
    ), notes
    assert error.args == ("original startup failure",)
    assert process.stdin.closed and process.stdout.closed


def test_rollback_reports_a_stuck_guest_and_releases_every_owner(monkeypatch):
    """The whole rollback keeps the original error and its cleanup notes."""
    signals, closed = [], []
    monkeypatch.setattr(
        "benchmarks.agentic_sessions.os.killpg",
        lambda pgid, number: signals.append((pgid, number)),
    )

    class Owner:
        def __init__(self, name):
            self.name = name

        def close(self, **kwargs):
            closed.append(self.name)

    client = RpcClient.__new__(RpcClient)
    client.process = StuckProcess()
    client.sandbox, client.transport = Owner("sandbox"), Owner("transport")
    client.sdk_trace, client.stderr = Owner("sdk-trace"), Owner("stderr")
    error = FileNotFoundError("original launch failure")
    client._rollback(error, process=True)
    notes = list(getattr(error, "__notes__", []))
    assert any("survived its" in note for note in notes), notes
    assert closed == ["sandbox", "transport", "sdk-trace", "stderr"]
    assert client.process.stdin.closed and client.process.stdout.closed
    assert error.args == ("original launch failure",)


def test_hour_has_no_request_floor_or_separate_warmup():
    args = build_parser().parse_args(
        [
            "--mode",
            "mtp",
            "--base-url",
            "unused",
            "--server-log",
            "unused",
            "--output-dir",
            "unused",
        ]
    )
    assert args.duration == 3600
    assert args.agents == 4
    assert not hasattr(args, "min_requests")
    assert not hasattr(args, "warmup_epoch_seconds")


def test_default_workload_is_the_corpus_translation_without_a_gateway():
    args = build_parser().parse_args(
        [
            "--mode",
            "mtp",
            "--base-url",
            "unused",
            "--server-log",
            "unused",
            "--output-dir",
            "unused",
        ]
    )
    assert args.workload == "translation"
    assert not hasattr(args, "sandbox_host")
    assert args.corpus is None


def soak_arguments(tmp_path, **overrides):
    values = {
        "--mode": "mtp",
        "--base-url": "unused",
        "--server-log": "unused",
        "--output-dir": str(tmp_path / "raw"),
        "--model": str(tmp_path / "model"),
    }
    arguments = []
    for key, value in values.items():
        arguments.extend((key, value))
    for key, value in overrides.items():
        arguments.extend((f"--{key.replace('_', '-')}", value))
    return build_parser().parse_args(arguments)


def test_formal_soak_requires_an_external_translation_corpus(tmp_path):
    from benchmarks.agentic_corpus import ROOT

    args = soak_arguments(tmp_path)
    with pytest.raises(ValueError, match="supply --corpus"):
        run(args)
    assert not args.output_dir.exists()
    inside = soak_arguments(tmp_path, corpus=str(ROOT / "benchmarks"))
    with pytest.raises(ValueError, match="outside the source repository"):
        run(inside)
    assert not inside.output_dir.exists()


def test_formal_soak_rejects_the_diagnostic_workloads(tmp_path):
    for workload in ("goal", "audit"):
        args = soak_arguments(tmp_path, workload=workload)
        with pytest.raises(ValueError, match="translation workload"):
            run(args)
        assert not args.output_dir.exists()


@pytest.mark.parametrize("change_identity", [False, True])
def test_two_prompts_use_one_process_and_reject_session_replacement(
    tmp_path, monkeypatch, change_identity
):
    executable = tmp_path / "fake-omp"
    executable.write_text(
        """#!/usr/bin/env python3
import json,sys
turn=0
for line in sys.stdin:
 command=json.loads(line); kind=command['type']; identity=command['id']
 data={}
 if kind=='get_available_models':
  data={'models':[{'provider':'oh-my-vllm-chat'}]}
 if kind=='get_entries':
  entry={'id':f'entry-{turn}','type':'message'}
  data={'entries':[] if command.get('since')==entry['id'] else [entry],
        'leafId':entry['id']}
 if kind=='get_state':
  data={'sessionId':'session-2' if CHANGE and turn>1 else 'session-1',
        'thinkingLevel':'max','autoCompactionEnabled':True,'isSettled':True,
        'model':{'contextWindow':262144,'tokenizer':'qwen3',
                 'api':'openai-completions','provider':'oh-my-vllm-chat',
                 'id':'qwen3.8-27b-fp8'}}
 print(json.dumps({'type':'response','id':identity,'success':True,'data':data}),flush=True)
 if kind=='prompt':
  turn+=1
  print(json.dumps({'type':'prompt_result','id':identity,'status':'completed',
                    'agentInvoked':True,'sessionSettled':True}),flush=True)
""".replace("CHANGE", repr(change_identity))
    )
    executable.chmod(0o755)
    directory = tmp_path / "session"
    directory.mkdir()

    def workspace(path):
        (path / "notes").mkdir()
        return {"files": []}

    monkeypatch.setattr("benchmarks.agentic_sessions.prepare_workspace", workspace)
    client = RpcClient(
        str(executable),
        "chat",
        0,
        directory,
        "http://127.0.0.1:1/v1",
        threading.Event(),
        2,
    )
    pid = client.process.pid
    try:
        client.initialize()
        client.prompt("first source audit", 0)
        assert client.session_id == "session-1"
        if change_identity:
            with pytest.raises(ValueError, match="identity changed"):
                client.prompt("continue the same audit", 1)
        else:
            client.prompt("continue the same audit", 1)
            assert len(client.turns) == 2
            assert client.collect()["entry_count"] == 2
        assert client.process.pid == pid
    finally:
        client.close()
    assert not Path("/proc", str(pid)).exists()
    commands = [
        json.loads(line)
        for line in (directory / "commands.jsonl").read_text().splitlines()
    ]
    assert [item["message"] for item in commands if item["type"] == "prompt"] == [
        "first source audit",
        "continue the same audit",
    ]
    assert json.loads((directory / "process.json").read_text())["reaped"] is True
    if not change_identity:
        checkpoints = [item for item in commands if item["type"] == "get_entries"]
        assert "since" not in checkpoints[0]
        assert checkpoints[1]["since"] == "entry-1"
        assert checkpoints[2]["since"] == "entry-2"
