"""Persistent RPC identity, context budgeting, isolation and cleanup regressions."""

import json
import os
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
from benchmarks.agentic_sessions import RpcClient, curriculum, followup
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
        'model':{'contextWindow':262144,'tokenizer':'qwen3'}}
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
