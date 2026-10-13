"""Actual acceptance parser must reject errors and response identity mixing."""

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from benchmarks import serving_identity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "budget_acceptance", ROOT / "scripts/thinking-budget-acceptance.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MODEL = "qwen3.8-27b-fp8"


class Attempt:
    def __init__(self, events):
        self.events = events

    @contextlib.contextmanager
    def stream(self, *args, **kwargs):
        lines = []
        for event in self.events:
            encoded = event if isinstance(event, str) else json.dumps(event)
            lines.extend([("data: " + encoded + "\n").encode(), b"\n"])
        yield lines


def chat(identity, content="", finish=None, usage=False):
    event = {
        "id": identity,
        "model": MODEL,
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish}
        ],
    }
    if usage:
        event["usage"] = {
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    return event


def valid_chat():
    return [
        chat("chatcmpl_a_1", "READY"),
        chat("chatcmpl_a_1", finish="stop", usage=True),
        "[DONE]",
    ]


def response(identity="resp_a_1"):
    return {
        "id": identity,
        "model": MODEL,
        "status": "completed",
        "usage": {
            "input_tokens": 8,
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "READY"}],
            }
        ],
    }


@pytest.mark.parametrize("case", ["chat_error", "chat_mixed_ids", "responses_failed"])
def test_failed_or_mixed_stream_rejected(case):
    api = "chat"
    events = valid_chat()
    if case == "chat_error":
        events.insert(1, {"error": {"message": "worker failed"}})
    elif case == "chat_mixed_ids":
        events[1]["id"] = "chatcmpl_b_2"
    else:
        api = "responses"
        events = [
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_a_1",
                    "status": "failed",
                    "error": {"message": "worker failed"},
                },
            },
            {"type": "response.completed", "response": response()},
        ]
    with pytest.raises(ValueError):
        module.stream_result(Attempt(events), "unused", {"model": MODEL}, api)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_valid_stream_still_accepted(api):
    events = (
        valid_chat()
        if api == "chat"
        else [
            {
                "type": "response.created",
                "response": dict(response(), status="in_progress", output=[]),
            },
            {"type": "response.completed", "response": response()},
        ]
    )
    result = module.stream_result(Attempt(events), "unused", {"model": MODEL}, api)
    assert result["id"] == ("chatcmpl_a_1" if api == "chat" else "resp_a_1")


@pytest.mark.parametrize(
    "mismatch", [None, "stamp", "binary", "sources", "worker", "parent"]
)
def test_acceptance_binds_live_binary_and_worker_startup(monkeypatch, mismatch):
    pid, worker = os.getpid(), 1_234_567_890
    sources = {"worker/serving.py": "reviewed"}
    source = {
        "rust_sources_sha256": "a" * 64,
        "binary_sha256": hashlib.sha256(
            Path("/proc", str(pid), "exe").read_bytes()
        ).hexdigest(),
    }
    if mismatch == "binary":
        source["binary_sha256"] = "incorrect"
    stamp = "b" * 64 if mismatch == "stamp" else source["rust_sources_sha256"]
    runtime = {
        "pid": worker,
        "message": "Independent runtime",
        "sources_sha256": {} if mismatch == "sources" else sources,
    }
    log = f'BENCH_CONFIG rust_sources_sha256="{stamp}"\n' + json.dumps(runtime)
    diagnostics = {"server_pid": pid, "worker_pid": worker + int(mismatch == "worker")}
    worker_path = Path("/proc", str(worker), "stat")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == worker_path:
            parent = pid + int(mismatch == "parent")
            return f"{worker} (python) S {parent} " + "0 " * 17 + "17"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        serving_identity, "runtime_contract", lambda: {"sources": sources}
    )
    if mismatch:
        with pytest.raises(ValueError):
            module.server_identity(diagnostics, log, source)
    else:
        identity = module.server_identity(diagnostics, log, source)
        assert identity["pid"] == pid
        assert identity["worker_pid"] == worker
        assert identity["worker_start_ticks"] == 17
