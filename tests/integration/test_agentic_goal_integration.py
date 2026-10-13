"""Real sandboxed OMP native goal continuations through deterministic CPU APIs."""

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks import agentic_sandbox
from benchmarks.agentic_context import ContextGuard
from benchmarks.agentic_goals import run_goal
from benchmarks.agentic_sandbox import AgentSandbox, docker
from benchmarks.agentic_sessions import RpcClient

pytestmark = pytest.mark.omp


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_native_goal_completion_advances_queue_without_replacing_session(
    tmp_path, agentic_sandbox_image, api
):
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            index = len(bodies)
            identity = f"resp_native_goal_{index}"
            tool = (index - 1) % 5
            calls = [
                ("read", {"path": "TASK.md"}),
                (
                    "bash",
                    {
                        "command": "python -m unittest discover -s tests -q",
                        "i": "Verify the fixture",
                        "name": "x" * 49,
                    },
                ),
                (
                    "bash",
                    {
                        "command": "python -m unittest discover -s tests -q",
                        "i": "Verify the fixture",
                        "async": False,
                    },
                ),
                ("goal", {"op": "complete"}),
            ]
            name, arguments = calls[tool] if tool < 4 else ("", {})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if api == "chat":
                delta = {"role": "assistant"}
                if tool < 4:
                    delta["tool_calls"] = [
                        {
                            "index": 0,
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                else:
                    delta["content"] = "Fixture goal completed."
                events = [
                    {
                        "id": identity,
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    },
                    {
                        "id": identity,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if tool < 4 else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 40,
                            "completion_tokens": 10,
                            "total_tokens": 50,
                        },
                    },
                ]
            else:
                item = (
                    {
                        "type": "function_call",
                        "id": f"fc_{index}",
                        "call_id": f"call_{index}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                    if tool < 4
                    else {
                        "type": "message",
                        "id": f"msg_{index}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Fixture goal completed.",
                                "annotations": [],
                            }
                        ],
                    }
                )
                response = {
                    "id": identity,
                    "object": "response",
                    "status": "completed",
                    "model": body["model"],
                    "output": [item],
                    "usage": {
                        "input_tokens": 40,
                        "output_tokens": 10,
                        "total_tokens": 50,
                    },
                }
                events = [
                    {
                        "type": "response.created",
                        "response": dict(response, status="in_progress", output=[]),
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": dict(item, status="in_progress", arguments="")
                        if tool < 4
                        else item,
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "item_id": item["id"],
                        "delta": item["arguments"],
                    }
                    if tool < 4
                    else {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": item["id"],
                        "delta": "Fixture goal completed.",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            if api == "chat":
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    directory = tmp_path / "client"
    directory.mkdir()
    client = RpcClient(
        "omp",
        api,
        0,
        directory,
        f"http://127.0.0.1:{server.server_port}/v1",
        threading.Event(),
        30,
        goal_mode=True,
        sandbox_image=agentic_sandbox_image,
    )
    (client.workspace / "tests").mkdir()
    (client.workspace / "tests/test_fixture.py").write_text(
        "import unittest\nclass Fixture(unittest.TestCase):\n"
        "    def test_cpu_check(self):\n        self.assertEqual(sum(range(10)), 45)\n"
    )
    try:
        client.initialize()
        client.command(
            "set_todos",
            phases=[
                {
                    "name": "Previous blocked goal",
                    "tasks": [
                        {
                            "content": "unavailable previous dependency",
                            "status": "blocked",
                        },
                    ],
                }
            ],
        )
        session, process = client.session_id, client.process.pid
        run_goal(client, {"monotonic": time.monotonic()}, 6)
        result = client.collect()
        assert client.session_id == session and client.process.pid == process
        assert len(client.turns) >= 2
        assert len({turn["goal_id"] for turn in client.turns}) == len(client.turns)
        assert any(event["type"] == "goal_updated" for event in client.events)
        assert any(
            event["type"] == "tool_execution_end"
            and event.get("toolName") == "read"
            and not event.get("isError")
            for event in client.events
        )
        assert result["workspace"]["source_root"] is None
        assert any(
            event["type"] == "tool_execution_end"
            and event.get("toolName") == "bash"
            and not event.get("isError")
            and "Ran 1 test" in str(event.get("result"))
            for event in client.events
        )
        assert any(
            event["type"] == "tool_execution_end"
            and event.get("toolName") == "bash"
            and event.get("isError")
            and "48" in str(event.get("result"))
            for event in client.events
        )
        native_tools = [tool.get("function", tool) for tool in bodies[0]["tools"]]
        bash = next(tool for tool in native_tools if tool["name"] == "bash")
        assert bash["parameters"]["properties"]["name"]["maxLength"] == 48
        assert any(
            "48" in json.dumps(body) and "49" in json.dumps(body) for body in bodies[2:]
        )
        model = os.environ.get("OH_MY_VLLM_MODEL")
        normalizer = (
            Path(__file__).resolve().parents[2] / "target/release/normalize-request"
        )
        if model and normalizer.is_file():
            guard = ContextGuard(Path(model), normalizer, directory)
            try:
                proof = guard.check(
                    bodies[0],
                    f"/v1/{'chat/completions' if api == 'chat' else 'responses'}",
                )
                assert proof["input_tokens"] > 0
                assert proof["reasoning_effort"] == "xhigh"
            finally:
                guard.close()
        assert all(
            body.get("reasoning_effort", body.get("reasoning", {}).get("effort"))
            == "xhigh"
            for body in bodies
        )
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
    assert not client.workspace.exists()
    assert json.loads((directory / "sandbox-cleanup.json").read_text())["closed"]


def test_sandbox_deletes_installed_tools_changes_and_child_processes(
    tmp_path, agentic_sandbox_image, monkeypatch
):
    directory = tmp_path / "isolation"
    directory.mkdir()
    sandbox = AgentSandbox(agentic_sandbox_image, directory)
    code = (
        "import pathlib,subprocess,time; "
        f"assert not pathlib.Path({str(Path.cwd())!r}).exists(); "
        "assert not pathlib.Path('/var/run/docker.sock').exists(); "
        "pathlib.Path('/usr/local/bin/sandbox-installed-tool')"
        ".write_text('temporary'); "
        "pathlib.Path('changed.txt').write_text('temporary'); "
        "subprocess.Popen(['python','-c','import time; time.sleep(120)']); "
        "print('isolated',flush=True); time.sleep(120)"
    )
    command = sandbox.command(["omp", "-c", code], {})
    command[command.index("/usr/local/bin/omp")] = "/usr/bin/python3"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    original_docker = agentic_sandbox.docker

    def fail_diff(*arguments, **kwargs):
        if arguments[:2] == ("container", "diff"):
            raise OSError("injected diff failure")
        return original_docker(*arguments, **kwargs)

    monkeypatch.setattr(agentic_sandbox, "docker", fail_diff)
    try:
        assert process.stdout.readline().strip() == "isolated"
        record = sandbox.inspect()
        assert len(record["mounts"]) == 1
        assert record["devices"] == []
        assert record["cap_drop"] == ["ALL"]
        assert (sandbox.root / "workspace/changed.txt").exists()
        host_pid = record["host_pid"]
        before = (
            Path(f"/proc/{host_pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
        )
    finally:
        sandbox.close()
        process.communicate(timeout=15)
    assert not sandbox.root.exists()
    assert not docker(
        "container", "ls", "--all", "--quiet", "--filter", f"name=^/{sandbox.name}$"
    )
    if Path(f"/proc/{host_pid}/stat").exists():
        assert (
            Path(f"/proc/{host_pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
            != before
        )
    assert not Path("/usr/local/bin/sandbox-installed-tool").exists()
    assert (
        "injected diff failure"
        in json.loads((directory / "sandbox-cleanup.json").read_text())["diff_error"]
    )


@pytest.mark.parametrize("failure", ["persist", "sdk"])
def test_rpc_cleanup_releases_real_sandbox_after_evidence_failure(
    tmp_path, agentic_sandbox_image, monkeypatch, failure
):
    directory = tmp_path / "client"
    directory.mkdir()
    client = RpcClient(
        "omp",
        "chat",
        0,
        directory,
        "http://127.0.0.1:1/v1",
        threading.Event(),
        30,
        goal_mode=True,
        sandbox_image=agentic_sandbox_image,
    )
    client.initialize()
    if failure == "persist":

        def fail():
            raise OSError("injected persistence failure")

        monkeypatch.setattr(client, "persist", fail)
    else:
        original_close = client.sdk_trace.close

        def fail(**kwargs):
            original_close(**kwargs)
            raise OSError("injected SDK failure")

        monkeypatch.setattr(client.sdk_trace, "close", fail)
    with pytest.raises(ValueError, match="injected"):
        client.close()
    assert client.process.poll() is not None
    assert not client.sandbox.root.exists()
    assert not docker(
        "container",
        "ls",
        "--all",
        "--quiet",
        "--filter",
        f"name=^/{client.sandbox.name}$",
    )
