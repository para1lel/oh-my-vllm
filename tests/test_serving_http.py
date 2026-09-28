"""Real Rust HTTP/ZMQ integration with a scripted CPU worker, not GPU acceptance."""

import concurrent.futures
import ctypes
import http.client
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def open_pidfd(pid):
    """Use libc when this Python build omits os.pidfd_open."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="oh-my-vllm-http-")
        cls.directory = Path(cls.temp.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}/v1"
        wrapper = cls.directory / "worker"
        wrapper.write_text(
            f'#!/bin/sh\nexec {PYTHON} {ROOT}/tests/fixtures/serving_worker.py "$@"\n'
        )
        wrapper.chmod(0o700)
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(wrapper),
            "VLLM_TARGET_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "OH_MY_VLLM_FIXTURE_CLEANUP": str(cls.directory / "cleanup"),
            "OH_MY_VLLM_FIXTURE_BLOCKS": str(cls.directory / "blocks.json"),
            "OH_MY_VLLM_FIXTURE_OVERLAP": str(cls.directory / "overlap"),
            "OH_MY_VLLM_FIXTURE_ABORTS": str(cls.directory / "aborts"),
            "OH_MY_VLLM_PREPARE_TIMEOUT_MS": "500",
        }
        cls.log = (cls.directory / "server.log").open("w+")
        cls.process = subprocess.Popen(
            [
                str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                "--socket",
                str(cls.directory / "worker.ipc"),
                "--num-speculative-tokens",
                "4",
                "serve",
                "--listen",
                f"127.0.0.1:{port}",
                "--response-ttl-seconds",
                "2",
            ],
            env=environment,
            stdout=cls.log,
            stderr=cls.log,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                cls.log.seek(0)
                raise RuntimeError(cls.log.read())
            try:
                with urllib.request.urlopen(cls.url + "/models", timeout=0.2):
                    return
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        raise RuntimeError("fixture HTTP service did not start")

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            os.killpg(cls.process.pid, signal.SIGINT)
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(cls.process.pid, signal.SIGKILL)
                cls.process.wait()
        cls.log.close()
        cls.temp.cleanup()

    def request(self, path, payload=None, method=None):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        return urllib.request.urlopen(req, timeout=20)

    def base(self, responses=False, **extra):
        return {
            "model": "qwen3.8-27b-fp8",
            **(
                {"input": "Hello"}
                if responses
                else {"messages": [{"role": "user", "content": "Hello"}]}
            ),
            **extra,
        }

    def test_chat_stream_matches_nonstream(self):
        with self.request("/chat/completions", self.base()) as response:
            normal = json.load(response)
        with self.request(
            "/chat/completions",
            self.base(stream=True, stream_options={"include_usage": True}),
        ) as response:
            rows = [
                line[6:].strip()
                for line in response.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        self.assertEqual(rows[-1], "[DONE]")
        chunks = [json.loads(row) for row in rows[:-1]]
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]
        )
        self.assertEqual(text, normal["choices"][0]["message"]["content"])
        self.assertGreater(chunks[-1]["usage"]["completion_tokens"], 0)

    def test_sigkill_parent_reaps_scripted_worker(self):
        """PDEATHSIG must work even when Rust cannot run Drop or a signal handler."""
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-parent-death-") as tmp:
            directory = Path(tmp)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            wrapper = directory / "worker"
            wrapper.write_text(
                f"#!/bin/sh\nexec {PYTHON} "
                f'{ROOT}/tests/fixtures/serving_worker.py "$@"\n'
            )
            wrapper.chmod(0o700)
            environment = os.environ | {
                "OH_MY_VLLM_WORKER_PYTHON": str(wrapper),
                "CUDA_VISIBLE_DEVICES": "",
            }
            with (directory / "server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                        "--socket",
                        str(directory / "worker.ipc"),
                        "serve",
                        "--listen",
                        f"127.0.0.1:{port}",
                    ],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                worker_pidfd = None
                try:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail((directory / "server.log").read_text())
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/v1/models", timeout=0.2
                            ):
                                break
                        except (OSError, urllib.error.URLError):
                            time.sleep(0.05)
                    else:
                        self.fail("parent-death fixture did not start")
                    children = (
                        Path(f"/proc/{process.pid}/task/{process.pid}/children")
                        .read_text()
                        .split()
                    )
                    self.assertEqual(len(children), 1)
                    worker_pidfd = open_pidfd(int(children[0]))
                    process.kill()
                    process.wait(timeout=5)
                    self.assertTrue(
                        select.select([worker_pidfd], [], [], 5)[0],
                        "scripted worker survived Rust SIGKILL",
                    )
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    if worker_pidfd is not None:
                        if not select.select([worker_pidfd], [], [], 0)[0]:
                            with suppress(ProcessLookupError):
                                signal.pidfd_send_signal(worker_pidfd, signal.SIGKILL)
                        os.close(worker_pidfd)

    def test_responses_stream_and_stored_continuation(self):
        with self.request(
            "/responses", self.base(True, stream=True, instructions="Only this turn")
        ) as response:
            events = [
                json.loads(line[6:])
                for line in response.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            [e["sequence_number"] for e in events], list(range(len(events)))
        )
        rid = events[-1]["response"]["id"]
        with self.request("/responses/" + rid) as response:
            self.assertEqual(json.load(response)["status"], "completed")
        with self.request(
            "/responses", self.base(True, previous_response_id=rid)
        ) as response:
            self.assertEqual(json.load(response)["status"], "completed")
        with self.request("/responses/" + rid, method="DELETE") as response:
            self.assertTrue(json.load(response)["deleted"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses/" + rid)
        self.assertEqual(error.exception.code, 404)

    def test_forced_tool_and_roundtrip(self):
        tool = {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
        body = self.base(
            tools=[tool], tool_choice="required", parallel_tool_calls=False
        )
        with self.request("/chat/completions", body) as response:
            reply = json.load(response)
        message = reply["choices"][0]["message"]
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
        call = message["tool_calls"][0]
        self.assertEqual(
            json.loads(call["function"]["arguments"]), {"path": "README.md"}
        )
        body["tool_choice"] = "auto"
        body["messages"].extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "File contents",
                },
            ]
        )
        with self.request("/chat/completions", body) as response:
            self.assertEqual(json.load(response)["choices"][0]["finish_reason"], "stop")

    def test_json_schema_and_bad_parameters(self):
        body = self.base(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        with self.request("/chat/completions", body) as response:
            self.assertEqual(
                json.loads(json.load(response)["choices"][0]["message"]["content"]),
                {"n": 123},
            )
        for extra in [
            {"reasoning_effort": 123},
            {"reasoning_effort": "unknown"},
            {"max_tokens": 0},
            {"repetition_penalty": 1e-39},
            {"top_p": 0},
            {"temperature": -1},
            {"top_k": 1.5},
            {"frequency_penalty": 3},
            {"unsupported": True},
        ]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/chat/completions", self.base(**extra))
            self.assertEqual(error.exception.code, 400)

    def test_malformed_nested_schema_is_client_error(self):
        for schema in (
            {"type": "object", "properties": []},
            {"type": "object", "required": "name"},
            {"type": "object", "$defs": []},
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
            {"type": "string", "enum": ["x"], "$ref": "#/enum/" + "0" * 5000},
            {"type": "string", "pattern": "["},
            {"type": "string", "pattern": "("},
        ):
            with self.subTest(schema=schema):
                body = self.base(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "bad", "schema": schema},
                    }
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request("/chat/completions", body)
                self.assertEqual(error.exception.code, 400)
        for malformed in ("bad", [], 3):
            with self.subTest(json_schema=malformed):
                body = self.base(
                    response_format={"type": "json_schema", "json_schema": malformed}
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request("/chat/completions", body)
                self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(
                "/chat/completions",
                self.base(
                    messages=[
                        {"role": "assistant", "content": "hi", "tool_calls": "bad"}
                    ]
                ),
            )
        self.assertEqual(error.exception.code, 400)
        for schema in (
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/S/type"}},
                "$defs": {"S": {"type": "string"}},
            },
        ):
            with self.subTest(schema=schema):
                tool = {
                    "type": "function",
                    "function": {"name": "broken", "parameters": schema},
                }
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request("/chat/completions", self.base(tools=[tool]))
                self.assertEqual(error.exception.code, 400)

    def test_prepare_error_kind_and_axum_rejection_status(self):
        for content, status, kind in (
            ("prepare-validation", 400, "invalid_request_error"),
            ("prepare-internal", 500, "server_error"),
        ):
            with self.subTest(content=content):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request("/responses", self.base(True, input=content))
                self.assertEqual(error.exception.code, status)
                self.assertEqual(json.load(error.exception)["error"]["type"], kind)
        request = urllib.request.Request(
            self.url + "/responses", data=b"{}", headers={"Content-Type": "text/plain"}
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 415)
        request = urllib.request.Request(
            self.url + "/responses",
            data=b" " * ((8 << 20) + 1),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 413)

    def test_prompt_file_rejects_out_of_range_token_before_worker_launch(self):
        prompt_file = self.directory / "invalid-tokens.txt"
        prompt_file.write_text("248320\n")
        result = subprocess.run(
            [
                str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                "--model",
                "/nonexistent-checkpoint",
                "run",
                "--prompt-file",
                str(prompt_file),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prompt token is outside", result.stderr)

    def test_late_prepare_reply_does_not_shift_next_rpc(self):
        aborts = self.directory / "aborts"
        before = aborts.read_text().splitlines() if aborts.exists() else []
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses", self.base(True, input="slow-prepare"))
        self.assertEqual(error.exception.code, 503)
        with self.request("/responses", self.base(True)) as response:
            self.assertEqual(json.load(response)["status"], "completed")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            lines = aborts.read_text().splitlines() if aborts.exists() else []
            if len(lines) > len(before):
                break
            time.sleep(0.02)
        self.assertGreater(len(lines), len(before))
        self.log.flush()
        log = (self.directory / "server.log").read_text()
        pairs = re.findall(
            r"discarding reply to cancelled RPC received=(\d+) expected=(\d+)", log
        )
        self.assertTrue(any(int(old) < int(new) for old, new in pairs), log[-2000:])

    def test_capacity_rejection_releases_prepared_worker_state(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        address = f"http://127.0.0.1:{port}/v1"
        aborts = self.directory / "capacity-aborts"
        log_path = self.directory / "capacity-server.log"
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(self.directory / "worker"),
            "OH_MY_VLLM_FIXTURE_ABORTS": str(aborts),
            "OH_MY_VLLM_PREPARE_TIMEOUT_MS": "5000",
            "CUDA_VISIBLE_DEVICES": "",
        }
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                    "--socket",
                    str(self.directory / "capacity-worker.ipc"),
                    "--scheduler-blocks",
                    "8",
                    "serve",
                    "--listen",
                    f"127.0.0.1:{port}",
                ],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(address + "/models", timeout=0.2):
                            break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.05)
                else:
                    self.fail("capacity fixture did not start")
                payload = self.base(True, input="hello " * 7000, max_output_tokens=1)
                request = urllib.request.Request(
                    address + "/responses",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(error.exception.code, 413)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not aborts.exists():
                    time.sleep(0.02)
                self.assertTrue(aborts.exists(), log_path.read_text())
                self.assertEqual(len(aborts.read_text().splitlines()), 1)
                payload["input"] = "Hello"
                request = urllib.request.Request(
                    address + "/responses",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(json.load(response)["status"], "incomplete")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    def test_sigint_drains_active_request_and_shuts_down_worker(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        address = f"http://127.0.0.1:{port}/v1"
        marker = self.directory / "graceful-shutdown"
        log_path = self.directory / "graceful-shutdown.log"
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(self.directory / "worker"),
            "OH_MY_VLLM_FIXTURE_SHUTDOWN": str(marker),
            "CUDA_VISIBLE_DEVICES": "",
        }
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                    "--socket",
                    str(self.directory / "graceful-worker.ipc"),
                    "serve",
                    "--listen",
                    f"127.0.0.1:{port}",
                ],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(address + "/models", timeout=0.2):
                            break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.05)
                else:
                    self.fail("graceful shutdown fixture did not start")
                request = urllib.request.Request(
                    address + "/responses",
                    data=json.dumps(
                        self.base(
                            True,
                            input="long hold-active",
                            stream=True,
                            max_output_tokens=1500,
                        )
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as stream:
                    self.assertIn(b"response.created", stream.readline())
                    process.send_signal(signal.SIGINT)
                    body = stream.read().decode()
                self.assertIn("server shutting down", body)
                self.assertEqual(process.wait(timeout=5), 0, log_path.read_text())
                self.assertEqual(marker.read_text(), "shutdown received\n")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def test_sigint_has_deadline_for_client_that_stops_reading_sse(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        address = f"http://127.0.0.1:{port}/v1"
        marker = self.directory / "stalled-shutdown"
        flood = self.directory / "stalled-steps"
        log_path = self.directory / "stalled-shutdown.log"
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(self.directory / "worker"),
            "OH_MY_VLLM_FIXTURE_SHUTDOWN": str(marker),
            "OH_MY_VLLM_FIXTURE_FLOOD": str(flood),
            "CUDA_VISIBLE_DEVICES": "",
        }
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                    "--socket",
                    str(self.directory / "stalled-worker.ipc"),
                    "serve",
                    "--listen",
                    f"127.0.0.1:{port}",
                    "--shutdown-grace-seconds",
                    "1",
                ],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(address + "/models", timeout=0.2):
                            break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.05)
                else:
                    self.fail("stalled shutdown fixture did not start")
                payload = json.dumps(
                    self.base(
                        True, input="flood-active", stream=True, max_output_tokens=1000
                    )
                ).encode()
                with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                    client.sendall(
                        b"POST /v1/responses HTTP/1.1\r\n"
                        b"Host: 127.0.0.1\r\n"
                        b"Content-Type: application/json\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                        + payload
                    )
                    status = b""
                    while b"\r\n" not in status and len(status) < 8192:
                        chunk = client.recv(4096)
                        if not chunk:
                            self.fail("server closed before the HTTP status line")
                        status += chunk
                    self.assertIn(b"200 OK", status.split(b"\r\n", 1)[0])
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        steps = flood.read_text().splitlines() if flood.exists() else []
                        if len(steps) >= 16:
                            break
                        time.sleep(0.02)
                    else:
                        self.fail("fixture did not fill the stalled stream")
                    process.send_signal(signal.SIGINT)
                    self.assertEqual(process.wait(timeout=5), 0, log_path.read_text())
                self.assertEqual(marker.read_text(), "shutdown received\n")
                self.assertIn(
                    "HTTP connections did not drain before shutdown deadline",
                    log_path.read_text(),
                )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def test_stalled_sse_pauses_and_recovers_without_blocking_another_request(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        steps = self.directory / "slow-stream-steps"
        log_path = self.directory / "slow-stream-server.log"
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(self.directory / "worker"),
            "OH_MY_VLLM_FIXTURE_STEPS": str(steps),
            "CUDA_VISIBLE_DEVICES": "",
        }
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                    "--socket",
                    str(self.directory / "slow-stream-worker.ipc"),
                    "serve",
                    "--listen",
                    f"127.0.0.1:{port}",
                ],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            try:
                address = f"http://127.0.0.1:{port}/v1"
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(address + "/models", timeout=0.2):
                            break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.05)
                else:
                    self.fail("slow-stream fixture did not start")

                connection.connect()
                connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                payload = self.base(
                    stream=True,
                    store=False,
                    messages=[{"role": "user", "content": "slow-stream"}],
                    max_tokens=750,
                    reasoning_effort="off",
                )
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(payload),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                first = response.readline()
                self.assertTrue(first.startswith(b"data: "), first)
                request_id = int(json.loads(first[6:])["id"].rsplit("_", 1)[1])

                def slow_steps():
                    if not steps.exists():
                        return 0
                    return sum(
                        json.loads(line)["rid"] == request_id
                        for line in steps.read_text().splitlines()
                    )

                previous = -1
                still_count = 0
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    count = slow_steps()
                    if count > 256 and count == previous:
                        still_count += 1
                    else:
                        still_count = 0
                    if still_count >= 10:
                        break
                    previous = count
                    time.sleep(0.05)
                else:
                    self.fail(f"slow stream did not pause after 256 steps: {previous}")
                self.assertLess(count, 750, "slow stream completed before resume")
                with urllib.request.urlopen(
                    urllib.request.Request(
                        address + "/chat/completions",
                        data=json.dumps(self.base()).encode(),
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=10,
                ) as other:
                    self.assertEqual(
                        json.load(other)["choices"][0]["finish_reason"], "stop"
                    )
                self.assertEqual(
                    slow_steps(), count, "slow stream advanced while paused"
                )

                content_steps = []
                saw_done = False
                while line := response.readline():
                    if not line.startswith(b"data: "):
                        continue
                    if line.strip() == b"data: [DONE]":
                        saw_done = True
                        break
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        content_steps.append(int(delta.split("|", 1)[0]))
                self.assertTrue(saw_done, "slow stream ended without [DONE]")
                self.assertEqual(content_steps, list(range(1, 751)))
                self.assertEqual(slow_steps(), 750)
            finally:
                connection.close()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)

    def test_length_inside_tool_is_not_server_error(self):
        tool = {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
        body = self.base(
            tools=[tool], tool_choice="required", max_tokens=12, reasoning_effort="off"
        )
        with self.request("/chat/completions", body) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertNotIn("tool_calls", result["choices"][0]["message"])
        with self.request(
            "/responses", self.base(True, max_output_tokens=1)
        ) as response:
            result = json.load(response)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_details"]["reason"], "max_output_tokens")

    def test_expiry_and_store_false(self):
        with self.request("/responses", self.base(True, store=False)) as response:
            rid = json.load(response)["id"]
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses/" + rid)
        self.assertEqual(error.exception.code, 404)
        with self.request("/responses", self.base(True)) as response:
            rid = json.load(response)["id"]
        time.sleep(2.05)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses/" + rid)
        self.assertEqual(error.exception.code, 404)

    def test_request_failure_does_not_stop_engine(self):
        with self.request(
            "/responses", self.base(True, input="worker-fail", stream=True)
        ) as response:
            events = [
                json.loads(line[6:])
                for line in response.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["message"], "fixture request failure")
        with self.request("/responses", self.base(True)) as response:
            self.assertEqual(json.load(response)["status"], "completed")

    def test_one_failed_request_leaves_active_stream_running(self):
        overlap = self.directory / "overlap"
        overlap.unlink(missing_ok=True)
        with self.request(
            "/responses",
            self.base(True, input="long", max_output_tokens=256, stream=True),
        ) as continuing:
            self.assertIn(b"response.created", continuing.readline())
            with self.request(
                "/responses", self.base(True, input="worker-fail", stream=True)
            ) as failing:
                events = [
                    json.loads(line[6:])
                    for line in failing.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
            self.assertEqual(events[-1]["message"], "fixture request failure")
            self.assertTrue(overlap.exists())
            self.assertTrue(overlap.read_text().strip())
            remaining = continuing.read().decode()
            self.assertIn("response.incomplete", remaining)
            self.assertNotIn('"type":"error"', remaining)

    def test_z_worker_failure_is_terminal(self):
        with self.request(
            "/responses", self.base(True, input="worker-fatal", stream=True)
        ) as response:
            events = [
                json.loads(line[6:])
                for line in response.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        self.assertEqual(events[-1]["message"], "worker failed")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses", self.base(True))
        self.assertEqual(error.exception.code, 503)

    def test_y_active_limit_returns_503(self):
        streams = []
        try:
            for _ in range(64):
                streams.append(
                    self.request(
                        "/responses",
                        self.base(
                            True,
                            input="long hold-active",
                            stream=True,
                            max_output_tokens=1500,
                        ),
                    )
                )
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/responses", self.base(True))
            self.assertEqual(error.exception.code, 503)
            self.assertEqual(
                json.load(error.exception)["error"]["type"], "server_error"
            )
        finally:
            for stream in streams:
                stream.close()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                try:
                    with self.request("/responses", self.base(True)) as response:
                        self.assertEqual(json.load(response)["status"], "completed")
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 503:
                        raise
                    time.sleep(0.05)
            else:
                self.fail(
                    "active request capacity did not recover after stream closure"
                )

    def test_cancel_does_not_break_other_request(self):
        cleanup = self.directory / "cleanup"
        blocks = self.directory / "blocks.json"
        initial = {"owners": [], "free_fa": 99, "free_mamba": 99}
        self.assertEqual(json.loads(blocks.read_text()), initial)

        before = cleanup.read_text().splitlines() if cleanup.exists() else []
        with self.request(
            "/chat/completions",
            self.base(
                stream=True,
                messages=[{"role": "user", "content": "long hold-active"}],
                max_tokens=1500,
            ),
        ) as response:
            response.readline()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                allocated = json.loads(blocks.read_text())
                if (
                    allocated["owners"]
                    and allocated["free_fa"] < 99
                    and allocated["free_mamba"] < 99
                ):
                    break
                time.sleep(0.02)
            self.assertTrue(allocated["owners"], "fixture saw no allocated request")
            self.assertLess(allocated["free_fa"], 99)
            self.assertLess(allocated["free_mamba"], 99)
        with concurrent.futures.ThreadPoolExecutor() as pool:

            def complete(_):
                with self.request("/chat/completions", self.base()) as response:
                    return json.load(response)["choices"][0]["finish_reason"]

            self.assertEqual(list(pool.map(complete, range(2))), ["stop", "stop"])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            lines = cleanup.read_text().splitlines() if cleanup.exists() else []
            if len(lines) > len(before):
                break
            time.sleep(0.02)
        self.assertGreater(len(lines), len(before))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if json.loads(blocks.read_text()) == initial:
                break
            time.sleep(0.02)
        self.assertEqual(json.loads(blocks.read_text()), initial)

    def test_cancel_reclaims_full_shared_scheduler_pool(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        address = f"http://127.0.0.1:{port}/v1"
        blocks = self.directory / "small-pool-blocks.json"
        steps = self.directory / "small-pool-steps"
        shutdown = self.directory / "small-pool-shutdown"
        log_path = self.directory / "small-pool-server.log"
        environment = os.environ | {
            "OH_MY_VLLM_WORKER_PYTHON": str(self.directory / "worker"),
            "OH_MY_VLLM_FIXTURE_BLOCKS": str(blocks),
            "OH_MY_VLLM_FIXTURE_STEPS": str(steps),
            "OH_MY_VLLM_FIXTURE_SHUTDOWN": str(shutdown),
            "CUDA_VISIBLE_DEVICES": "",
        }
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    str(ROOT / "target/debug/oh-my-vllm-zmq-worker"),
                    "--socket",
                    str(self.directory / "small-pool-worker.ipc"),
                    "--scheduler-blocks",
                    "5",
                    "serve",
                    "--listen",
                    f"127.0.0.1:{port}",
                ],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(log_path.read_text())
                    try:
                        with urllib.request.urlopen(address + "/models", timeout=0.2):
                            break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.05)
                else:
                    self.fail("small scheduler pool fixture did not start")

                # Five shared physical blocks leave four usable slots. Crossing
                # position 784 needs two FA pages and two GDN state slots.
                payload = self.base(
                    True, input="long hold-active", max_output_tokens=900
                )
                first = urllib.request.Request(
                    address + "/responses",
                    data=json.dumps(payload | {"stream": True}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(first, timeout=5) as stream:
                    self.assertIn(b"response.created", stream.readline())
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        observed = json.loads(blocks.read_text())
                        lines = steps.read_text().splitlines() if steps.exists() else []
                        if len(lines) >= 2 and observed["owners"]:
                            break
                        time.sleep(0.02)
                    self.assertGreaterEqual(len(lines), 2)
                    self.assertEqual(observed["free_fa"], 98)
                    self.assertEqual(observed["free_mamba"], 98)

                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    observed = json.loads(blocks.read_text())
                    if not observed["owners"]:
                        break
                    time.sleep(0.02)
                self.assertEqual(observed["owners"], [])

                # The same demand must cross 784 after cancellation. A leaked
                # Rust block leaves fewer than four allocatable slots.
                payload["input"] = "long"
                second = urllib.request.Request(
                    address + "/responses",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(second, timeout=15) as response:
                    self.assertEqual(json.load(response)["status"], "incomplete")
                recorded = [json.loads(line) for line in steps.read_text().splitlines()]
                first_id = recorded[0]["rid"]
                self.assertTrue(
                    any(
                        row["rid"] != first_id and row["fa"] >= 2 and row["mamba"] >= 2
                        for row in recorded
                    ),
                    "replacement request never occupied both FA and GDN pages",
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    observed = json.loads(blocks.read_text())
                    if observed == {"owners": [], "free_fa": 99, "free_mamba": 99}:
                        break
                    time.sleep(0.02)
                self.assertEqual(
                    observed, {"owners": [], "free_fa": 99, "free_mamba": 99}
                )
                self.assertIn("fa_pool_blocks=5", log_path.read_text())
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        self.assertEqual(shutdown.read_text(), "shutdown received\n")


if __name__ == "__main__":
    unittest.main()
