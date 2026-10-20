import ast
import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ["OH_MY_VLLM_TEST_BINARY"]
MODEL = "qwen3.8-27b-fp8"


def wait_for(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.025)
    raise AssertionError("condition did not become true before its deadline")


def events(text):
    return [
        json.loads(line[5:])
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


class Service(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="oh-my-vllm-http-")
        self.addCleanup(self.directory.cleanup)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.process = subprocess.Popen(
            [
                BINARY,
                "--socket",
                self.directory.name + "/worker.ipc",
                "serve",
                "--listen",
                f"127.0.0.1:{self.port}",
                "--diagnostics-interval-seconds",
                "1",
                "--request-timeout-seconds",
                "5",
                "--shutdown-grace-seconds",
                "2",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop)
        self.worker = wait_for(lambda: self.diagnostics().get("worker_pid"))

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        worker = getattr(self, "worker", None)
        if worker:
            wait_for(lambda: not Path(f"/proc/{worker}").exists())

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        with contextlib.closing(connection):
            connection.request(
                method,
                path,
                None if body is None else json.dumps(body),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read().decode()
            return response.status, content

    def json(self, method, path, body=None, status=200):
        actual, content = self.request(method, path, body)
        self.assertEqual(actual, status, content)
        return json.loads(content)

    def diagnostics(self):
        return self.json("GET", "/diagnostics")

    def chat(self, **extra):
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "reasoning_effort": "off",
            "seed": 7,
            **extra,
        }

    def test_chat_and_responses_streams(self):
        models = self.json("GET", "/v1/models")
        self.assertEqual(models["data"][0]["id"], MODEL)
        plain = self.json("POST", "/v1/chat/completions", self.chat())
        status, stream = self.request(
            "POST",
            "/v1/chat/completions",
            self.chat(stream=True, stream_options={"include_usage": True}),
        )
        self.assertEqual(status, 200)
        self.assertIn("[DONE]", stream)
        chunks = events(stream)
        text = "".join(
            c["delta"].get("content", "") for event in chunks for c in event["choices"]
        )
        self.assertEqual(text, plain["choices"][0]["message"]["content"])
        self.assertEqual(
            chunks[-1]["usage"]["completion_tokens"],
            plain["usage"]["completion_tokens"],
        )
        self.assertTrue(chunks[-1]["usage"]["completion_tokens"] > 0)
        payload = {
            "model": MODEL,
            "input": "hello",
            "max_output_tokens": 128,
            "seed": 7,
            "reasoning": {"effort": "off"},
            "stream": True,
            "store": False,
        }
        status, stream = self.request("POST", "/v1/responses", payload)
        self.assertEqual(status, 200)
        chunks = events(stream)
        self.assertEqual(
            [e["sequence_number"] for e in chunks], list(range(len(chunks)))
        )
        self.assertEqual(chunks[-1]["type"], "response.completed")
        response = chunks[-1]["response"]
        self.assertEqual(response["output"][0]["content"][0]["text"], text)
        self.assertEqual(
            response["usage"]["output_tokens"], plain["usage"]["completion_tokens"]
        )

    def test_media_chat_responses_and_continuation(self):
        image = "data:image/png;base64," + "AQID" * 70000
        content = [
            {"type": "text", "text": "Compare these."},
            {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
            {
                "type": "video_url",
                "video_url": {"url": "https://example.invalid/v.mp4"},
            },
        ]
        chat = self.json(
            "POST",
            "/v1/chat/completions",
            self.chat(messages=[{"role": "user", "content": content}]),
        )
        response = self.json(
            "POST",
            "/v1/responses",
            {
                "model": MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Compare these."},
                            {
                                "type": "input_image",
                                "image_url": image,
                                "detail": "low",
                            },
                            {
                                "type": "input_video",
                                "video_url": "https://example.invalid/v.mp4",
                            },
                        ],
                    }
                ],
                "max_output_tokens": 128,
                "reasoning": {"effort": "off"},
                "seed": 7,
            },
        )
        self.assertEqual(
            chat["choices"][0]["message"]["content"],
            response["output"][0]["content"][0]["text"],
        )
        self.assertEqual(
            chat["usage"]["prompt_tokens"], response["usage"]["input_tokens"]
        )
        child = self.json(
            "POST",
            "/v1/responses",
            {
                "model": MODEL,
                "previous_response_id": response["id"],
                "input": "Continue.",
                "max_output_tokens": 16,
            },
        )
        self.assertGreater(
            child["usage"]["input_tokens"], response["usage"]["input_tokens"]
        )
        for invalid in (
            [{"type": "image_url", "image_url": {"url": "file:///tmp/p.png"}}],
            [content[1]] * 17,
        ):
            self.json(
                "POST",
                "/v1/chat/completions",
                self.chat(messages=[{"role": "user", "content": invalid}]),
                status=400,
            )

    def test_tools_schema_and_stored_history(self):
        function = {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "enum": ["hello"]}},
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        tools = [{"type": "function", **function}]
        payload = {
            "model": MODEL,
            "input": "lookup",
            "reasoning": {"effort": "off"},
            "tools": tools,
            "tool_choice": "required",
            "max_output_tokens": 256,
        }
        parent = self.json("POST", "/v1/responses", payload)
        call = parent["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(json.loads(call["arguments"]), {"key": "hello"})
        self.assertEqual(self.json("GET", "/v1/responses/" + parent["id"]), parent)
        child = self.json(
            "POST",
            "/v1/responses",
            {
                "model": MODEL,
                "previous_response_id": parent["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "found",
                    }
                ],
                "reasoning": {"effort": "off"},
                "max_output_tokens": 128,
            },
        )
        self.assertGreater(
            child["usage"]["input_tokens"], parent["usage"]["input_tokens"]
        )
        self.json("DELETE", "/v1/responses/" + parent["id"])
        self.json("GET", "/v1/responses/" + parent["id"], status=404)
        self.assertEqual(self.json("GET", "/v1/responses/" + child["id"]), child)
        schema = function["parameters"]
        result = self.json(
            "POST",
            "/v1/chat/completions",
            self.chat(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": True, "schema": schema},
                }
            ),
        )
        self.assertEqual(
            json.loads(result["choices"][0]["message"]["content"]), {"key": "hello"}
        )

    def test_validation_and_reasoning_budget(self):
        for invalid in [
            {"n": 2},
            {"unknown": True},
            {"top_p": -1},
            {"thinking_token_budget": -2},
            {"max_tokens": 262144},
            {"max_tokens": 8, "min_tokens": 9},
            {"min_tokens": -1},
        ]:
            self.json("POST", "/v1/chat/completions", self.chat(**invalid), status=400)
        result = self.json(
            "POST",
            "/v1/chat/completions",
            self.chat(reasoning_effort="high", thinking_token_budget=0),
        )
        self.assertEqual(
            result["choices"][0]["message"].get("reasoning_content", ""), ""
        )
        self.assertLessEqual(
            result["usage"]["completion_tokens_details"]["reasoning_tokens"], 1
        )
        self.assertTrue(result["choices"][0]["message"]["content"])
        epoch = self.diagnostics()["epoch"]
        stopped = self.json(
            "POST", "/v1/chat/completions", self.chat(seed=0, max_tokens=1, stop=["t"])
        )
        self.assertFalse(stopped["choices"][0]["message"]["content"])
        self.assertEqual(stopped["choices"][0]["finish_reason"], "stop")
        oversized = {"type": "string"}
        for _ in range(5):
            oversized = {"type": "array", "minItems": 64, "items": oversized}
        self.json(
            "POST",
            "/v1/chat/completions",
            self.chat(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "big", "schema": oversized},
                }
            ),
            status=400,
        )
        self.assertEqual(self.diagnostics()["epoch"], epoch)

    def test_cancellation_and_worker_recovery(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        large = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "array",
                    "minItems": 64,
                    "items": {"type": "string", "minLength": 4096},
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        }
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(
                self.chat(
                    stream=True,
                    max_tokens=65536,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "long", "schema": large},
                    },
                )
            ),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read(1)
        wait_for(lambda: self.diagnostics()["active_requests"] > 0)
        response.close()
        connection.close()
        wait_for(lambda: self.diagnostics()["active_requests"] == 0)
        wait_for(lambda: self.diagnostics().get("worker", {}).get("requests") == 0)
        worker = self.diagnostics()["worker_pid"]
        parent = Path(f"/proc/{worker}/status").read_text().split("PPid:")[1].split()[0]
        self.assertEqual(int(parent), self.process.pid)
        os.kill(worker, signal.SIGKILL)
        wait_for(lambda: self.diagnostics().get("epoch", 0) >= 2)
        self.worker = self.diagnostics()["worker_pid"]
        self.assertNotEqual(self.worker, worker)
        result = self.json("POST", "/v1/chat/completions", self.chat())
        self.assertTrue(result["choices"][0]["message"]["content"])
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "100")
        connection.endheaders()
        connection.close()
        wait_for(lambda: self.diagnostics()["active_requests"] == 0)


class Offline(unittest.TestCase):
    def test_run_bench_and_prefix_reuse(self):
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-cli-") as directory:
            for mode in ("none", "mtp"):
                result = subprocess.run(
                    [
                        BINARY,
                        "--socket",
                        directory + "/worker.ipc",
                        "--speculative-mode",
                        mode,
                        "run",
                        "--tokens",
                        "1",
                        "2",
                        "--max-tokens",
                        "8",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(
                    line
                    for line in result.stdout.splitlines()
                    if line.startswith("output token ids:")
                )
                self.assertEqual(len(ast.literal_eval(line.split(":", 1)[1])), 8)
            result = subprocess.run(
                [
                    BINARY,
                    "--socket",
                    directory + "/bench.ipc",
                    "bench",
                    "--batch-size",
                    "2",
                    "--input-len",
                    "786",
                    "--output-len",
                    "8",
                    "--prefix-hit",
                    "--warmup",
                    "0",
                    "--repetitions",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            row = next(
                json.loads(line.split(" ", 1)[1])
                for line in result.stdout.splitlines()
                if line.startswith("BENCH_RESULT ")
            )
            self.assertEqual(row["initial_prefix_hit_tokens"], 2 * 785)


if __name__ == "__main__":
    unittest.main()
