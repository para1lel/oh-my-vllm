"""Real Rust HTTP/ZMQ integration with a scripted CPU worker, not GPU acceptance."""

import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


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
            "model": "qwen3.5-27b-fp8",
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
            {"unsupported": True},
        ]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/chat/completions", self.base(**extra))
            self.assertEqual(error.exception.code, 400)

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

    def test_z_worker_failure_is_terminal(self):
        with self.request(
            "/responses", self.base(True, input="worker-fail", stream=True)
        ) as response:
            events = [
                json.loads(line[6:])
                for line in response.read().decode().splitlines()
                if line.startswith("data: ")
            ]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["message"], "worker failed")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/responses", self.base(True))
        self.assertEqual(error.exception.code, 503)

    def test_cancel_does_not_break_other_request(self):
        with self.request(
            "/chat/completions",
            self.base(stream=True, messages=[{"role": "user", "content": "long"}]),
        ) as response:
            response.readline()
        with concurrent.futures.ThreadPoolExecutor() as pool:

            def complete(_):
                with self.request("/chat/completions", self.base()) as response:
                    return json.load(response)["choices"][0]["finish_reason"]

            self.assertEqual(list(pool.map(complete, range(2))), ["stop", "stop"])


if __name__ == "__main__":
    unittest.main()
