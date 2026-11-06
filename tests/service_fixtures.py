import contextlib
import http.client
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from tests.worker_fixture import install

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


class ServiceFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="oh-my-vllm-http-")
        self.addCleanup(self.directory.cleanup)
        environment = install(self.directory.name)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.process = subprocess.Popen(
            [
                BINARY,
                "--model",
                self.directory.name,
                "--socket",
                self.directory.name + "/worker.ipc",
                "--num-gpu-blocks",
                "128",
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
            env=environment,
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
