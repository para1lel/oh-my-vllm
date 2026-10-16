import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

from oh_my_vllm.kernels.native import cache_directory
from oh_my_vllm.worker.tokenizer import Tokenization

from .roofline import lower_bound

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = [("ordinary", batch, 32768) for batch in (1, 2, 4)] + [
    ("mtp4", batch, length) for batch in (1, 2, 4) for length in (32768, 65536, 131072)
]


def post(port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        value = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {value}")
        return value
    finally:
        connection.close()


def source_identity():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = ROOT / name
        if path.is_file() and (
            name.startswith(("python/", "crates/", "development/serving/"))
            or name
            in {
                "Cargo.toml",
                "Cargo.lock",
                "requirements/runtime.txt",
                "development/kernels/work.py",
                "development/kernels/roofline.py",
            }
        ):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prompt(tokenizer, length, identifier):
    options = {
        "messages": [{"role": "user", "content": ""}],
        "effort": "off",
        "tools": [],
        "tool_choice": "none",
    }
    copies = length
    for _ in range(10):
        text = (
            f"Request {identifier}.\n"
            + " x" * copies
            + "\nContinue with a detailed account."
        )
        options["messages"][0]["content"] = text
        actual = len(tokenizer.prepare(options))
        if actual == length:
            return options["messages"]
        copies += length - actual
    raise ValueError("cannot construct the exact prompt length")


class Service:
    def __init__(self, arguments, mode):
        self.arguments, self.mode = arguments, mode
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.ipc = Path(f"/tmp/oh-my-vllm-benchmark-{os.getpid()}.ipc")
        self.trace = arguments.artifacts / f"{mode}-execution.jsonl"
        self.log = (arguments.artifacts / f"{mode}-service.log").open("x")
        environment = dict(os.environ, OH_MY_VLLM_EXECUTION_TRACE=str(self.trace))
        command = [
            str(arguments.binary),
            "--model",
            str(arguments.model),
            "--socket",
            str(self.ipc),
            "--num-gpu-blocks",
            str(arguments.blocks),
            "--mamba-blocks",
            str(arguments.mamba_blocks),
            "--max-num-batched-tokens",
            "32768",
            "--max-num-seqs",
            "4",
        ]
        if mode == "mtp4":
            command += ["--num-speculative-tokens", "4", "--speculative-mode", "mtp"]
        command += [
            "serve",
            "--listen",
            f"127.0.0.1:{self.port}",
            "--request-timeout-seconds",
            "7200",
            "--worker-step-timeout-seconds",
            "600",
        ]
        self.process = subprocess.Popen(
            command,
            env=environment,
            cwd=ROOT,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def ready(self):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"service exited: {self.mode}")
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            try:
                connection.request("GET", "/diagnostics")
                response = connection.getresponse()
                if response.status == 200:
                    return
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(0.2)
        raise TimeoutError("service initialization")

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()
        if self.ipc.exists():
            raise RuntimeError("service did not remove its IPC socket")

    def run(self, tokenizer, batch, length, output):
        nonce = uuid.uuid4().hex
        bodies = [
            {
                "model": "qwen3.8-27b-fp8",
                "messages": prompt(tokenizer, length, f"{nonce}-{index}"),
                "max_tokens": output,
                "min_tokens": output,
                "temperature": 0,
                "presence_penalty": 0,
                "repetition_guard": False,
                "reasoning_effort": "off",
                "seed": index,
            }
            for index in range(batch)
        ]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
            responses = list(pool.map(lambda body: post(self.port, body), bodies))
        seconds = time.perf_counter() - started
        for value in responses:
            usage = value["usage"]
            if (
                usage["prompt_tokens"] != length
                or usage["completion_tokens"] != output
                or usage["prompt_tokens_details"]["cached_tokens"] != 0
            ):
                raise ValueError(f"token count or cold-prefix check failed: {usage}")
        ids = {int(value["id"].rsplit("_", 1)[1]) for value in responses}
        events = [json.loads(line) for line in self.trace.read_text().splitlines()]
        bound = lower_bound(events, ids, output)
        limit = bound["seconds"] * 5 / 3
        applicable = output == 8192 and (self.mode, batch, length) in CONFIGURATIONS
        return {
            "mode": self.mode,
            "batch": batch,
            "input_tokens": length,
            "output_tokens": output,
            "seconds": seconds,
            "lower_bound": bound,
            "limit_seconds": limit,
            "efficiency": bound["seconds"] / seconds,
            "gate_applicable": applicable,
            "passed": applicable and seconds <= limit,
            "responses": responses,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--case",
        choices=[f"{mode}:{batch}:{length}" for mode, batch, length in CONFIGURATIONS],
        action="append",
    )
    parser.add_argument("--output-tokens", type=int, default=8192)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--mamba-blocks", type=int, default=1024)
    arguments = parser.parse_args()
    arguments.artifacts = arguments.artifacts.resolve()
    arguments.binary = arguments.binary.resolve()
    if arguments.artifacts.is_relative_to(ROOT) or arguments.output_tokens < 1:
        parser.error("use external artifacts and a positive output token count")
    arguments.artifacts.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenization(arguments.model)
    identity = {
        "source": source_identity(),
        "binary": hashlib.sha256(arguments.binary.read_bytes()).hexdigest(),
        "native": {
            name: json.loads(path.read_text())
            for name in ("device", "ffn", "fa", "gdn", "mtp", "sample", "vision")
            if (path := cache_directory() / f"{name}.json").exists()
        },
        "device": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
    }
    (arguments.artifacts / "identity.json").write_text(json.dumps(identity, indent=2))
    selected = set(arguments.case or [])
    configurations = [
        item
        for item in CONFIGURATIONS
        if not selected or f"{item[0]}:{item[1]}:{item[2]}" in selected
    ]
    results = []
    for mode in ("ordinary", "mtp4"):
        cases = [item for item in configurations if item[0] == mode]
        if not cases:
            continue
        service = Service(arguments, mode)
        try:
            service.ready()
            for batch in sorted({item[1] for item in cases}):
                service.run(tokenizer, batch, 64, 16)
            for _, batch, length in cases:
                print(
                    json.dumps(
                        {
                            "event": "start",
                            "mode": mode,
                            "batch": batch,
                            "input": length,
                        }
                    ),
                    flush=True,
                )
                result = service.run(tokenizer, batch, length, arguments.output_tokens)
                results.append(result)
                (arguments.artifacts / f"{mode}-{batch}-{length}.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "mode",
                                "batch",
                                "input_tokens",
                                "seconds",
                                "limit_seconds",
                                "efficiency",
                                "gate_applicable",
                                "passed",
                            )
                        }
                    ),
                    flush=True,
                )
        finally:
            service.close()
    if source_identity() != identity["source"]:
        raise RuntimeError("repository sources changed during the benchmark")
    passed = len(results) == len(CONFIGURATIONS) and all(
        result["passed"] for result in results
    )
    (arguments.artifacts / "summary.json").write_text(
        json.dumps({"passed": passed, "configurations": len(results)}, indent=2)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
