import concurrent.futures
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from . import deepseek_accounting as accounting
from .benchmark import ROOT
from .deepseek_accounting import BindingError, efficiency_percent
from .deepseek_benchmark_config import (
    effective_cache_configuration,
    effective_service_configuration,
)
from .deepseek_benchmark_reference import reference_configuration, runtime_reference
from .deepseek_matrix import case_id
from .deepseek_runtime_trace import assemble_trace, read_rank_events

SHUTDOWN_GRACE_SECONDS = 35


DEFAULT_REQUEST_TIMEOUT_SECONDS = 7200


SERVED_MODEL = "deepseek-v4.1-flash"


def post(port, body, *, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
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


def exact_prompt(tokenizer, length, identifier):
    messages = [{"role": "user", "content": ""}]
    copies = max(1, length // 2)
    for _ in range(12):
        messages[0]["content"] = (
            f"Request {identifier}.\n"
            + " x" * copies
            + "\nContinue with a detailed account."
        )
        actual = len(tokenizer.prepare({"messages": messages, "effort": "off"}))
        if actual == length:
            return messages
        copies += length - actual
    raise ValueError(f"cannot construct the exact prompt length: {length}")


def request_body(messages, output, seed):
    return {
        "model": SERVED_MODEL,
        "messages": messages,
        "max_tokens": output,
        "min_tokens": output,
        "ignore_eos": True,
        "reasoning_effort": "off",
        "seed": seed,
        "temperature": 0,
    }


class Service:
    def __init__(self, arguments, mode, strategy):
        self.arguments = arguments
        self.mode = mode
        self.strategy = strategy
        self.request_timeout_seconds = getattr(
            arguments, "request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        cache_configuration = effective_cache_configuration(arguments, strategy)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.trace = arguments.artifacts / f"{mode}-{strategy}-execution.jsonl"
        self.log = (arguments.artifacts / f"{mode}-{strategy}-service.log").open("x")
        self._ipc_directory = tempfile.TemporaryDirectory(prefix="omv-deepseek-")
        self.ipc = Path(self._ipc_directory.name) / "worker.ipc"
        environment = dict(
            os.environ,
            OH_MY_VLLM_DEEPSEEK_STRATEGY=strategy,
            OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT=str(arguments.cache_context),
            OH_MY_VLLM_EXECUTION_TRACE=str(self.trace),
        )
        command = [
            str(arguments.binary),
            "--model",
            str(arguments.model),
            "--max-num-batched-tokens",
            str(effective_service_configuration(arguments)["max_num_batched_tokens"]),
            "--max-num-seqs",
            str(effective_service_configuration(arguments)["max_num_seqs"]),
            "--max-model-len",
            "1048576",
            "--num-speculative-tokens",
            str(5 if mode == "dspark5" else 0),
        ]
        if mode == "dspark5":
            command += ["--speculative-mode", "dspark5"]
        command += ["--num-gpu-blocks", str(cache_configuration["blocks"])]
        command += ["--mamba-blocks", str(cache_configuration["mamba_blocks"])]
        command += [
            "--socket",
            str(self.ipc),
            "serve",
            "--listen",
            f"127.0.0.1:{self.port}",
            "--request-timeout-seconds",
            str(self.request_timeout_seconds),
            "--worker-step-timeout-seconds",
            "600",
            "--shutdown-grace-seconds",
            str(SHUTDOWN_GRACE_SECONDS),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                env=environment,
                cwd=ROOT,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            self._ipc_directory.cleanup()
            raise

    def ready(self):
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"service exited: {self.mode}:{self.strategy}")
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            try:
                connection.request("GET", "/diagnostics")
                response = connection.getresponse()
                if response.status == 200:
                    return json.loads(response.read())
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(1)
        raise TimeoutError("DeepSeek service initialization")

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=SHUTDOWN_GRACE_SECONDS + 10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()
        stale_socket = self.ipc.exists()
        self._ipc_directory.cleanup()
        if stale_socket:
            raise RuntimeError("DeepSeek service did not remove its IPC socket")

    def effective_strategy(self):
        with self.trace.open() as source:
            identity = json.loads(source.readline())
        return identity["cache_memory"]["effective_strategy"]

    def run(self, tokenizer, case, prompt_seed, *, identity):
        version = getattr(self.arguments, "reference_version", accounting.VERSION)
        if identity.get("reference") != reference_configuration(version):
            raise BindingError("benchmark reference identity is missing or changed")
        started = time.perf_counter()
        nonce = hashlib.sha256(f"{prompt_seed}:{case_id(case)}".encode()).hexdigest()[
            :32
        ]
        bodies = [
            request_body(
                exact_prompt(tokenizer, case["input_tokens"], f"{nonce}-{index}"),
                case["output_tokens"],
                index,
            )
            for index in range(case["batch"])
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=case["batch"]) as pool:
            responses = list(
                pool.map(
                    lambda body: post(
                        self.port,
                        body,
                        timeout=self.request_timeout_seconds,
                    ),
                    bodies,
                )
            )
        seconds = time.perf_counter() - started
        trace = assemble_trace(
            read_rank_events(self.trace), responses, harness=identity, case=case
        )
        reference, metadata = runtime_reference(
            trace,
            strategy=case["strategy"],
            expected_binding=trace["binding"],
            version=version,
        )
        return {
            **case,
            "status": "completed",
            "measurement_identity": identity,
            "runtime_modes": trace["binding"].get("runtime_modes"),
            "effective_strategy": trace["binding"]["effective_strategy"],
            **metadata,
            "seconds": seconds,
            "request_count": len(responses),
            "output_tokens_per_second": (
                case["batch"] * case["output_tokens"] / seconds
            ),
            "reference_seconds": reference.seconds,
            "reference_acceptance": reference.acceptance,
            "efficiency_percent": efficiency_percent(reference.seconds, seconds),
            "accepted": trace["binding"]["accepted"],
            "proposed": trace["binding"]["proposed"],
            "trace": trace,
            "responses": responses,
        }
