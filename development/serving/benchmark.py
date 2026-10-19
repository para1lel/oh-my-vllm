import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

from oh_my_vllm.kernels.native import cache_directory, file_sha256
from oh_my_vllm.worker.tokenizer import Tokenization, tokenizer_identity

from development.peaks import (
    device_identity,
    load_profile,
    profile_peaks,
    selected_uuid,
)

from .roofline import lower_bound

MIN_EFFICIENCY = 0.50

ROOT = Path(__file__).resolve().parents[2]
NATIVE_MODULES = ("device", "ffn", "fa", "gdn", "mtp", "sample", "vision")
REQUEST_SHAPES = [("ordinary", batch, 32768) for batch in (1, 2, 4)] + [
    ("mtp4", batch, length) for batch in (1, 2, 4) for length in (32768, 65536, 131072)
]
OUTPUT_LENGTHS = (64, 8192)
CONFIGURATIONS = [
    (*shape, output) for shape in REQUEST_SHAPES for output in OUTPUT_LENGTHS
]


def select_configurations(selected, outputs):
    if (
        not outputs
        or len(set(outputs)) != len(outputs)
        or any(value < 1 for value in outputs)
    ):
        raise ValueError("use distinct positive output token counts")
    selected = list(selected or [])
    if len(set(selected)) != len(selected):
        raise ValueError("use distinct case selections")
    configurations = [
        (*shape, output)
        for shape in REQUEST_SHAPES
        for output in outputs
        if not selected
        or ":".join(map(str, shape)) in selected
        or ":".join(map(str, (*shape, output))) in selected
    ]
    matched = {
        ":".join(map(str, case))
        for configuration in configurations
        for case in (configuration[:3], configuration)
    }
    if set(selected) - matched:
        raise ValueError("each case must match the selected output token counts")
    return configurations


def gate_result(configuration, seconds, roofline_seconds):
    if any(
        not math.isfinite(value) or value <= 0 for value in (seconds, roofline_seconds)
    ):
        raise ValueError("measurement and roofline times must be finite and positive")
    efficiency = roofline_seconds / seconds
    applicable = configuration in CONFIGURATIONS
    return {
        "limit_seconds": roofline_seconds / MIN_EFFICIENCY,
        "efficiency": efficiency,
        "gate_applicable": applicable,
        "passed": applicable and efficiency >= MIN_EFFICIENCY,
    }


def comparison_passed(results):
    configurations = [
        (
            result["mode"],
            result["batch"],
            result["input_tokens"],
            result["output_tokens"],
        )
        for result in results
    ]
    return (
        len(configurations) == len(CONFIGURATIONS)
        and set(configurations) == set(CONFIGURATIONS)
        and all(result["passed"] for result in results)
    )


def trace_bound(events, request_ids, mode, length, output, peaks):
    identities = [event for event in events if event["operation"] == "identity"]
    if len(identities) != 1 or identities[0]["config"]["num_speculative_tokens"] != (
        4 if mode == "mtp4" else 0
    ):
        raise ValueError("trace identity differs from the requested service mode")
    for event in events:
        if event["operation"] == "target" and any(
            row["prompt"] != length
            for row in event["requests"]
            if row["id"] in request_ids
        ):
            raise ValueError("trace prompt length differs from HTTP usage")
    return lower_bound(events, request_ids, output, peaks=peaks)


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
                "development/kernels/workload.py",
                "development/peaks.py",
                "development/peak_measurement.py",
                "development/peak_kernels.cu",
            }
        ):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_identity(model):
    model = Path(model)
    index = (model / "model.safetensors.index.json").read_bytes()
    names = set(json.loads(index)["weight_map"].values())
    if not names or any(
        Path(name).name != name or not name.endswith(".safetensors") for name in names
    ):
        raise ValueError("checkpoint index must name local safetensors shards")
    return {
        "config": file_sha256(model / "config.json"),
        "index": hashlib.sha256(index).hexdigest(),
        **{name: file_sha256(model / name) for name in sorted(names)},
    }


def native_identity():
    directory = cache_directory()
    result = {}
    for name in NATIVE_MODULES:
        metadata = json.loads((directory / f"{name}.json").read_text())
        if file_sha256(metadata["library"]) != metadata["binary_id"]:
            raise RuntimeError(
                f"native binary identity differs from its manifest: {name}"
            )
        for key in ("vision_vendor", "ffn_mx_vendor", "gdn_mx_vendor", "fa_mx_vendor"):
            vendor = metadata.get(key, {})
            files = {
                **vendor.get("headers", {}),
                **{
                    image["path"]: image["sha256"] for image in vendor.get("images", [])
                },
            }
            for path, expected in files.items():
                if file_sha256(path) != expected:
                    raise RuntimeError(
                        "native vendor file identity differs from its manifest"
                    )
        result[name] = metadata
    return result


def artifact_identity(arguments):
    return {
        "source": source_identity(),
        "binary": file_sha256(arguments.binary),
        "native": native_identity(),
        "checkpoint": checkpoint_identity(arguments.model),
        "tokenizer": tokenizer_identity(arguments.model),
        "calibration": load_profile(arguments.peaks, arguments.gpu_uuid),
    }


def verify_artifact_identity(arguments, expected):
    if device_identity(arguments.gpu_uuid) != expected["calibration"]["device"]:
        raise RuntimeError("GPU settings changed during the benchmark")
    for name, value in artifact_identity(arguments).items():
        if value != expected[name]:
            raise RuntimeError(f"{name} artifacts changed during the benchmark")


def verify_trace_identity(events, checkpoint, config, tokenizer=None):
    identities = [event for event in events if event["operation"] == "identity"]
    if len(identities) != 1:
        raise ValueError("trace requires exactly one identity record")
    identity = identities[0]
    actual_config = identity.get("config", {})
    if any(actual_config.get(name) != value for name, value in config.items()):
        raise ValueError("trace configuration differs from the requested service")
    actual = identity.get("checkpoint")
    if (
        not isinstance(actual, dict)
        or not {"config", "index"} < actual.keys()
        or any(
            name not in checkpoint or checkpoint[name] != value
            for name, value in actual.items()
        )
    ):
        raise ValueError("trace checkpoint differs from the frozen checkpoint files")
    if tokenizer is not None and identity.get("tokenizer") != tokenizer:
        raise ValueError("trace loaded tokenizer assets differ from the frozen files")


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

    def verify_identity(self, checkpoint, tokenizer):
        arguments = self.arguments
        config = {
            "model_path": str(arguments.model),
            "num_gpu_blocks": arguments.blocks,
            "mamba_blocks": arguments.mamba_blocks,
            "block_size": 784,
            "tensor_parallel_size": 1,
            "max_model_len": 262144,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 4,
            "num_speculative_tokens": 4 if self.mode == "mtp4" else 0,
        }
        with self.trace.open() as source:
            verify_trace_identity(
                (json.loads(line) for line in source), checkpoint, config, tokenizer
            )

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
        if len(ids) != batch:
            raise ValueError("HTTP responses must have distinct request IDs")
        events = [json.loads(line) for line in self.trace.read_text().splitlines()]
        bound = trace_bound(
            events, ids, self.mode, length, output, self.arguments.peak_rates
        )
        return {
            "mode": self.mode,
            "batch": batch,
            "input_tokens": length,
            "output_tokens": output,
            "seconds": seconds,
            "lower_bound": bound,
            **gate_result(
                (self.mode, batch, length, output), seconds, bound["seconds"]
            ),
            "responses": responses,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peaks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--case",
        choices=[
            ":".join(map(str, case)) for case in (*REQUEST_SHAPES, *CONFIGURATIONS)
        ],
        action="append",
    )
    parser.add_argument("--output-tokens", type=int, nargs="+", default=OUTPUT_LENGTHS)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--mamba-blocks", type=int, default=1024)
    arguments = parser.parse_args()
    arguments.gpu_uuid = selected_uuid()
    calibration = load_profile(arguments.peaks, arguments.gpu_uuid)
    if calibration["device"] != device_identity(arguments.gpu_uuid):
        parser.error("GPU settings differ from calibration; measure the GPU again")
    arguments.peak_rates = profile_peaks(calibration, arguments.gpu_uuid)
    arguments.artifacts = arguments.artifacts.resolve()
    arguments.binary = arguments.binary.resolve()
    arguments.model = arguments.model.resolve()
    if arguments.artifacts.is_relative_to(ROOT):
        parser.error("use external artifacts")
    try:
        configurations = select_configurations(arguments.case, arguments.output_tokens)
    except ValueError as error:
        parser.error(str(error))
    if not configurations:
        parser.error("no configurations match the case and output selections")
    arguments.artifacts.mkdir(parents=True, exist_ok=False)
    identity = {
        **artifact_identity(arguments),
        "device": subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                arguments.gpu_uuid,
                "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "config": {
            key: str(value)
            if isinstance(value, Path)
            else vars(value)
            if key == "peak_rates"
            else value
            for key, value in vars(arguments).items()
        },
    }
    tokenizer = Tokenization(arguments.model)
    tokenizer.verify_identity(identity["tokenizer"])
    (arguments.artifacts / "identity.json").write_text(json.dumps(identity, indent=2))
    results = []
    for mode in ("ordinary", "mtp4"):
        cases = [item for item in configurations if item[0] == mode]
        if not cases:
            continue
        verify_artifact_identity(arguments, identity)
        service = Service(arguments, mode)
        try:
            service.ready()
            service.verify_identity(identity["checkpoint"], identity["tokenizer"])
            for batch in sorted({item[1] for item in cases}):
                service.run(tokenizer, batch, 64, 16)
            for _, batch, length, output in cases:
                print(
                    json.dumps(
                        {
                            "event": "start",
                            "mode": mode,
                            "batch": batch,
                            "input": length,
                            "output": output,
                        }
                    ),
                    flush=True,
                )
                result = service.run(tokenizer, batch, length, output)
                results.append(result)
                (
                    arguments.artifacts / f"{mode}-{batch}-{length}-{output}.json"
                ).write_text(json.dumps(result, indent=2))
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "mode",
                                "batch",
                                "input_tokens",
                                "output_tokens",
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
        service.verify_identity(identity["checkpoint"], identity["tokenizer"])
        verify_artifact_identity(arguments, identity)
    verify_artifact_identity(arguments, identity)
    passed = comparison_passed(results)
    (arguments.artifacts / "summary.json").write_text(
        json.dumps({"passed": passed, "configurations": len(results)}, indent=2)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
