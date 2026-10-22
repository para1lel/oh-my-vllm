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

from oh_my_vllm.operators.runtime.native import cache_directory, file_sha256
from oh_my_vllm.worker.execution import execution_policy
from oh_my_vllm.worker.tokenizer import Tokenization, tokenizer_identity

from .accounting import RATES, VERSION
from .device import device_identity, selected_uuid
from .roofline import lower_bound
from .workload import PROFILE, SAMPLING_FIELDS, TOOLS, request_body, validate_tools

ROOT = Path(__file__).resolve().parents[2]
NATIVE_MODULES = (
    "device",
    "ffn",
    "fa",
    "gdn",
    "mtp",
    "sample",
    "vision",
)
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


def efficiency(seconds, reference_seconds):
    if any(
        not math.isfinite(value) or value <= 0 for value in (seconds, reference_seconds)
    ):
        raise ValueError("measurement and reference times must be finite and positive")
    return reference_seconds / seconds


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
                "development/native_build.py",
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
        vendor = metadata.get("vision_vendor", {})
        files = {
            **vendor.get("headers", {}),
            **{image["path"]: image["sha256"] for image in vendor.get("images", [])},
        }
        for path, expected in files.items():
            if file_sha256(path) != expected:
                raise RuntimeError(
                    "native vendor file identity differs from its manifest"
                )
        result[name] = metadata
    return result


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_identity(arguments):
    return {
        "source": source_identity(),
        "execution": execution_policy(),
        "prompt_seed": arguments.prompt_seed,
        "binary": file_sha256(arguments.binary),
        "native": native_identity(),
        "checkpoint": checkpoint_identity(arguments.model),
        "tokenizer": tokenizer_identity(arguments.model),
        "device": device_identity(arguments.gpu_uuid),
        "accounting": dict(version=VERSION, rates=RATES),
        "workload": PROFILE,
    }


def verify_artifact_identity(arguments, expected):
    for name, value in artifact_identity(arguments).items():
        if value != expected[name]:
            raise RuntimeError(f"{name} artifacts changed during the benchmark")


def verify_trace_identity(events, checkpoint, config, tokenizer=None, execution=None):
    identities = [event for event in events if event["operation"] == "identity"]
    if len(identities) != 1:
        raise ValueError("trace requires exactly one identity record")
    identity = identities[0]
    if execution is not None and identity.get("execution") != execution:
        raise ValueError("trace execution differs from the selected phase paths")
    actual_config = identity.get("config", {})
    for name, value in config.items():
        actual_value = actual_config.get(name)
        if name == "num_gpu_blocks" and value is None:
            valid = type(actual_value) is int and 2 <= actual_value <= 0xFFFFFFFF
        else:
            valid = actual_value == value
        if not valid:
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
        # Rust sorts object keys before tool JSON enters the chat template.
        "tools": json.loads(json.dumps(TOOLS, sort_keys=True)),
        "tool_choice": "auto",
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
            "--mamba-blocks",
            str(arguments.mamba_blocks),
            "--max-num-batched-tokens",
            "32768",
            "--max-num-seqs",
            "4",
        ]
        if arguments.blocks is not None:
            command += ["--num-gpu-blocks", str(arguments.blocks)]
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
                    self.sampling_defaults = json.loads(response.read())["limits"][
                        "override_generation_config"
                    ]
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
                (json.loads(line) for line in source),
                checkpoint,
                config,
                tokenizer,
                execution_policy(),
            )

    def check_tools(self, tokenizer):
        body = request_body(
            [{"role": "user", "content": "Find a reference about field research."}],
            128,
            0,
        )
        body.update(tool_choice="required", min_tokens=0)
        response = post(self.port, body)
        validate_tools(response, tokenizer, required=True)
        (self.arguments.artifacts / f"{self.mode}-tool-check.json").write_text(
            json.dumps(dict(request=body, response=response), indent=2)
        )

    def run(self, tokenizer, batch, length, output):
        nonce = hashlib.sha256(
            f"{self.arguments.prompt_seed}:{self.mode}:{batch}:{length}:{output}".encode()
        ).hexdigest()[:32]
        bodies = [
            request_body(prompt(tokenizer, length, f"{nonce}-{index}"), output, index)
            for index in range(batch)
        ]
        sampling = {
            key: value
            for key, value in (tokenizer.generation | self.sampling_defaults).items()
            if key in SAMPLING_FIELDS
        }
        if sampling.get("temperature", 1) <= 0:
            raise ValueError("benchmark requires stochastic serving defaults")
        previous = self.trace.read_bytes()
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
            responses = list(pool.map(lambda body: post(self.port, body), bodies))
        finished = time.perf_counter()
        seconds = finished - started
        for value in responses:
            usage = value["usage"]
            if (
                usage["prompt_tokens"] != length
                or usage["completion_tokens"] != output
                or usage["prompt_tokens_details"]["cached_tokens"] != 0
            ):
                raise ValueError(f"token count or cold-prefix check failed: {usage}")
        manifest = {
            "artifact_id": self.arguments.artifact_id,
            "mode": self.mode,
            "sampling": sampling,
            "workload": PROFILE,
            "requests": [
                dict(
                    ordinal=index,
                    id=int(value["id"].rsplit("_", 1)[1]),
                    response_id=value["id"],
                    input_tokens=length,
                    output_tokens=output,
                    seed=index,
                    body_sha256=json_digest(bodies[index]),
                    tools=validate_tools(value, tokenizer),
                )
                for index, value in enumerate(responses)
            ],
            "timer": dict(clock="perf_counter", started=started, finished=finished),
        }
        current = self.trace.read_bytes()
        if not current.startswith(previous):
            raise ValueError("execution trace changed before the case boundary")
        identities = [
            json.loads(line)
            for line in previous.splitlines()
            if json.loads(line)["operation"] == "identity"
        ]
        events = identities + [
            json.loads(line) for line in current[len(previous) :].splitlines()
        ]
        bound = lower_bound(events, manifest)
        return {
            "mode": self.mode,
            "batch": batch,
            "input_tokens": length,
            "output_tokens": output,
            "seconds": seconds,
            "lower_bound": bound,
            "manifest": manifest,
            "manifest_id": json_digest(manifest),
            "trace_id": json_digest(events),
            "efficiency": efficiency(seconds, bound["seconds"]),
            "responses": responses,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prompt-seed", default=uuid.uuid4().hex)
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
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int, default=128)
    arguments = parser.parse_args()
    arguments.gpu_uuid = selected_uuid()
    if "B200" not in device_identity(arguments.gpu_uuid)["name"]:
        parser.error("the nominal resource model requires one B200")
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
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
    }
    arguments.artifact_id = json_digest(identity)
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
            service.check_tools(tokenizer)
            for batch in sorted({item[1] for item in cases}):
                service.run(tokenizer, batch, 1024, 16)
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
                                "efficiency",
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
    (arguments.artifacts / "summary.json").write_text(
        json.dumps({"configurations": len(results)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
