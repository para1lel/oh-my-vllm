import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.tokenizer import Tokenization
from oh_my_vllm.operators.runtime.native import cache_directory, file_sha256

from .benchmark import ROOT, source_identity
from .deepseek_matrix import HTTP_CASES, case_id
from .device import device_identity, selected_uuids

DEEPSEEK_NATIVE_MODULES = (
    "device",
    "activation",
    "hyperconnection",
    "projection",
    "routing",
    "sparse_attention",
    "transform",
)
SHUTDOWN_GRACE_SECONDS = 35
SERVED_MODEL = "deepseek-v4.1-flash"


def _checkpoint_shard_manifest(root):
    index_path = Path(root) / "model.safetensors.index.json"
    index = index_path.read_bytes()
    names = set(json.loads(index)["weight_map"].values())
    if not names or any(
        re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) is None
        for name in names
    ):
        raise ValueError("checkpoint index must name local safetensors shards")
    return hashlib.sha256(index).hexdigest(), sorted(names)


def _checkpoint_shard_assignments(root, names, world_size):
    loads = [0] * world_size
    assignments = {}
    sizes = {name: (Path(root) / name).stat().st_size for name in names}
    for name in sorted(names, key=lambda value: (-sizes[value], value)):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[name] = rank
        loads[rank] += sizes[name]
    return assignments


def checkpoint_shard_identity(root, *, rank, world_size):
    if type(rank) is not int or type(world_size) is not int:
        raise ValueError("checkpoint identity rank and world size must be integers")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("checkpoint identity rank is outside its world")
    root = Path(root)
    _, names = _checkpoint_shard_manifest(root)
    assignments = _checkpoint_shard_assignments(root, names, world_size)
    return {
        name: file_sha256(root / name) for name in names if assignments[name] == rank
    }


def checkpoint_identity(model):
    model = Path(model)
    index_digest, names = _checkpoint_shard_manifest(model)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(
            pool.map(
                lambda rank: checkpoint_shard_identity(
                    model,
                    rank=rank,
                    world_size=4,
                ),
                range(4),
            )
        )
    result = {
        "config": file_sha256(model / "config.json"),
        "index": index_digest,
    }
    for part in parts:
        if set(result) & set(part):
            raise ValueError("checkpoint identity partitions overlap")
        result.update(part)
    if set(result) != {"config", "index", *names}:
        raise ValueError("checkpoint identity partitions are incomplete")
    return result


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
    }


def native_identity():
    directory = cache_directory()
    result = {}
    for name in DEEPSEEK_NATIVE_MODULES:
        metadata = json.loads((directory / f"{name}.json").read_text())
        if file_sha256(metadata["library"]) != metadata["binary_id"]:
            raise RuntimeError(
                f"native binary identity differs from its manifest: {name}"
            )
        result[name] = metadata
    return result


def topology_identity(uuids):
    return [device_identity(uuid) for uuid in uuids]


def effective_cache_configuration(arguments):
    blocks = arguments.blocks
    if blocks is None:
        value = os.environ.get("OH_MY_VLLM_NUM_GPU_BLOCKS")
        blocks = 2048 if value is None else int(value)
    return {
        "blocks": blocks,
        "mamba_blocks": arguments.mamba_blocks or 128,
    }


def artifact_identity(arguments, tokenizer, uuids, *, workload):
    cache_configuration = effective_cache_configuration(arguments)
    return {
        "source": source_identity(),
        "binary": file_sha256(arguments.binary),
        "native": native_identity(),
        "checkpoint": checkpoint_identity(arguments.model),
        "tokenizer": tokenizer.identity,
        "topology": topology_identity(uuids),
        "workload": workload,
        "configuration": {
            "cache_context": arguments.cache_context,
            **cache_configuration,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 64,
            "native_sparse": os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "0"),
        },
    }


def verify_artifact_identity(arguments, tokenizer, uuids, expected, *, workload):
    actual = artifact_identity(
        arguments,
        tokenizer,
        uuids,
        workload=workload,
    )
    if actual != expected:
        raise RuntimeError("DeepSeek benchmark inputs changed during measurement")


def service_config(arguments, mode, strategy):
    drafts = 5 if mode == "dspark5" else 0
    return {
        "model_path": str(arguments.model),
        "strategy": strategy,
        "mode": mode,
        "drafts": drafts,
        "cache_context": arguments.cache_context,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 32768,
    }


class Service:
    def __init__(self, arguments, mode, strategy):
        self.arguments = arguments
        self.mode = mode
        self.strategy = strategy
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.ipc = Path(f"/tmp/oh-my-vllm-deepseek-{os.getpid()}.ipc")
        self.trace = arguments.artifacts / f"{mode}-{strategy}-execution.jsonl"
        self.log = (arguments.artifacts / f"{mode}-{strategy}-service.log").open("x")
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
            "32768",
            "--max-num-seqs",
            "64",
            "--max-model-len",
            "1048576",
            "--num-speculative-tokens",
            str(5 if mode == "dspark5" else 0),
        ]
        if mode == "dspark5":
            command += ["--speculative-mode", "dspark5"]
        if arguments.blocks is not None:
            command += ["--num-gpu-blocks", str(arguments.blocks)]
        if arguments.mamba_blocks is not None:
            command += ["--mamba-blocks", str(arguments.mamba_blocks)]
        command += [
            "--socket",
            str(self.ipc),
            "serve",
            "--listen",
            f"127.0.0.1:{self.port}",
            "--request-timeout-seconds",
            "7200",
            "--worker-step-timeout-seconds",
            "600",
            "--shutdown-grace-seconds",
            str(SHUTDOWN_GRACE_SECONDS),
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
        if self.ipc.exists():
            raise RuntimeError("DeepSeek service did not remove its IPC socket")

    def run(self, tokenizer, case, prompt_seed):
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
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=case["batch"]) as pool:
            responses = list(pool.map(lambda body: post(self.port, body), bodies))
        seconds = time.perf_counter() - started
        for response in responses:
            usage = response["usage"]
            if (
                usage["prompt_tokens"] != case["input_tokens"]
                or usage["completion_tokens"] != case["output_tokens"]
                or usage["prompt_tokens_details"]["cached_tokens"] != 0
            ):
                raise ValueError(
                    f"DeepSeek token count or prefix check failed: {usage}"
                )
        return {
            **case,
            "seconds": seconds,
            "request_count": len(responses),
            "output_tokens_per_second": (
                case["batch"] * case["output_tokens"] / seconds
            ),
            "responses": responses,
        }


def selected_cases(values):
    if not values:
        return list(HTTP_CASES)
    selected = set(values)
    cases = [case for case in HTTP_CASES if case_id(case) in selected]
    if len(cases) != len(selected):
        raise ValueError("each selected case must match the DeepSeek matrix")
    return cases


def write_csv(path, results):
    fields = (
        "mode",
        "strategy",
        "batch",
        "input_tokens",
        "output_tokens",
        "seconds",
        "request_count",
        "output_tokens_per_second",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: result[field] for field in fields} for result in results
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--prompt-seed", default="deepseek-v41-nightly")
    parser.add_argument("--cache-context", type=int, default=1048576)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    arguments = parser.parse_args()
    arguments.model = arguments.model.resolve()
    arguments.artifacts = arguments.artifacts.resolve()
    arguments.binary = arguments.binary.resolve()
    arguments.csv = arguments.csv.resolve()
    if arguments.artifacts.is_relative_to(ROOT):
        parser.error("put raw artifacts external to the repository")
    if arguments.cache_context < 1:
        parser.error("cache context must be positive")
    try:
        cases = selected_cases(arguments.case)
        uuids = selected_uuids()
    except ValueError as error:
        parser.error(str(error))
    if len(uuids) != 4:
        parser.error("DeepSeek benchmark requires four selected GPUs")
    arguments.artifacts.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenization(arguments.model)
    workload = {
        "kind": "http",
        "case_ids": [case_id(case) for case in cases],
        "prompt_seed": arguments.prompt_seed,
    }
    identity = artifact_identity(
        arguments,
        tokenizer,
        uuids,
        workload=workload,
    )
    (arguments.artifacts / "identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    results = []
    for strategy, mode in sorted({(case["strategy"], case["mode"]) for case in cases}):
        selected = [
            case
            for case in cases
            if case["strategy"] == strategy and case["mode"] == mode
        ]
        service = Service(arguments, mode, strategy)
        try:
            ready = service.ready()
            (arguments.artifacts / f"{mode}-{strategy}-ready.json").write_text(
                json.dumps(ready, indent=2) + "\n"
            )
            for case in selected:
                print(json.dumps({"event": "start", **case}), flush=True)
                result = service.run(tokenizer, case, arguments.prompt_seed)
                results.append(result)
                name = case_id(case).replace(":", "-")
                (arguments.artifacts / f"{name}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "mode",
                                "strategy",
                                "batch",
                                "input_tokens",
                                "output_tokens",
                                "seconds",
                            )
                        }
                    ),
                    flush=True,
                )
        finally:
            service.close()
    verify_artifact_identity(
        arguments,
        tokenizer,
        uuids,
        identity,
        workload=workload,
    )
    write_csv(arguments.csv, results)
    (arguments.artifacts / "summary.json").write_text(
        json.dumps({"cases": len(results)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
