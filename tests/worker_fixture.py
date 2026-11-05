import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

from tests.test_models import sample_descriptor

QWEN_CONFIG = Path(__file__).parent / "fixtures" / "qwen38" / "config.json"
REPLIES = {
    "text": "test answer",
    "tool": '<tool_call>{"name":"lookup","arguments":{"key":"hello"}}</tool_call>',
    "json": '{"key":"hello"}',
}


def config_bytes():
    # The checkpoint configuration has no final newline.
    return QWEN_CONFIG.read_bytes().removesuffix(b"\n")


def install(directory):
    directory = Path(directory)
    (directory / "config.json").write_bytes(config_bytes())
    (directory / "replies.json").write_text(json.dumps(REPLIES))
    launcher = directory / "worker.sh"
    launcher.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m tests.worker_fixture "$@"\n'
    )
    launcher.chmod(0o700)
    return {**os.environ, "OH_MY_VLLM_WORKER_PYTHON": str(launcher)}


class ScriptedWorker:
    def __init__(self, config):
        self.config = config
        self.requests = {}
        self.snapshots = {}
        self.store_bytes = self.load_bytes = 0
        self.blocks = config["num_gpu_blocks"] or 8
        self.mamba_blocks = config["mamba_blocks"] or 8

    def ready(self):
        descriptor = sample_descriptor(
            fa_capacity_blocks=self.blocks,
            gdn_capacity_blocks=self.mamba_blocks,
            num_speculative_tokens=self.config["num_speculative_tokens"],
        )
        descriptor["config_sha256"] = hashlib.sha256(config_bytes()).hexdigest()
        return dict(
            type="ready",
            descriptor=descriptor,
            logical_num_blocks=self.blocks,
            mamba_blocks=self.mamba_blocks,
            host_cache_bytes=self.config["host_cache_bytes"] or 0,
        )

    def prepare(self, message):
        request = message["request"]
        prompt = []
        for item in request["messages"]:
            digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).digest()
            prompt.extend(byte + 1 for byte in digest)
        limit = request["max_tokens"]
        if len(prompt) + limit > self.config["max_model_len"]:
            raise ValueError("prompt plus output exceeds context limit")
        replies = json.loads(
            (Path(self.config["model_path"]) / "replies.json").read_text()
        )
        key = "text"
        if request["tools"] and request["tool_choice"] != "none":
            key = "tool"
        elif request["format"]["type"] in ("json_object", "json_schema"):
            key = "json"
        text = replies[key]
        end = min(
            (text.find(stop) for stop in request["stop"] if stop in text),
            default=len(text),
        )
        output = list(text[:end])
        if request.get("ignore_eos") and end == len(text):
            output.extend([""] * max(0, limit - len(output)))
        reasoning = []
        if request["effort"] != "off":
            reasoning = ["</think>"]
        stopped = end < len(text) and len(reasoning) + end <= limit
        output = (reasoning + output)[:limit]
        finish = "stop" if stopped or len(output) < limit else "length"
        self.requests[message["request_id"]] = dict(
            prompt=prompt,
            output=output or [""],
            emitted=0,
            finish=finish,
            reasoning=len(reasoning),
        )
        return dict(type="prepared", rpc_id=message["rpc_id"], prompt_token_ids=prompt)

    def execute(self, message):
        for request_id in message["finished_request_ids"]:
            self.requests.pop(request_id, None)
        outputs = []
        for row in message["scheduled"]:
            request = self.requests[row["request_id"]]
            final = (
                row["num_computed_tokens"] + len(row["token_ids"])
                >= len(request["prompt"]) + request["emitted"]
            )
            text = request["output"][request["emitted"]] if final else ""
            request["emitted"] += int(final)
            token = 151668 if text == "</think>" else ord(text) + 1 if text else 151645
            outputs.append(
                dict(
                    request_id=row["request_id"],
                    token_ids=[token] if final else [],
                    num_accepted_draft_tokens=0,
                    new_draft_token_ids=[],
                    text=text,
                    finish_reason=request["finish"]
                    if request["emitted"] == len(request["output"])
                    else None,
                    reasoning_tokens=min(request["emitted"], request["reasoning"]),
                )
            )
        return dict(type="execute_result", rpc_id=message["rpc_id"], outputs=outputs)

    def cache(self, message):
        completed = []
        for operation in message["operations"]:
            snapshot_id, request_id = operation["snapshot_id"], operation["request_id"]
            if operation["kind"] == "store":
                size = max(
                    1,
                    64 * (len(operation["fa_blocks"]) + len(operation["mamba_blocks"])),
                )
                self.snapshots[snapshot_id] = (self.requests.get(request_id), size)
                self.store_bytes += size
            else:
                request, size = self.snapshots[snapshot_id]
                if operation["kind"] == "load":
                    if request_id is not None:
                        self.requests[request_id] = request
                    self.load_bytes += size
                else:
                    del self.snapshots[snapshot_id]
            completed.append(
                dict(
                    transfer_id=operation["transfer_id"], status="complete", bytes=size
                )
            )
        return dict(
            type="cache_result",
            rpc_id=message["rpc_id"],
            completed=completed,
            host_cache_bytes=self.config["host_cache_bytes"] or 0,
        )

    def handle(self, message):
        kind = message["type"]
        if kind == "prepare":
            return self.prepare(message)
        if kind == "register":
            self.requests[message["request_id"]] = dict(
                prompt=message["prompt_token_ids"],
                output=["x"] * message.get("max_tokens", 64),
                emitted=0,
                finish="length",
                reasoning=0,
            )
        elif kind == "execute":
            return self.execute(message)
        elif kind == "cache":
            return self.cache(message)
        elif kind == "abort":
            self.requests.pop(message["request_id"], None)
        elif kind == "stats":
            return dict(
                type="stats_result",
                rpc_id=message["rpc_id"],
                stats=dict(
                    requests=len(self.requests),
                    process_id=os.getpid(),
                    host_cache_store_bytes=self.store_bytes,
                    host_cache_load_bytes=self.load_bytes,
                ),
            )
        else:
            raise ValueError("unexpected test worker message")


def main():
    import msgpack
    import zmq

    address = sys.argv[sys.argv.index("--socket") + 1]
    with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(address)
        worker = None
        while True:
            message = msgpack.unpackb(socket.recv(), raw=False)
            if message["type"] == "shutdown":
                return
            try:
                if message["type"] == "init":
                    worker = ScriptedWorker(message)
                    reply = worker.ready()
                else:
                    reply = worker.handle(message)
            except ValueError as error:
                reply = dict(
                    type="error",
                    rpc_id=message.get("rpc_id", 0),
                    kind="validation",
                    message=str(error),
                )
            if reply is not None:
                socket.send(msgpack.packb(reply, use_bin_type=True))


if __name__ == "__main__":
    main()
