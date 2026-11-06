import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

QWEN_CONFIG = Path(__file__).parent / "fixtures" / "qwen38" / "config.json"
REPLIES = {
    "text": "test answer",
    "tool": '<tool_call>{"name":"lookup","arguments":{"key":"hello"}}</tool_call>',
    "json": '{"key":"hello"}',
}


def config_bytes():
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


def cache_layout(config, pages, checkpoints):
    return dict(
        kind="qwen_radix_v1",
        layout_id="qwen_radix_gdn8192_fp8kv128_v1",
        gdn_period_tokens=8192,
        kv_page_tokens=128,
        usable_checkpoints=checkpoints,
        usable_pages=pages,
        null_checkpoints=1,
        null_pages=1,
        private_checkpoints=6 * config["max_num_seqs"],
        scratch_checkpoints=(config["max_num_batched_tokens"] + 8191) // 8192
        + config["max_num_seqs"],
        pool_bytes={"fixture": 1},
        format_fingerprint="qwen_fp8_e4m3fn_rne_fp32scale_v1",
    )


def descriptor(config, pages, checkpoints):
    groups = []
    for name, kind, period, count, layout, layers in (
        (
            "fa",
            "full_attention",
            128,
            pages,
            "qwen38_radix_fa_fp8kv128",
            list(range(3, 64, 4)) + ([64] if config["num_speculative_tokens"] else []),
        ),
        (
            "gdn",
            "recurrent",
            8192,
            checkpoints,
            "qwen38_radix_gdn8192_f32_conv_bf16_hidden_bf16",
            [layer for layer in range(64) if layer % 4 != 3],
        ),
    ):
        groups.append(
            dict(
                name=name,
                kind=kind,
                block_tokens=period,
                capacity_blocks=count + 1,
                layers=layers,
                storage=dict(type="layout", id=layout, version=1),
                source=dict(type="owned"),
            )
        )
    return dict(
        schema_version=1,
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5",
        config_sha256=hashlib.sha256(config_bytes()).hexdigest(),
        checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        vocabulary_size=248320,
        max_context_tokens=262144,
        capabilities=[
            "text",
            "images",
            "video",
            "tools",
            "reasoning",
            "structured_outputs",
            "prefix_cache",
        ],
        weight_formats=["bf16", "f32", "qwen_fp8_e4m3_block128x128_fp32_scales"],
        cache_schema=dict(schema_version=1, groups=groups),
        parallel_plan=dict(
            strategy="tp1", world_size=1, ranks=[0], request_placement="replicated"
        ),
        speculation=[
            dict(
                method="mtp",
                predictor_stages=1,
                max_draft_tokens=4,
                max_verification_rows=5,
            )
        ],
    )


class ScriptedWorker:
    def __init__(self, config):
        self.config = config
        self.requests = {}
        self.mamba_blocks = config["mamba_blocks"] or 8
        self.blocks = config["num_gpu_blocks"] or 64 * self.mamba_blocks

    def ready(self):
        return dict(
            type="ready",
            protocol_version=2,
            worker_epoch=self.config["worker_epoch"],
            descriptor=descriptor(self.config, self.blocks, self.mamba_blocks),
            logical_num_blocks=self.blocks,
            mamba_blocks=self.mamba_blocks,
            cache_layout=cache_layout(self.config, self.blocks, self.mamba_blocks),
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
        reasoning = ["</think>"] if request["effort"] != "off" else []
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
        self.trace_execute(message)
        for request_id in message["finished_request_ids"]:
            self.requests.pop(request_id, None)
        outputs = [self.execute_row(row) for row in message["scheduled"]]
        return dict(
            type="execute_result",
            rpc_id=message["rpc_id"],
            step_id=message["step_id"],
            worker_epoch=message["worker_epoch"],
            outputs=outputs,
        )

    def execute_row(self, row):
        request = self.requests[row["request_id"]]
        sampled = row["sampling"]["kind"] != "none"
        text = request["output"][request["emitted"]] if sampled else ""
        request["emitted"] += int(sampled)
        token = 151668 if text == "</think>" else ord(text) + 1 if text else 151645
        state = row["model_state"]
        computed = row["forward_end"] - row["sampling"].get("drafts", 0)
        pages = []
        for page in state["page_table"]:
            start = page["page_index"] * 128
            valid_end = max(start, min(computed, start + 128))
            if row["purpose"] != "finalize":
                valid_end = max(valid_end, start + page["valid_tokens"])
            pages.append(
                dict(
                    page_index=page["page_index"],
                    slot=page["slot"],
                    version=page["version"],
                    valid_end=valid_end,
                )
            )
        return dict(
            request_id=row["request_id"],
            lease_id=row["lease_id"],
            purpose=row["purpose"],
            action=row["action"],
            error=None,
            token_ids=[token] if sampled else [],
            num_accepted_draft_tokens=0,
            new_draft_token_ids=[],
            text=text,
            finish_reason=request["finish"]
            if sampled and request["emitted"] == len(request["output"])
            else None,
            reasoning_tokens=min(request["emitted"], request["reasoning"])
            if row["purpose"] == "advance"
            else 0,
            computed_end=computed,
            accepted_end=row["accepted_end"] + int(sampled),
            validated_exports=[
                {key: export[key] for key in ("end", "slot", "version")}
                for export in state["exports"]
            ],
            committed_page_valid_ends=pages,
            mtp_accepted_endpoint=self.mtp_endpoint(state, computed),
        )

    def mtp_endpoint(self, state, computed):
        if not self.config["num_speculative_tokens"]:
            return None
        candidates = [
            export for export in state["exports"] if export["end"] == computed
        ]
        if candidates:
            source = candidates[-1]
        elif state["endpoint_source"]["kind"] == "saved_hidden":
            source = state["endpoint_source"]
        else:
            return None
        return dict(
            end=max(0, computed - 1), slot=source["slot"], version=source["version"]
        )

    def trace_execute(self, message):
        root = os.environ.get("OH_MY_VLLM_FIXTURE_TRACE")
        if root is None:
            return
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        record = {key: message[key] for key in ("rpc_id", "step_id", "worker_epoch")}
        record["rows"] = []
        for row in message["scheduled"]:
            item = {
                key: row[key]
                for key in (
                    "request_id",
                    "lease_id",
                    "purpose",
                    "action",
                    "sampling",
                    "forward_start",
                    "forward_end",
                    "accepted_end",
                )
            }
            item["token_rows"] = len(row["token_ids"])
            item["prefix"] = {
                key: row["model_state"][key]
                for key in ("checkpoint_hit_end", "kv_contiguous_end", "resume")
            }
            record["rows"].append(item)
        with (directory / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

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
        elif kind == "abort":
            self.requests.pop(message["request_id"], None)
        elif kind == "stats":
            return dict(
                type="stats_result",
                rpc_id=message["rpc_id"],
                stats=dict(requests=len(self.requests), process_id=os.getpid()),
            )
        else:
            raise ValueError("unexpected test worker message")


def main():
    import msgpack
    import zmq
    from oh_my_vllm.worker.wire import validate_execute, validate_init, validate_result

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
                    validate_init(message)
                    worker = ScriptedWorker(message)
                    reply = worker.ready()
                elif message["type"] == "execute":
                    expected = validate_execute(message, worker.config["worker_epoch"])
                    reply = worker.handle(message)
                    validate_result(message, reply, expected)
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
