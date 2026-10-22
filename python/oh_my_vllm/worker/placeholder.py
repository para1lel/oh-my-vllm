import hashlib
import json
import os
import random
import string
from dataclasses import dataclass

from ..models.stub import StubOffload, build_descriptor


def token_id(character):
    return ord(character) % 248319 + 1


def control_or_text_token(text):
    if text == "</think>":
        return 151668
    return token_id(text) if text else 151645


def prompt_message(message):
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        kind = part["type"]
        if kind == "text":
            parts.append(part)
        else:
            digest = hashlib.sha256(part[kind]["url"].encode()).hexdigest()
            parts.append({"type": kind, "digest": digest})
    return {**message, "content": parts}


def schema_value(schema, root=None, depth=0, budget=None):
    budget = [4096, 1 << 20] if budget is None else budget
    budget[0] -= 1
    if depth >= 32 or budget[0] < 0 or budget[1] < 0:
        raise ValueError("placeholder schema resource limit exceeded")
    root = schema if root is None else root
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError("only local schema references are supported")
        target = root
        for key in reference[2:].split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        return schema_value(target, root, depth + 1, budget)
    if "const" in schema:
        value = schema["const"]
        budget[1] -= len(json.dumps(value).encode())
        if budget[1] < 0:
            raise ValueError("placeholder schema resource limit exceeded")
        return value
    if "enum" in schema:
        return schema_value({"const": schema["enum"][0]}, root, depth + 1, budget)
    if "anyOf" in schema:
        return schema_value(schema["anyOf"][0], root, depth + 1, budget)
    kind = schema.get("type", "object")
    if isinstance(kind, list):
        kind = kind[0]
    match kind:
        case "object":
            budget[1] -= sum(
                len(key.encode()) + 4 for key in schema.get("properties", {})
            )
            return {
                key: schema_value(value, root, depth + 1, budget)
                for key, value in schema.get("properties", {}).items()
            }
        case "array":
            count = schema.get("minItems", 0)
            if not 0 <= count <= 64:
                raise ValueError("placeholder array limit exceeded")
            return [
                schema_value(schema["items"], root, depth + 1, budget)
                for _ in range(count)
            ]
        case "string":
            count = schema.get("minLength", min(4, schema.get("maxLength", 4)))
            if not 0 <= count <= 4096:
                raise ValueError("placeholder string limit exceeded")
            budget[1] -= count
            if budget[1] < 0:
                raise ValueError("placeholder schema resource limit exceeded")
            return "x" * count
        case "integer" | "number":
            return schema.get("minimum", 0)
        case "boolean":
            return False
        case "null":
            return None
        case _:
            raise ValueError("unsupported schema type")


@dataclass
class Request:
    prompt_len: int
    history: list[int]
    output: list[str]
    rng: random.Random
    reason_tokens: int = 0
    finish: str | None = None

    @classmethod
    def offline(cls, prompt, limit):
        rng = random.Random()
        return cls(
            len(prompt), prompt, rng.choices(string.ascii_lowercase, k=limit), rng
        )

    @classmethod
    def prepare(cls, request, max_model_len):
        rng = random.Random(request["sampling"].get("seed"))
        limit = request["max_tokens"]
        prompt = json.dumps(
            [
                [prompt_message(message) for message in request[key]]
                if key == "messages"
                else request[key]
                for key in ("messages", "tools", "effort", "preserve_thinking")
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tokens = list(map(token_id, prompt))
        if len(tokens) + limit > max_model_len:
            raise ValueError("prompt plus output exceeds context limit")
        reasoning = []
        if request["effort"] != "off":
            budget = request.get("thinking_token_budget", -1)
            count = min(limit, 4 if budget is None or budget < 0 else min(budget, 4))
            reasoning = [*rng.choices(string.ascii_lowercase, k=count), "</think>"]
        choice = request["tool_choice"]
        tools = request["tools"] if choice != "none" else []
        if tools and (choice != "auto" or rng.randrange(2)):
            name = (
                choice.get("function", {}).get("name")
                if isinstance(choice, dict)
                else None
            )
            function = next(
                t["function"]
                for t in tools
                if name is None or t["function"]["name"] == name
            )
            arguments = schema_value(function.get("parameters", {}))
            if not isinstance(arguments, dict):
                raise ValueError("tool parameters must be an object")
            parts = [f"<tool_call><function={function['name']}>\n"]
            for key, value in arguments.items():
                text = value if isinstance(value, str) else json.dumps(value)
                parts.append(f"<parameter={key}>\n{text}\n</parameter>\n")
            answer = "".join(parts) + "</function></tool_call>"
        elif request["format"]["type"] == "json_schema":
            answer = json.dumps(
                schema_value(request["format"]["schema"]), ensure_ascii=False
            )
        elif request["format"]["type"] == "json_object":
            answer = json.dumps({"token": rng.randrange(1, 248320)})
        else:
            count = rng.randint(1, max(1, min(64, limit - len(reasoning))))
            answer = "".join(rng.choices(string.ascii_lowercase, k=count))
        end = min(
            (answer.find(stop) for stop in request["stop"] if stop in answer),
            default=len(answer),
        )
        output = reasoning + list(answer[:end])
        if request.get("ignore_eos", False) and end == len(answer):
            output.extend([""] * max(0, limit - len(output)))
        finish = "length" if len(output) >= limit else "stop"
        if not output:
            output = [""]
        return cls(len(tokens), tokens, output[:limit], rng, len(reasoning), finish)

    def execute(self, row, draft_count):
        if "prefill_token_ids" in row:
            self.history = row["prefill_token_ids"]
        start = row["num_computed_tokens"]
        if start > len(self.history):
            raise ValueError("computed position exceeds accepted history")
        end = start + len(row["token_ids"])
        emitted = len(self.history) - self.prompt_len
        final_chunk = end >= len(self.history)
        drafts = max(0, end - len(self.history))
        accepted = (
            self.rng.randint(0, min(drafts, len(self.output) - emitted - 1))
            if final_chunk
            else 0
        )
        kept = self.output[emitted : emitted + accepted + 1] if final_chunk else []
        ids = [control_or_text_token(text) for text in kept]
        self.history.extend(ids)
        generated = emitted + len(ids)
        proposal = (
            self.output[generated : generated + draft_count] if final_chunk else []
        )
        return {
            "request_id": row["request_id"],
            "token_ids": ids,
            "num_accepted_draft_tokens": accepted,
            "new_draft_token_ids": [control_or_text_token(text) for text in proposal],
            "text": "".join(kept),
            "finish_reason": self.finish if generated == len(self.output) else None,
            "reasoning_tokens": min(generated, self.reason_tokens),
        }


class Worker:
    def __init__(self, config):
        if config["block_size"] != 784 or config["tensor_parallel_size"] != 1:
            raise ValueError("invalid cache layout")
        self.config = dict(config)
        if self.config["num_gpu_blocks"] is None:
            self.config["num_gpu_blocks"] = 1024
        self.requests = {}
        self.offload = StubOffload(self.config, self.requests)

    def ready(self):
        fa_blocks = self.config["num_gpu_blocks"]
        gdn_blocks = self.config["mamba_blocks"] or fa_blocks
        return {
            "type": "ready",
            "descriptor": build_descriptor(fa_blocks, gdn_blocks),
            "logical_num_blocks": fa_blocks,
            "mamba_blocks": gdn_blocks,
            "host_cache_bytes": self.offload.limit,
        }

    def handle(self, message):
        match message["type"]:
            case "prepare":
                self.offload.check_register(message["request_id"])
                request = Request.prepare(
                    message["request"], self.config["max_model_len"]
                )
                self.requests[message["request_id"]] = request
                return {
                    "type": "prepared",
                    "rpc_id": message["rpc_id"],
                    "prompt_token_ids": request.history,
                }
            case "register":
                self.offload.check_register(message["request_id"])
                self.requests[message["request_id"]] = Request.offline(
                    message["prompt_token_ids"], message.get("max_tokens", 64)
                )
            case "execute":
                for request_id in message["finished_request_ids"]:
                    self.offload.cancel(request_id)
                    self.requests.pop(request_id, None)
                self.offload.check_execute(
                    row["request_id"] for row in message["scheduled"]
                )
                outputs = [
                    self.requests[row["request_id"]].execute(
                        row, self.config["num_speculative_tokens"]
                    )
                    for row in message["scheduled"]
                ]
                return {
                    "type": "execute_result",
                    "rpc_id": message["rpc_id"],
                    "outputs": outputs,
                }
            case "abort":
                self.offload.cancel(message["request_id"])
                self.requests.pop(message["request_id"], None)
            case "cache":
                return self.offload.handle(message)
            case "stats":
                return {
                    "type": "stats_result",
                    "rpc_id": message["rpc_id"],
                    "stats": {
                        "placeholder": True,
                        "requests": len(self.requests),
                        "process_id": os.getpid(),
                        "gpu_allocated_bytes": 0,
                        "gpu_reserved_bytes": 0,
                        **self.offload.stats(),
                    },
                }
            case _:
                raise ValueError("unsupported worker message")
