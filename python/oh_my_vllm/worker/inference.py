import os

from .cache import Cache
from .device import Device
from .media import MediaProcessor
from .model import Model
from .request import Request
from .sampling import Sampling
from .speculation import Speculation
from .tokenizer import Tokenization
from .trace import Trace
from .weights import Weights


class Worker:
    def __init__(self, config):
        if config["block_size"] != 784 or config["tensor_parallel_size"] != 1:
            raise ValueError("invalid cache layout")
        self.config = config
        self.tokenizer = Tokenization(config["model_path"])
        self.device = Device()
        self.weights = Weights(
            self.device, config["model_path"], bool(config["num_speculative_tokens"])
        )
        self.trace = Trace(config, self.weights.checkpoint.identity)
        self.cache = Cache(self.device, config)
        self.media = MediaProcessor(self.device, self.weights, self.tokenizer)
        self.model = Model(
            self.device,
            self.weights,
            self.cache,
            bool(config["num_speculative_tokens"]),
            self.media,
        )
        self.speculation = (
            Speculation(self.device, self.weights, self.cache, self.media, self.trace)
            if config["num_speculative_tokens"]
            else None
        )
        self.requests = {}
        self.samplers = {}
        self.device.synchronize()

    def ready(self):
        return {
            "type": "ready",
            "logical_num_blocks": self.cache.blocks,
            "mamba_blocks": self.cache.mamba_blocks,
        }

    def release(self, request_id):
        self.cache.release(request_id)
        self.requests.pop(request_id, None)

    def sampler(self, rows):
        if rows not in self.samplers:
            self.samplers[rows] = Sampling(self.device, self.weights.sample, rows)
        return self.samplers[rows]

    def fragments(self, rows, limit, batch_limit):
        pending, size = [], 0
        for row in rows:
            offset = 0
            while offset < len(row["token_ids"]):
                count = min(len(row["token_ids"]) - offset, limit - size)
                fragment = dict(
                    row,
                    token_ids=row["token_ids"][offset : offset + count],
                    num_computed_tokens=row["num_computed_tokens"] + offset,
                )
                pending.append(fragment)
                offset += count
                size += count
                if (
                    size == limit
                    or len(pending) == batch_limit
                    or offset < len(row["token_ids"])
                ):
                    yield pending
                    pending, size = [], 0
        if pending:
            yield pending

    def execute_batch(self, rows, verify, graph):
        requests = [self.requests[row["request_id"]] for row in rows]
        batch = self.cache.batch(rows, requests, verify)
        self.trace.record(
            "target",
            verify=verify,
            requests=[
                {
                    "id": row["request_id"],
                    "start": row["num_computed_tokens"],
                    "count": len(row["token_ids"]),
                    "prompt": request.prompt_len,
                }
                for row, request in zip(rows, requests, strict=True)
            ],
        )
        (value, residual), hidden = self.model.forward(batch, graph)
        chosen = {}
        if verify and self.speculation is not None:
            candidates = self.speculation.accept(
                batch, value, residual, self.sampler(batch.rows), graph
            )
            chosen = dict(enumerate(candidates))
        else:
            final = [
                index
                for index, (row, request) in enumerate(zip(rows, requests, strict=True))
                if row["num_computed_tokens"] + len(row["token_ids"])
                >= len(request.history)
            ]
            if final:
                tokens, _ = self.sampler(len(final)).select(
                    value,
                    residual,
                    [batch.schedule.sequence_rows[index + 1] - 1 for index in final],
                    [requests[index] for index in final],
                    graph,
                )
                chosen = {
                    index: [int(token)]
                    for index, token in zip(final, tokens, strict=True)
                }
        outputs = []
        for index, (row, request) in enumerate(zip(rows, requests, strict=True)):
            tokens, text = (
                request.append(chosen[index]) if index in chosen else ([], "")
            )
            accepted = len(tokens) - 1 if verify and tokens else None
            self.cache.commit(request, batch, index, accepted)
            outputs.append(
                {
                    "request_id": row["request_id"],
                    "token_ids": tokens,
                    "num_accepted_draft_tokens": max(0, len(tokens) - 1),
                    "new_draft_token_ids": [],
                    "text": text,
                    "finish_reason": request.finish,
                    "reasoning_tokens": request.thinking_count,
                }
            )
        if self.speculation is not None:
            endpoints = self.speculation.context(batch, hidden, graph, chosen)
            self.speculation.propose(batch, endpoints, graph)
            for output, request in zip(outputs, requests, strict=True):
                output["new_draft_token_ids"] = request.drafts.copy()
        self.trace.record(
            "commit",
            requests=[
                {
                    "id": output["request_id"],
                    "tokens": len(output["token_ids"]),
                    "accepted": output["num_accepted_draft_tokens"],
                    "proposed": len(output["new_draft_token_ids"]),
                    "position": self.requests[output["request_id"]].state_position,
                }
                for output in outputs
            ],
        )
        return outputs

    def execute(self, message):
        for request_id in message["finished_request_ids"]:
            self.release(request_id)
        for request_id in message["preempted_request_ids"]:
            self.cache.release(request_id)
            request = self.requests[request_id]
            request.state_slot = None
            request.state_position = request.mtp_position = 0
            request.drafts = []
        groups = {False: [], True: []}
        for row in message["scheduled"]:
            request = self.requests[row["request_id"]]
            if "prefill_token_ids" in row:
                request.history = list(row["prefill_token_ids"])
            end = max(request.prompt_len, len(request.history) - 1)
            groups[row["num_computed_tokens"] >= end].append(row)
        outputs = {}
        graph = not groups[False]
        for verify in (True, False):
            batch_limit = 4 if self.speculation is not None else 20
            limit = 32764 if self.speculation is not None else 32768
            for rows in self.fragments(groups[verify], limit, batch_limit):
                for output in self.execute_batch(rows, verify, graph):
                    outputs[output["request_id"]] = output
        return {
            "type": "execute_result",
            "rpc_id": message["rpc_id"],
            "outputs": list(outputs.values()),
        }

    def handle(self, message):
        match message["type"]:
            case "prepare":
                options = message["request"]
                tokens, media, positions, delta = self.media.prepare(options)
                if len(tokens) + options["max_tokens"] > self.config["max_model_len"]:
                    raise ValueError("prompt plus output exceeds context limit")
                self.requests[message["request_id"]] = Request(
                    self.tokenizer, tokens, options
                )
                request = self.requests[message["request_id"]]
                request.media, request.rope_positions, request.rope_delta = (
                    media,
                    positions,
                    delta,
                )
                return {
                    "type": "prepared",
                    "rpc_id": message["rpc_id"],
                    "prompt_token_ids": tokens,
                }
            case "register":
                options = {
                    "max_tokens": message.get("max_tokens", 64),
                    "effort": "off",
                    "sampling": {"temperature": 0},
                    "format": {"type": "text"},
                    "tools": [],
                    "tool_choice": "none",
                }
                self.requests[message["request_id"]] = Request(
                    self.tokenizer,
                    message["prompt_token_ids"],
                    options,
                    fixed_length=True,
                )
            case "execute":
                return self.execute(message)
            case "abort":
                self.release(message["request_id"])
            case "stats":
                return {
                    "type": "stats_result",
                    "rpc_id": message["rpc_id"],
                    "stats": {
                        "placeholder": False,
                        "requests": len(self.requests),
                        "process_id": os.getpid(),
                        "gpu_free_bytes": self.device.module.free_bytes(),
                        "gpu_allocated_bytes": self.device.module.allocated_bytes(),
                        "gpu_reserved_bytes": self.device.module.allocated_bytes(),
                    },
                }
            case _:
                raise ValueError("unsupported worker message")

    def close(self):
        self.trace.close()
        if self.speculation is not None:
            self.speculation.close()
        self.model.close()
        for sampler in self.samplers.values():
            sampler.close()
        self.device.synchronize()
