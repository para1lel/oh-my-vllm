import copy
import os

import numpy as np

from ...operators.gdn_radix.api import Schedule
from ...operators.output.api import FUNCTIONS as OUTPUT_FUNCTIONS
from ...operators.runtime.device import Device
from ...operators.runtime.native import load
from ...worker.trace import Trace
from .cache import Batch, Cache, page_valid_ends
from .cache_budget import PAGE_TOKENS, cache_budget, layout_template, target_token_limit
from .cache_wire import sequence_plan
from .descriptor import build_descriptor
from .media import MediaProcessor
from .memory import workspace_budget
from .model import Model
from .request import Request
from .sampling import Sampling
from .speculation import Speculation
from .tokenizer import Tokenization
from .trace_identity import dispatch_facts, loaded_identity, record_prepared
from .weights import Weights


class Worker:
    def __init__(self, config, *, raw_config):
        if config["block_size"] != PAGE_TOKENS or config["tensor_parallel_size"] != 1:
            raise ValueError("invalid Qwen radix cache layout")
        layout_template(config)
        self.epoch = config["worker_epoch"]
        self.tokenizer = Tokenization(config["model_path"])
        self.device = Device()
        self.weights = Weights(
            self.device, config["model_path"], bool(config["num_speculative_tokens"])
        )
        output_module, _, _ = load("sample", OUTPUT_FUNCTIONS)
        sample_rows = min(
            config["max_num_seqs"], 4 if config["num_speculative_tokens"] else 20
        ) * (5 if config["num_speculative_tokens"] else 1)
        native_sort = {
            rows: output_module.sample_workspace(rows)
            for rows in range(1, sample_rows + 1)
        }
        workspace = workspace_budget(config, self.weights, native_sort=native_sort)
        self.device.synchronize()
        free, total = self.device.memory_info()
        self.cache_memory = cache_budget(
            config,
            free_bytes=free,
            total_bytes=total,
            overhead_bytes=workspace["prefill"] + workspace["retained"],
        )
        self.cache_memory["workspace_bytes"] = workspace
        config = dict(
            config,
            num_gpu_blocks=self.cache_memory["num_gpu_blocks"],
            mamba_blocks=self.cache_memory["mamba_blocks"],
        )
        self.config = config
        self.cache = Cache(self.device, config)
        self.media = MediaProcessor(self.device, self.weights, self.tokenizer)
        self.model = Model(
            self.device,
            self.weights,
            self.cache,
            bool(config["num_speculative_tokens"]),
            self.media,
        )
        self.device.module.reserve_scratch(workspace["prefill"])
        self.requests, self.samplers = {}, {}
        self.empty_hidden = self.device.scratch((0, 5120), "bfloat16")
        self.device.synchronize()
        free, total = self.device.memory_info()
        self.cache_memory.update(
            initialized_free_bytes=free, initialized_used_bytes=total - free
        )
        self.tokenizer.verify_identity(self.tokenizer.identity)
        self.descriptor = build_descriptor(
            self.weights.config,
            raw_config=raw_config,
            checkpoint_identity=self.weights.checkpoint.complete_identity(),
            tokenizer_identity=self.tokenizer.identity,
            fa_capacity_blocks=self.cache.blocks + 1,
            gdn_capacity_blocks=self.cache.mamba_blocks + 1,
            num_speculative_tokens=config["num_speculative_tokens"],
        )
        self.trace = Trace(
            config,
            self.weights.checkpoint.identity,
            tokenizer=self.tokenizer.identity,
            cache_memory=self.cache_memory,
            cache_layout=self.cache.layout.wire(),
            loaded=loaded_identity(
                self.weights.checkpoint.identity, self.tokenizer.identity
            ),
        )
        self.speculation = (
            Speculation(self.device, self.weights, self.cache, self.media, self.trace)
            if config["num_speculative_tokens"]
            else None
        )
        self.last_step = 0

    def ready(self):
        return {
            "type": "ready",
            "protocol_version": 2,
            "worker_epoch": self.epoch,
            "descriptor": copy.deepcopy(self.descriptor),
            "logical_num_blocks": self.cache.blocks,
            "mamba_blocks": self.cache.mamba_blocks,
            "cache_layout": self.cache.layout.wire(),
            "cache_memory": self.cache_memory,
        }

    def release(self, request_id):
        self.device.wait_pinned_safe()
        self.requests.pop(request_id, None)

    def sampler(self, rows):
        if rows not in self.samplers:
            self.samplers[rows] = Sampling(self.device, self.weights.sample, rows)
        return self.samplers[rows]

    def batches(self, rows):
        pending, size = [], 0
        limit = target_token_limit(self.config)
        for row in rows:
            count = len(row["token_ids"])
            if count > limit:
                raise ValueError(
                    "scheduled Qwen forward exceeds the prepared row limit"
                )
            if pending and (
                size + count > limit or len(pending) == (4 if self.speculation else 20)
            ):
                yield pending
                pending, size = [], 0
            pending.append(row)
            size += count
        if pending:
            yield pending

    def execute_batch(self, rows, verify, graph):
        requests = [self.requests[row["request_id"]] for row in rows]
        batch = self.cache.batch(rows, requests, verify)
        copies = [item for row in rows for item in row["model_state"]["page_copies"]]
        self.cache.validate_pages(batch.plans, copies)
        self.cache.copy_prefix(copies)
        (value, residual), hidden, graph = self.model.forward(batch, graph)
        self.cache.complete_exports(batch)
        chosen = self.sample_batch(batch, value, residual, verify, graph)
        outputs = []
        for index, (row, request) in enumerate(zip(rows, requests, strict=True)):
            tokens, text = (
                request.append(chosen[index]) if index in chosen else ([], "")
            )
            accepted = len(tokens) - 1 if verify else None
            if verify and not tokens:
                raise ValueError("verification returned no accepted output token")
            self.cache.commit(request, batch, index, accepted)
            outputs.append(self.result(row, request, batch.plans[index], tokens, text))
        if self.speculation is not None:
            endpoints = self.speculation.context(batch, hidden, graph, chosen)
            self.cache.complete_context(batch)
            if any(row["purpose"] == "advance" for row in rows):
                self.speculation.propose(batch, endpoints, graph)
            for output, request in zip(outputs, requests, strict=True):
                output["new_draft_token_ids"] = (
                    request.drafts.copy()
                    if output["purpose"] == "advance" and not request.finish
                    else []
                )
                output["mtp_accepted_endpoint"] = self.mtp_endpoint(request)
        self.trace.record(
            "dispatch",
            worker_epoch=self.epoch,
            step_id=self.trace_step,
            **dispatch_facts(rows, verify, graph),
        )
        self.trace.record(
            "target",
            verify=verify,
            execution="cuda_graph" if graph else "eager",
            requests=[
                {
                    "id": row["request_id"],
                    "start": row["forward_start"],
                    "count": len(row["token_ids"]),
                }
                for row in rows
            ],
        )
        return outputs

    def sample_batch(self, batch, value, residual, verify, graph):
        if verify:
            if self.speculation is None:
                raise ValueError("Qwen verification requires MTP")
            return dict(
                enumerate(
                    self.speculation.accept(
                        batch, value, residual, self.sampler(batch.rows), graph
                    )
                )
            )
        selected = [
            index
            for index, row in enumerate(batch.scheduled)
            if row["sampling"]["kind"] == "target"
        ]
        if not selected:
            return {}
        tokens, _ = self.sampler(len(selected)).select(
            value,
            residual,
            [batch.schedule.sequence_rows[index + 1] - 1 for index in selected],
            [batch.requests[index] for index in selected],
            graph,
        )
        return {
            index: [int(token)] for index, token in zip(selected, tokens, strict=True)
        }

    def mtp_endpoint(self, request):
        if self.speculation is None:
            return None
        end = request.state_position
        source = request.endpoints[end]
        self.cache.check_endpoint(source)
        if self.cache.mtp_contexts.get(source.slot) != end - 1:
            raise ValueError("accepted endpoint has incomplete MTP context")
        result = source.wire()
        result["end"] = max(0, end - 1)
        return result

    def result(self, row, request, plan, tokens=(), text=""):
        final = row["purpose"] == "finalize"
        return {
            "request_id": row["request_id"],
            "lease_id": row["lease_id"],
            "purpose": row["purpose"],
            "action": row["action"],
            "error": None,
            "token_ids": list(tokens),
            "num_accepted_draft_tokens": max(0, len(tokens) - 1),
            "new_draft_token_ids": [],
            "text": "" if final else text,
            "finish_reason": None if final else request.finish,
            "reasoning_tokens": 0 if final else request.thinking_count,
            "computed_end": request.computed_end,
            "accepted_end": row["accepted_end"] if final else len(request.history),
            "validated_exports": [value.wire() for value in plan.exports],
            "committed_page_valid_ends": page_valid_ends(plan, request.computed_end),
            "mtp_accepted_endpoint": None,
        }

    def endpoint_batch(self, row, request, plan, source):
        tables = np.asarray(
            [[page["slot"] if page is not None else 0 for page in plan.pages]], np.int32
        )
        schedule = Schedule(
            (0, 0), (source.slot,), (), (), (), (source.end,), self.cache.slots, False
        )
        metadata = (
            np.zeros((3, 0), np.int64),
            np.zeros(0, np.int32),
            tables,
            np.zeros(2, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.int8),
        )
        return Batch(
            [row], [request], schedule, np.zeros(0, np.int64), metadata, (), (plan,), ()
        )

    def execute_endpoint(self, row):
        request = self.requests[row["request_id"]]
        source = self.cache.endpoint_source(row, request)
        self.cache.validate_pages(
            (sequence_plan(row, self.cache.layout),), row["model_state"]["page_copies"]
        )
        self.cache.copy_prefix(row["model_state"]["page_copies"])
        plan = self.cache.export_endpoint(row, request)
        request.mtp_position = min(request.mtp_position, source.end - 1)
        tokens, text = [], ""
        if row["action"] == "head_only":
            hidden = self.device.view(
                self.cache.head_hidden, (1, 5120), source.slot * 5120
            )
            selected, _ = self.sampler(1).select_hidden(hidden, [request], False)
            tokens, text = request.append([int(selected[0])])
            if len(tokens) != 1:
                raise ValueError("head-only must return one target token")
            if self.speculation is not None:
                batch = self.endpoint_batch(row, request, plan, source)
                endpoints = self.speculation.context(
                    batch, self.empty_hidden, False, {0: tokens}
                )
                self.cache.complete_context(batch)
                self.speculation.propose(batch, endpoints, False)
        output = self.result(row, request, plan, tokens, text)
        output["new_draft_token_ids"] = (
            request.drafts.copy()
            if row["action"] == "head_only" and not request.finish
            else []
        )
        output["mtp_accepted_endpoint"] = self.mtp_endpoint(request)
        return output

    def execute(self, message):
        self.device.ensure_healthy()
        if (
            message["worker_epoch"] != self.epoch
            or message["step_id"] <= self.last_step
        ):
            raise ValueError("stale Qwen execution epoch or step")
        for row in message["scheduled"]:
            sequence_plan(row, self.cache.layout)
        self.trace.record(
            "worker_scheduled",
            **{key: value for key, value in message.items() if key != "type"},
        )
        self.device.wait_pinned_safe()
        for request_id in message["finished_request_ids"]:
            self.release(request_id)
        for request_id in message["preempted_request_ids"]:
            request = self.requests[request_id]
            request.state_slot = None
            request.state_position = request.computed_end = request.kv_valid_end = (
                request.mtp_position
            ) = 0
            request.endpoints, request.drafts = {}, []
            request.proposals, request.proposal_rows = None, ()
            self.media.offload(request)
        groups = {False: [], True: []}
        outputs = {}
        for row in message["scheduled"]:
            request = self.requests[row["request_id"]]
            if "prefill_token_ids" in row:
                request.history = list(row["prefill_token_ids"])
            if self.speculation is not None:
                request.mtp_position = self.cache.restore_context(row)
            if row["action"] != "forward":
                outputs[row["request_id"]] = self.execute_endpoint(row)
            else:
                groups[row["sampling"]["kind"] == "verify"].append(row)
        self.trace_step = message["step_id"]
        for verify in (True, False):
            for rows in self.batches(groups[verify]):
                for output in self.execute_batch(
                    rows, verify, None if groups[False] else True
                ):
                    outputs[output["request_id"]] = output
        self.device.wait_pinned_safe()
        self.last_step = message["step_id"]
        result = {
            "type": "execute_result",
            "rpc_id": message["rpc_id"],
            "step_id": message["step_id"],
            "worker_epoch": self.epoch,
            "outputs": [outputs[row["request_id"]] for row in message["scheduled"]],
            "rank_completions": [],
        }
        self.trace.record("worker_result", **result)
        return result

    def register(self, request_id, options, *, tokens=None, fixed_length=False):
        if request_id in self.requests:
            raise ValueError("request identifier is already owned by the worker")
        if tokens is None:
            tokens, media, positions, delta = self.media.prepare(options)
        else:
            media, positions, delta = [], None, 0
        if len(tokens) + options["max_tokens"] > self.config["max_model_len"]:
            raise ValueError("prompt plus output exceeds context limit")
        request = Request(self.tokenizer, tokens, options, fixed_length=fixed_length)
        request.media, request.rope_positions, request.rope_delta = (
            media,
            positions,
            delta,
        )
        self.requests[request_id] = request
        return tokens

    def handle(self, message):
        match message["type"]:
            case "prepare":
                tokens = self.register(message["request_id"], message["request"])
                record_prepared(
                    self.trace,
                    self.epoch,
                    message,
                    self.requests[message["request_id"]],
                    tokens,
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
                self.register(
                    message["request_id"],
                    options,
                    tokens=message["prompt_token_ids"],
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
                        "requests": len(self.requests),
                        "process_id": os.getpid(),
                        "gpu_free_bytes": self.device.module.free_bytes(),
                        "gpu_allocated_bytes": self.device.module.allocated_bytes(),
                        "gpu_reserved_bytes": self.device.module.reserved_bytes(),
                    },
                }
            case _:
                raise ValueError("unsupported worker message")

    def close(self):
        self.device.wait_pinned_safe()
        self.trace.close()
        if self.speculation is not None:
            self.speculation.close()
        self.model.close()
        for sampler in self.samplers.values():
            sampler.close()
        self.device.wait_pinned_safe()
        self.device.close()
