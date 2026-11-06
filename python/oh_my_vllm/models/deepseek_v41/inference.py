import copy

import torch

from .model import context_token_bucket
from .request import Request
from .row_execution import RowExecution
from .target_batch import BatchExecution
from .target_execution import TargetExecution
from .worker_limits import prefill_row_limit as prefill_row_limit
from .worker_limits import prefill_subbatch_limit as prefill_subbatch_limit
from .worker_startup import WorkerStartup


class Worker(WorkerStartup, TargetExecution, BatchExecution, RowExecution):
    def ready(self):
        return {
            "type": "ready",
            "descriptor": copy.deepcopy(self.descriptor),
            "logical_num_blocks": self.logical_num_blocks,
            "mamba_blocks": self.mamba_blocks,
            "host_cache_bytes": self.cache.host_limit,
            "cache_memory": {
                "weight_info": self.weight_info,
                "rank": self.rank,
                "physical_cache": {
                    "block_tokens": self.cache.block_tokens,
                    "host_cache_bytes": self.cache.host_limit,
                    "fa_capacity": self.cache.fa_capacity,
                    "accepted_state_capacity": self.cache.mamba_capacity,
                    "request_slots": self.args.max_batch_size,
                    "context_tokens": self.runtime_max_context,
                    "kv_dtype": "bf16",
                },
            },
        }

    def _slot(self, request_id):
        request = self.requests[request_id]
        if request.cache_slot is None:
            if not self.slots:
                raise ValueError("DeepSeek request slot capacity is exhausted")
            request.cache_slot = self.slots.pop()
        return request.cache_slot

    def _free_slot(self, request):
        slot = request.cache_slot
        if slot is None:
            return
        self._close_proposal_graph(slot)
        self.cache.clear_slot(slot)
        self.slots.add(slot)
        request.cache_slot = None

    def _release(self, request_id):
        accounting = getattr(getattr(self, "dep4_executor", None), "accounting", None)
        if accounting is not None:
            accounting.frontiers.pop(request_id, None)
        getattr(self, "target_trace_frontiers", {}).pop(request_id, None)
        request = self.requests.pop(request_id, None)
        if request is not None:
            self._free_slot(request)

    def _validate_execute_message(self, message):
        scheduled = message["scheduled"]
        scheduled_ids = [row["request_id"] for row in scheduled]
        finished_ids = list(message["finished_request_ids"])
        preempted_ids = list(message["preempted_request_ids"])
        for name, values in (
            ("scheduled", scheduled_ids),
            ("finished", finished_ids),
            ("preempted", preempted_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"DeepSeek {name} request IDs must be unique")
        scheduled_set = set(scheduled_ids)
        preempted_set = set(preempted_ids)
        if set(finished_ids) & (scheduled_set | preempted_set):
            raise ValueError("DeepSeek request lifecycle sets overlap")
        for request_id in (*scheduled_ids, *finished_ids, *preempted_ids):
            if request_id not in self.requests:
                raise ValueError("DeepSeek execute references an unknown request")
        for row in scheduled:
            if row["request_id"] in preempted_set:
                # Preemption releases the resident position before this row is restored.
                continue
            request = self.requests[row["request_id"]]
            if request.cache_slot is not None:
                expected = int(row["num_computed_tokens"])
                actual = request.state_position
                if actual != expected:
                    raise ValueError(
                        "DeepSeek resident cache position differs from the schedule"
                    )

    def execute(self, message):
        self._validate_execute_message(message)
        for request_id in message["finished_request_ids"]:
            self._release(request_id)
        for request_id in message["preempted_request_ids"]:
            request = self.requests.get(request_id)
            if request is not None:
                self._free_slot(request)
                request.state_position = 0
        for row in message["scheduled"]:
            if row.get("fa_block_copies"):
                self.cache.copy_prefix(row["fa_block_copies"])
        initial_prefill_groups = {}
        prefill_groups = {}
        decode_groups = {}
        for row in message["scheduled"]:
            request = self.requests[row["request_id"]]
            if row.get("prefill_token_ids") is not None:
                request.history = list(row["prefill_token_ids"])
            token_ids = list(row["token_ids"])
            start_pos = int(row["num_computed_tokens"])
            end_pos = start_pos + len(token_ids)
            if (
                self.batch_decode_enabled
                and token_ids
                and start_pos >= request.prompt_len
                and len(request.history) - start_pos == 1
                and 1 <= len(token_ids) <= 6
            ):
                length = len(token_ids)
                window = self.args.window_size
                key = (
                    length,
                    start_pos % self.decode_cycle,
                    context_token_bucket(end_pos),
                    min(window, start_pos) if length > 1 else window,
                )
                decode_groups.setdefault(key, []).append(row)
            elif (
                token_ids
                and start_pos < request.prompt_len
                and end_pos == len(request.history)
            ):
                initial_prefill_groups.setdefault(
                    (start_pos, len(token_ids)),
                    [],
                ).append(row)
            elif token_ids and end_pos < len(request.history):
                prefill_groups.setdefault(
                    (start_pos, len(token_ids)),
                    [],
                ).append(row)
        outputs_by_id = {}
        executed_ids = set()
        for rows in initial_prefill_groups.values():
            outputs_by_id.update(self._execute_prefill_group(rows, final=True))
            executed_ids.update(row["request_id"] for row in rows)
        for rows in prefill_groups.values():
            outputs_by_id.update(self._execute_prefill_group(rows))
            executed_ids.update(row["request_id"] for row in rows)
        for rows in decode_groups.values():
            outputs_by_id.update(self._execute_decode_group(rows))
            executed_ids.update(row["request_id"] for row in rows)
        for row in message["scheduled"]:
            if row["request_id"] not in executed_ids:
                outputs_by_id[row["request_id"]] = self._execute_row(row)
        outputs = [outputs_by_id[row["request_id"]] for row in message["scheduled"]]
        return {
            "type": "execute_result",
            "rpc_id": message["rpc_id"],
            "outputs": outputs,
        }

    def handle(self, message):
        match message["type"]:
            case "prepare":
                if message["request_id"] in self.requests:
                    raise ValueError("DeepSeek request identifier is already owned")
                options = message["request"]
                tokens = self.tokenizer.prepare(options)
                if len(tokens) + options["max_tokens"] > self.runtime_max_context:
                    raise ValueError("prompt plus output exceeds context limit")
                request = Request(self.tokenizer, tokens, options)
                self.requests[message["request_id"]] = request
                return {
                    "type": "prepared",
                    "rpc_id": message["rpc_id"],
                    "prompt_token_ids": tokens,
                }
            case "register":
                if message["request_id"] in self.requests:
                    raise ValueError("DeepSeek request identifier is already owned")
                if (
                    len(message["prompt_token_ids"]) + message.get("max_tokens", 64)
                    > self.runtime_max_context
                ):
                    raise ValueError("prompt plus output exceeds context limit")
                options = {
                    "max_tokens": message.get("max_tokens", 64),
                    "effort": "off",
                    "sampling": {"temperature": 0},
                    "format": {"type": "text"},
                    "tools": [],
                    "tool_choice": "none",
                }
                request = Request(
                    self.tokenizer,
                    message["prompt_token_ids"],
                    options,
                    fixed_length=True,
                )
                self.requests[message["request_id"]] = request
            case "execute":
                return self.execute(message)
            case "cache":
                completed = self.cache.handle(message)
                return {
                    "type": "cache_result",
                    "rpc_id": message["rpc_id"],
                    "completed": completed,
                    "host_cache_bytes": self.cache.host_limit,
                }
            case "abort":
                self._release(message["request_id"])
            case "stats":
                return {
                    "type": "stats_result",
                    "rpc_id": message["rpc_id"],
                    "stats": {
                        "model": "deepseek_v41",
                        "rank": self.rank,
                        "requests": len(self.requests),
                        "weight_info": self.weight_info,
                        "fa_capacity": self.cache.fa_capacity,
                        "accepted_state_capacity": self.cache.mamba_capacity,
                        "request_slots": self.args.max_batch_size,
                        "context_tokens": self.runtime_max_context,
                        "decode_graph_enabled": self.decode_graph_enabled,
                        "decode_graph_limit": self.decode_graph_limit,
                        "decode_graphs": len(self.decode_graphs),
                        "gpu_free_bytes": torch.cuda.mem_get_info(self.device)[0],
                        **self.cache.stats(),
                    },
                }
            case _:
                raise ValueError("unsupported worker message")

    def close(self):
        try:
            torch.cuda.synchronize(self.device)
            for slot in tuple(getattr(self, "proposal_graphs", {})):
                self._close_proposal_graph(slot)
            for graph in self.decode_graphs.values():
                graph.close()
            self.decode_graphs.clear()
            for bindings in self.decode_bindings.values():
                bindings.close()
            self.decode_bindings.clear()
            self.decode_engram.clear()
            self.trace.close()
            self.cache.close()
            del self.model
            self.weights.close()
            torch.cuda.empty_cache()
        finally:
            self.startup_trace.close()
