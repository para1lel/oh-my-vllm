"""DeepSeek-V4.1-Flash worker adapter."""

import copy
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from ...worker.trace import Trace
from .adapter import INFERENCE_CONFIG_SHA256, MAX_CONTEXT_TOKENS, validate_config
from .cache import DeepSeekCache
from .descriptor import build_descriptor, checkpoint_metadata_identity
from .loader import RankLoader
from .model import ModelArgs, Transformer
from .request import Request
from .speculation import verify_greedy_drafts
from .tokenizer import Tokenization

DEFAULT_CACHE_CONTEXT = 65536
MAX_NATIVE_ROWS = 384


class _StartupTrace:
    def __init__(self, rank):
        template = os.environ.get("OH_MY_VLLM_STARTUP_TRACE")
        self.output = None
        if template:
            path = Path(template.format(rank=rank)).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.output = path.open("w")

    def record(self, phase, **values):
        if self.output is None:
            return
        record = {
            "phase": phase,
            "monotonic_s": time.monotonic(),
            **values,
        }
        self.output.write(json.dumps(record) + "\n")
        self.output.flush()

    @contextmanager
    def measure(self, phase):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(phase, duration_s=time.perf_counter() - started)

    def close(self):
        if self.output is not None:
            self.output.close()
            self.output = None


class Worker:
    def __init__(self, config, *, raw_config, strategy=None):
        if config["block_size"] != 128 or config["tensor_parallel_size"] != 4:
            raise ValueError("DeepSeek requires four ranks and 128-token blocks")
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError(
                "DeepSeek worker requires an initialized four-rank process group"
            )
        if dist.get_world_size() != 4:
            raise ValueError("DeepSeek worker requires a four-rank process group")
        root = Path(config["model_path"]).resolve()
        parsed = json.loads(raw_config)
        validate_config(parsed)
        strategy = (
            strategy or os.environ.get("OH_MY_VLLM_DEEPSEEK_STRATEGY", "tep4")
        ).lower()
        if strategy not in {"tp4", "tep4", "dep4"}:
            raise ValueError("DeepSeek strategy must be TP4, TEP4, or DEP4")
        model_strategy = "tep4" if strategy == "dep4" else strategy
        inference_bytes = (root / "inference/config.json").read_bytes()
        if hashlib.sha256(inference_bytes).hexdigest() != INFERENCE_CONFIG_SHA256:
            raise ValueError("DeepSeek inference configuration identity differs")
        inference_config = json.loads(inference_bytes)
        self.config = dict(config)
        self.root = root
        self.strategy = strategy
        self.dspark = config["num_speculative_tokens"] == 5
        self.rank = dist.get_rank()
        self.startup_trace = _StartupTrace(self.rank)
        self.startup_trace.record("process_group_ready")
        with self.startup_trace.measure("tokenizer"):
            self.tokenizer = Tokenization(root)
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.bfloat16)
        with self.startup_trace.measure("model_args"):
            args = ModelArgs(**inference_config)
        args.max_batch_size = config["max_num_seqs"]
        cache_context = int(
            os.environ.get("OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT", DEFAULT_CACHE_CONTEXT)
        )
        if cache_context < 1:
            raise ValueError("DeepSeek cache context must be positive")
        args.max_seq_len = min(
            config["max_model_len"], MAX_CONTEXT_TOKENS, cache_context
        )
        self.runtime_max_context = args.max_seq_len
        args.temperature = 0
        args.parallel_strategy = model_strategy.upper()
        self.args = args
        with self.startup_trace.measure("model_construct"), torch.device(self.device):
            self.model = Transformer(args, self.tokenizer.tokenizer)
        with self.startup_trace.measure("loader_construct"):
            self.weights = RankLoader(
                root,
                parsed,
                self.rank,
                strategy=model_strategy.upper(),
            )
        with self.startup_trace.measure("checkpoint_metadata"):
            self.checkpoint_identity = checkpoint_metadata_identity(
                root,
                raw_config=raw_config,
                weight_map=self.weights.index,
                index_sha256=self.weights.index_sha256,
            )
        with self.startup_trace.measure("weight_load"):
            self.weight_info = self.weights.load_into(self.model)
        self.model.eval()
        self.requests = {}
        self.slots = set(range(config["max_num_seqs"]))
        with self.startup_trace.measure("cache_construct"):
            self.cache = DeepSeekCache(
                self.model,
                block_tokens=config["block_size"],
                max_seq_len=args.max_seq_len,
                host_limit=int(config.get("host_cache_bytes") or 0),
                fa_capacity=config.get("num_gpu_blocks") or 2048,
                mamba_capacity=config.get("mamba_blocks") or 128,
                requests=self.requests,
                free_slots=self.slots,
            )
        with self.startup_trace.measure("descriptor"):
            self.descriptor = build_descriptor(
                parsed,
                raw_config=raw_config,
                checkpoint_identity=self.checkpoint_identity,
                tokenizer_identity=self.tokenizer.identity,
                strategy=model_strategy,
                cache_capacities={
                    "window": config.get("num_gpu_blocks") or 2048,
                    "compressed_kv": config.get("num_gpu_blocks") or 2048,
                    "index_keys": config.get("num_gpu_blocks") or 2048,
                    "accepted_state": config.get("mamba_blocks") or 128,
                },
            )
        self.logical_num_blocks = self.descriptor["cache_schema"]["groups"][0][
            "capacity_blocks"
        ]
        self.mamba_blocks = self.descriptor["cache_schema"]["groups"][-1][
            "capacity_blocks"
        ]
        with self.startup_trace.measure("execution_trace"):
            self.trace = Trace(
                config,
                self.checkpoint_identity,
                tokenizer=self.tokenizer.identity,
                cache_memory={
                    "model": "deepseek_v41",
                    "rank": self.rank,
                    "cache_context": args.max_seq_len,
                    "strategy": strategy,
                    "effective_strategy": model_strategy,
                },
                execution={
                    "prefill": "eager",
                    "decode_only": "eager",
                    "mixed": "eager",
                },
            )
        self.startup_trace.record("ready")

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
                },
            },
        }

    def _slot(self, request_id):
        request = self.requests[request_id]
        if getattr(request, "cache_slot", None) is None:
            if not self.slots:
                raise ValueError("DeepSeek request slot capacity is exhausted")
            request.cache_slot = self.slots.pop()
        return request.cache_slot

    def _free_slot(self, request):
        slot = getattr(request, "cache_slot", None)
        if slot is None:
            return
        if hasattr(self, "cache"):
            self.cache.clear_slot(slot)
        self.slots.add(slot)
        request.cache_slot = None

    def _release(self, request_id):
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
        if scheduled_set & (set(finished_ids) | set(preempted_ids)):
            raise ValueError("DeepSeek request lifecycle sets overlap")
        for request_id in (*scheduled_ids, *finished_ids, *preempted_ids):
            if request_id not in self.requests:
                raise ValueError("DeepSeek execute references an unknown request")
        for row in scheduled:
            request = self.requests[row["request_id"]]
            if getattr(request, "cache_slot", None) is not None:
                expected = int(row["num_computed_tokens"])
                actual = getattr(request, "state_position", None)
                if actual is not None and actual != expected:
                    raise ValueError(
                        "DeepSeek resident cache position differs from the schedule"
                    )

    def _target(self, token_ids, start_pos, slot, *, full_logits=False):
        cache_slots = torch.tensor([slot], device=self.device)
        results = []
        for offset in range(0, len(token_ids), MAX_NATIVE_ROWS):
            inputs = torch.tensor(
                [token_ids[offset : offset + MAX_NATIVE_ROWS]],
                device=self.device,
                dtype=torch.long,
            )
            results.append(
                self.model(
                    inputs,
                    start_pos=start_pos + offset,
                    cache_slots=cache_slots,
                    full_logits=full_logits,
                )
            )
        if len(results) == 1 or not full_logits:
            if len(results) == 1:
                return results[-1]
            output_ids, logits, _ = results[-1]
            hidden = [result[2] for result in results if result[2] is not None]
            if hidden and isinstance(hidden[0], torch.Tensor):
                return output_ids, logits, torch.cat(hidden, dim=1)
            return output_ids, logits, results[-1][2]
        output_ids, logits, main_hidden = results[-1]
        logits = torch.cat([result[1] for result in results], dim=1)
        hidden = [result[2] for result in results if result[2] is not None]
        if hidden:
            main_hidden = torch.cat(hidden, dim=1)
        return output_ids, logits, main_hidden

    def _proposal(self, token, main_hidden, start_pos, slot):
        if not self.dspark or main_hidden is None:
            return []
        input_ids = torch.tensor([[token]], device=self.device, dtype=torch.long)
        result = self.model.forward_spec(
            input_ids,
            main_hidden,
            start_pos=start_pos,
            cache_slots=torch.tensor([slot], device=self.device),
        )
        if result is None:
            return []
        output_ids, _, _ = result
        return output_ids[0, 1:].tolist()

    def _seed_mtp(self, main_hidden, start_pos, slot):
        if (
            not self.dspark
            or main_hidden is None
            or not hasattr(self, "model")
            or not hasattr(self.model, "forward_spec_seed")
        ):
            return False
        self.model.forward_spec_seed(
            main_hidden,
            start_pos=start_pos,
            cache_slots=torch.tensor([slot], device=self.device),
        )
        return True

    @staticmethod
    def _output(request, row, tokens, text, accepted=0, drafts=None):
        return {
            "request_id": row["request_id"],
            "token_ids": tokens,
            "num_accepted_draft_tokens": accepted,
            "new_draft_token_ids": [] if drafts is None else drafts,
            "text": text,
            "finish_reason": request.finish,
            "reasoning_tokens": request.thinking_count,
        }

    def _execute_row(self, row):
        request = self.requests[row["request_id"]]
        try:
            return self._execute_row_impl(row)
        except BaseException:
            self._free_slot(request)
            request.state_position = 0
            raise

    def _execute_row_impl(self, row):
        request = self.requests[row["request_id"]]
        if row.get("prefill_token_ids") is not None:
            request.history = list(row["prefill_token_ids"])
        slot_was_empty = request.cache_slot is None
        slot = self._slot(row["request_id"])
        token_ids = list(row["token_ids"])
        start_pos = int(row["num_computed_tokens"])
        if hasattr(self, "cache"):
            try:
                if slot_was_empty:
                    self.cache.restore(
                        row["request_id"],
                        slot,
                        row,
                        start_pos=start_pos,
                    )
                self.cache.ensure_mamba_row(row)
            except BaseException:
                if slot_was_empty:
                    self._free_slot(request)
                raise
        if not token_ids:
            return self._output(request, row, [], "")

        known_count = max(0, min(len(token_ids), len(request.history) - start_pos))
        final = start_pos + len(token_ids) >= len(request.history)
        accepted_end = start_pos
        mtp_state_updated = False
        if not final:
            _, _, main_hidden = self._target(token_ids, start_pos, slot)
            mtp_state_updated = self._seed_mtp(main_hidden, start_pos, slot)
            accepted_end = start_pos + len(token_ids)
            tokens, text = [], ""
            accepted = 0
            proposals = []
        else:
            known = token_ids[:known_count]
            if not known:
                raise ValueError("DeepSeek execution row has no accepted input")
            _, logits, main_hidden = self._target(known, start_pos, slot)
            mtp_state_updated = self._seed_mtp(main_hidden, start_pos, slot)
            target_predictions = [int(logits.argmax(dim=-1).item())]
            drafts = token_ids[known_count:]
            accepted_drafts = []
            current_logits = logits
            mismatch = None
            for index, draft in enumerate(drafts):
                target = int(current_logits.argmax(dim=-1).item())
                target_predictions[index] = target
                if draft != target:
                    mismatch = index
                    break
                accepted_drafts.append(draft)
                _, current_logits, current_hidden = self._target(
                    [draft],
                    start_pos + known_count + index,
                    slot,
                )
                mtp_state_updated |= self._seed_mtp(
                    current_hidden,
                    start_pos + known_count + index,
                    slot,
                )
                target_predictions.append(int(current_logits.argmax(dim=-1).item()))
            if not drafts:
                target_predictions = [int(logits.argmax(dim=-1).item())]
                verified_drafts = []
            elif mismatch is not None:
                target_predictions.append(target_predictions[-1])
                verified_drafts = drafts[: mismatch + 1]
            else:
                verified_drafts = drafts
            verification = verify_greedy_drafts(verified_drafts, target_predictions)
            emitted = list(verification.token_ids)
            tokens, text = request.append(emitted)
            accepted = min(
                verification.accepted_draft_tokens,
                max(0, len(tokens) - 1),
            )
            accepted_end = start_pos + known_count + accepted
            proposals = []
            if not drafts and self.dspark and not request.finish:
                mtp_state_updated = True
                proposals = self._proposal(
                    int(target_predictions[0]),
                    main_hidden[:, -1:],
                    start_pos + known_count - 1,
                    slot,
                )
                remaining = max(0, request.limit - request.generated)
                proposals = proposals[: max(0, remaining - 1)]
        if hasattr(self, "cache"):
            self.cache.commit(
                row["request_id"],
                row,
                slot,
                start_pos=start_pos,
                accepted_end=accepted_end,
                include_mtp=mtp_state_updated,
            )
            if final and row["token_ids"][known_count:]:
                self.cache.restore(
                    row["request_id"],
                    slot,
                    row,
                    start_pos=accepted_end,
                    use_prefix_state=False,
                )
        request.state_position = accepted_end
        self.trace.record(
            "target",
            verify=bool(token_ids[known_count:]) if final else False,
            execution="eager",
            requests=[
                {
                    "id": row["request_id"],
                    "start": row["num_computed_tokens"],
                    "count": len(token_ids),
                    "prompt": request.prompt_len,
                }
            ],
        )
        return self._output(request, row, tokens, text, accepted, proposals)

    def execute(self, message):
        self._validate_execute_message(message)
        for request_id in message["finished_request_ids"]:
            self._release(request_id)
        for request_id in message["preempted_request_ids"]:
            request = self.requests.get(request_id)
            if request is not None:
                if (
                    hasattr(self, "cache")
                    and getattr(request, "cache_slot", None) is not None
                ):
                    self._free_slot(request)
                request.cache_slot = None
                request.state_position = 0
        for row in message["scheduled"]:
            if hasattr(self, "cache") and row.get("fa_block_copies"):
                self.cache.copy_prefix(row["fa_block_copies"])
        outputs = [self._execute_row(row) for row in message["scheduled"]]
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
                request.cache_slot = None
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
                request.cache_slot = None
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
                        "placeholder": False,
                        "model": "deepseek_v41",
                        "rank": self.rank,
                        "requests": len(self.requests),
                        "gpu_free_bytes": torch.cuda.mem_get_info(self.device)[0],
                        **self.cache.stats(),
                    },
                }
            case _:
                raise ValueError("unsupported worker message")

    def close(self):
        try:
            self.trace.close()
            self.cache.close()
            del self.model
            self.weights.close()
            torch.cuda.empty_cache()
        finally:
            self.startup_trace.close()
