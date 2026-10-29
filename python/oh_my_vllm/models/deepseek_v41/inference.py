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
from .adapter import (
    CONFIG_SHA256,
    DS_PARK_DRAFT_TOKENS,
    INFERENCE_CONFIG_SHA256,
    MAX_CONTEXT_TOKENS,
    validate_config,
)
from .cache import DeepSeekCache
from .descriptor import build_descriptor
from .loader import DEFAULT_TILE_ROWS, RankLoader
from .model import ModelArgs, Transformer
from .request import Request
from .speculation import verify_greedy_drafts
from .tokenizer import Tokenization

DEFAULT_CACHE_CONTEXT = 65536
MAX_NATIVE_ROWS = 384
MAX_PREFILL_ROWS = 8192
DEFAULT_PREFILL_ROWS = MAX_PREFILL_ROWS
MAX_PREFILL_SUBBATCH = 4


def prefill_row_limit(value=None):
    rows = DEFAULT_PREFILL_ROWS if value is None else int(value)
    if not 1 <= rows <= MAX_PREFILL_ROWS:
        raise ValueError("DeepSeek prefill row limit must be between one and 8192")
    return rows


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
        self.prefill_rows = prefill_row_limit(
            os.environ.get("OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS")
        )
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
        args.lazy_cache = True
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
        tile_rows = int(
            os.environ.get("OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS", DEFAULT_TILE_ROWS)
        )
        with self.startup_trace.measure("model_construct"), torch.device(self.device):
            self.model = Transformer(args, self.tokenizer.tokenizer)
        self._input_host = self._host_input_buffer(
            config["max_num_seqs"], self.prefill_rows
        )
        self._input_device = torch.empty(
            (config["max_num_seqs"], self.prefill_rows),
            dtype=torch.long,
            device=self.device,
        )
        self._input_copy_event = None
        with self.startup_trace.measure("loader_construct"):
            self.weights = RankLoader(
                root,
                parsed,
                self.rank,
                strategy=model_strategy.upper(),
                tile_rows=tile_rows,
            )
        self.checkpoint_identity = {
            "config": CONFIG_SHA256,
            "index": self.weights.index_sha256,
        }
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
                    "window": config.get("mamba_blocks") or 128,
                    "compressed_kv": config.get("num_gpu_blocks") or 2048,
                    "index_keys": config.get("num_gpu_blocks") or 2048,
                    "accepted_state": config.get("mamba_blocks") or 128,
                },
            )
        self.logical_num_blocks = self.cache.fa_capacity
        self.mamba_blocks = self.cache.mamba_capacity
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

    @staticmethod
    def _host_input_buffer(rows, columns):
        try:
            return torch.empty(
                (rows, columns),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            return torch.empty((rows, columns), dtype=torch.long, device="cpu")

    def _input_tokens(self, token_batches):
        rows = len(token_batches)
        if not hasattr(self, "_input_device"):
            return torch.tensor(token_batches, device=self.device, dtype=torch.long)
        if not rows:
            raise ValueError("DeepSeek token input must contain a row")
        length = len(token_batches[0])
        if (
            length < 1
            or any(len(tokens) != length for tokens in token_batches)
            or rows > self._input_device.size(0)
            or length > self._input_device.size(1)
        ):
            raise ValueError("DeepSeek token input exceeds the prepared buffer")
        event = self._input_copy_event
        if event is not None:
            event.synchronize()
        source = torch.as_tensor(token_batches, dtype=torch.long, device="cpu")
        host = self._input_host[:rows, :length]
        host.copy_(source)
        device = self._input_device[:rows, :length]
        device.copy_(host, non_blocking=True)
        if event is None:
            event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._input_copy_event = event
        return device

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

    def _target(
        self,
        token_ids,
        start_pos,
        slot,
        *,
        full_logits=False,
        compute_logits=True,
        logit_start=0,
    ):
        cache_slots = torch.tensor([slot], device=self.device)
        output_ids = logits = main_hidden = None
        logit_chunks = [] if full_logits else None
        limit_rows = getattr(self, "prefill_rows", DEFAULT_PREFILL_ROWS)
        for offset in range(0, len(token_ids), limit_rows):
            inputs = self._input_tokens((token_ids[offset : offset + limit_rows],))
            chunk_logit_start = max(0, logit_start - offset) if full_logits else 0
            output_ids, chunk_logits, chunk_hidden = self.model(
                inputs,
                start_pos=start_pos + offset,
                cache_slots=cache_slots,
                full_logits=full_logits,
                compute_logits=compute_logits,
                collect_main_hidden=getattr(self, "dspark", False),
                logit_start=chunk_logit_start,
                sample_output=not full_logits,
                cache_slots_cpu=(slot,),
            )
            logits = chunk_logits
            if logit_chunks is not None:
                logit_chunks.append(chunk_logits)
            if isinstance(chunk_hidden, torch.Tensor):
                self._seed_mtp_checkpoints(
                    chunk_hidden,
                    main_hidden,
                    start_pos + offset,
                    cache_slots,
                    (slot,),
                    start_pos + len(token_ids),
                )
                if isinstance(main_hidden, torch.Tensor):
                    main_hidden = torch.cat((main_hidden, chunk_hidden), dim=1)
                else:
                    main_hidden = chunk_hidden
                limit = getattr(
                    getattr(self, "args", None),
                    "window_size",
                    MAX_NATIVE_ROWS,
                )
                limit += DS_PARK_DRAFT_TOKENS if full_logits else 0
                main_hidden = main_hidden[:, -limit:].clone()
            elif main_hidden is None:
                main_hidden = chunk_hidden
            del chunk_hidden
        if logit_chunks is not None:
            logits = torch.cat(logit_chunks, dim=1)
        return output_ids, logits, main_hidden

    def _main_hidden_tail(self, results):
        hidden = [
            result[2] for result in results if isinstance(result[2], torch.Tensor)
        ]
        if not hidden:
            return results[-1][2]
        limit = getattr(getattr(self, "args", None), "window_size", MAX_NATIVE_ROWS)
        tail = hidden[0]
        for value in hidden[1:]:
            tail = torch.cat((tail, value), dim=1)
            if tail.size(1) > limit:
                tail = tail[:, -limit:]
        return tail[:, -limit:]

    def _target_batch(
        self,
        token_batches,
        start_pos,
        cache_slots,
        *,
        full_logits=False,
        compute_logits=True,
    ):
        if not token_batches:
            raise ValueError("DeepSeek target batch must contain a request")
        length = len(token_batches[0])
        if length < 1 or any(len(tokens) != length for tokens in token_batches):
            raise ValueError("DeepSeek target batch rows must have one length")
        groups = []
        for offset in range(0, len(token_batches), MAX_PREFILL_SUBBATCH):
            groups.append(
                self._target_batch_group(
                    token_batches[offset : offset + MAX_PREFILL_SUBBATCH],
                    start_pos,
                    cache_slots[offset : offset + MAX_PREFILL_SUBBATCH],
                    full_logits=full_logits,
                    compute_logits=compute_logits,
                )
            )
        if len(groups) == 1:
            return groups[0]
        output_ids = self._merge_batch_values(groups, 0)
        logits = self._merge_batch_values(groups, 1)
        hidden = self._merge_batch_values(groups, 2)
        return output_ids, logits, hidden

    @staticmethod
    def _merge_batch_values(results, index):
        values = [result[index] for result in results]
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if len(tensors) != len(values):
            return values[-1]
        return torch.cat(tensors, dim=0)

    def _target_batch_group(
        self,
        token_batches,
        start_pos,
        cache_slots,
        *,
        full_logits=False,
        compute_logits=True,
    ):
        if full_logits and not compute_logits:
            raise ValueError("full logits require vocab projection")
        length = len(token_batches[0])
        limit_rows = getattr(self, "prefill_rows", DEFAULT_PREFILL_ROWS)
        chunk_size = max(1, limit_rows // len(token_batches))
        cache_slots_cpu = cache_slots.tolist()
        output_ids = logits = main_hidden = None
        logit_chunks = [] if full_logits else None
        for offset in range(0, length, chunk_size):
            inputs = self._input_tokens(
                [tokens[offset : offset + chunk_size] for tokens in token_batches]
            )
            output_ids, chunk_logits, chunk_hidden = self.model(
                inputs,
                start_pos=start_pos + offset,
                cache_slots=cache_slots,
                full_logits=full_logits,
                compute_logits=compute_logits,
                collect_main_hidden=getattr(self, "dspark", False),
                cache_slots_cpu=cache_slots_cpu,
            )
            logits = chunk_logits
            if logit_chunks is not None:
                logit_chunks.append(chunk_logits)
            if isinstance(chunk_hidden, torch.Tensor):
                self._seed_mtp_checkpoints(
                    chunk_hidden,
                    main_hidden,
                    start_pos + offset,
                    cache_slots,
                    cache_slots_cpu,
                    start_pos + length,
                )
                if isinstance(main_hidden, torch.Tensor):
                    main_hidden = torch.cat((main_hidden, chunk_hidden), dim=1)
                else:
                    main_hidden = chunk_hidden
                limit = getattr(
                    getattr(self, "args", None),
                    "window_size",
                    MAX_NATIVE_ROWS,
                )
                limit += DS_PARK_DRAFT_TOKENS if full_logits else 0
                main_hidden = main_hidden[:, -limit:].clone()
            elif main_hidden is None:
                main_hidden = chunk_hidden
            del chunk_hidden
        if logit_chunks is not None:
            logits = torch.cat(logit_chunks, dim=1)
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

    def _seed_mtp(self, main_hidden, start_pos, slot, *, source_length=None):
        if main_hidden is None:
            return False
        return self._seed_mtp_batch(
            main_hidden,
            start_pos,
            torch.tensor([slot], device=main_hidden.device),
            source_length=source_length,
        )

    def _mtp_source_length(self, length):
        window_size = getattr(
            getattr(self, "args", None),
            "window_size",
            MAX_NATIVE_ROWS,
        )
        return length if length > window_size else None

    def _seed_mtp_checkpoints(
        self, hidden, previous, start_pos, slots, slots_cpu, final_end
    ):
        if not getattr(self, "dspark", False) or not hasattr(self, "cache"):
            return
        ends = self.cache.mtp_checkpoint_ends(
            slots_cpu, start_pos, start_pos + hidden.size(1), final_end
        )
        width = self.args.window_size
        for end in ends:
            count = end - start_pos
            selected = hidden[:, max(0, count - width) : count]
            if selected.size(1) < width and isinstance(previous, torch.Tensor):
                selected = torch.cat(
                    (previous[:, -(width - selected.size(1)) :], selected), dim=1
                )
            self._seed_mtp_batch(selected, end - selected.size(1), slots)

    def _seed_mtp_batch(
        self,
        main_hidden,
        start_pos,
        cache_slots,
        *,
        source_length=None,
    ):
        if (
            not self.dspark
            or main_hidden is None
            or not hasattr(self, "model")
            or not hasattr(self.model, "forward_spec_seed")
        ):
            return False
        batch_size, hidden_length, _ = main_hidden.size()
        source_length = hidden_length if source_length is None else source_length
        if type(source_length) is not int or source_length < hidden_length:
            raise ValueError("DeepSeek MTP source length is shorter than hidden state")
        window_size = getattr(
            getattr(self, "args", None),
            "window_size",
            hidden_length,
        )
        export = getattr(getattr(self, "cache", None), "exports", None)
        if export is not None and export.steps:
            window_size += DS_PARK_DRAFT_TOKENS
        hidden_start = start_pos + source_length - hidden_length
        if hidden_length > window_size:
            main_hidden = main_hidden[:, -window_size:]
            hidden_start += hidden_length - window_size
            hidden_length = window_size
        chunk_size = max(1, MAX_NATIVE_ROWS // batch_size)
        for offset in range(0, hidden_length, chunk_size):
            self.model.forward_spec_seed(
                main_hidden[:, offset : offset + chunk_size],
                start_pos=hidden_start + offset,
                cache_slots=cache_slots,
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

    def _prepare_row(self, row):
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
        return request, slot, token_ids, start_pos

    def _execute_prefill_group(self, rows, *, final=False):
        prepared = []
        try:
            for row in rows:
                request, slot, token_ids, start_pos = self._prepare_row(row)
                prepared.append((row, request, slot, token_ids, start_pos))
                if hasattr(self, "cache"):
                    self.cache.begin_step(
                        row,
                        slot,
                        start_pos=start_pos,
                        end_pos=start_pos + len(token_ids),
                        minimum_end=start_pos + len(token_ids),
                    )
            start_pos = prepared[0][4]
            length = len(prepared[0][3])
            if any(item[4] != start_pos or len(item[3]) != length for item in prepared):
                raise ValueError("DeepSeek prefill batch rows are not aligned")
            cache_slots = torch.tensor(
                [item[2] for item in prepared],
                device=self.device,
                dtype=torch.long,
            )
            _, logits, main_hidden = self._target_batch(
                [item[3] for item in prepared],
                start_pos,
                cache_slots,
                compute_logits=final,
            )
            mtp_state_updated = self._seed_mtp_batch(
                main_hidden,
                start_pos,
                cache_slots,
                source_length=self._mtp_source_length(length),
            )
            outputs = {}
            accepted_end = start_pos + length
            for index, (row, request, slot, token_ids, _) in enumerate(prepared):
                tokens, text, proposals = [], "", []
                if final:
                    target = int(logits[index].argmax(dim=-1).item())
                    tokens, text = request.append([target])
                    if self.dspark and not request.finish:
                        proposals = self._proposal(
                            target,
                            main_hidden[index : index + 1, -1:],
                            accepted_end - 1,
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
                request.state_position = accepted_end
                self.trace.record(
                    "target",
                    verify=False,
                    execution="eager",
                    requests=[
                        {
                            "id": row["request_id"],
                            "start": start_pos,
                            "count": len(token_ids),
                            "prompt": request.prompt_len,
                        }
                    ],
                )
                outputs[row["request_id"]] = self._output(
                    request,
                    row,
                    tokens,
                    text,
                    0,
                    proposals,
                )
            return outputs
        except BaseException:
            for _row, request, _slot, _token_ids, _start_pos in prepared:
                self._free_slot(request)
                request.state_position = 0
            raise

    def _execute_row_impl(self, row):
        request = self.requests[row["request_id"]]
        _, slot, token_ids, start_pos = self._prepare_row(row)
        if not token_ids:
            return self._output(request, row, [], "")

        known_count = max(0, min(len(token_ids), len(request.history) - start_pos))
        final = start_pos + len(token_ids) >= len(request.history)
        accepted_end = start_pos
        mtp_state_updated = False
        if not final:
            if hasattr(self, "cache"):
                self.cache.begin_step(
                    row,
                    slot,
                    start_pos=start_pos,
                    end_pos=start_pos + len(token_ids),
                    minimum_end=start_pos + len(token_ids),
                )
            _, _, main_hidden = self._target(
                token_ids,
                start_pos,
                slot,
                compute_logits=False,
            )
            mtp_state_updated = self._seed_mtp(
                main_hidden,
                start_pos,
                slot,
                source_length=self._mtp_source_length(len(token_ids)),
            )
            accepted_end = start_pos + len(token_ids)
            tokens, text = [], ""
            accepted = 0
            proposals = []
        else:
            known = token_ids[:known_count]
            if not known:
                raise ValueError("DeepSeek execution row has no accepted input")
            drafts = token_ids[known_count:]
            if hasattr(self, "cache"):
                self.cache.begin_step(
                    row,
                    slot,
                    start_pos=start_pos,
                    end_pos=start_pos + len(token_ids),
                    minimum_end=start_pos + known_count,
                )
            if drafts:
                _, logits, main_hidden = self._target(
                    token_ids,
                    start_pos,
                    slot,
                    full_logits=True,
                    logit_start=known_count - 1,
                )
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 3
                    or logits.size(0) != 1
                    or logits.size(1) != len(drafts) + 1
                ):
                    raise ValueError(
                        "DeepSeek batched verification returned invalid logits"
                    )
                target_predictions = logits[0].argmax(dim=-1).tolist()
                verification = verify_greedy_drafts(drafts, target_predictions)
                emitted = list(verification.token_ids)
                tokens, text = request.append(emitted)
                accepted = min(
                    verification.accepted_draft_tokens,
                    max(0, len(tokens) - 1),
                )
                accepted_input_count = known_count + accepted
                hidden_start = start_pos
                hidden_count = accepted_input_count
                accepted_hidden = None
                if isinstance(main_hidden, torch.Tensor):
                    hidden_start = start_pos + len(token_ids) - main_hidden.size(1)
                    hidden_count = start_pos + accepted_input_count - hidden_start
                    if not 0 < hidden_count <= main_hidden.size(1):
                        raise ValueError(
                            "DeepSeek accepted hidden state is unavailable"
                        )
                    accepted_hidden = main_hidden[:, :hidden_count]
                mtp_state_updated = self._seed_mtp(
                    accepted_hidden,
                    hidden_start,
                    slot,
                    source_length=hidden_count,
                )
                accepted_end = start_pos + accepted_input_count
                proposals = []
            else:
                _, logits, main_hidden = self._target(known, start_pos, slot)
                mtp_state_updated = self._seed_mtp(
                    main_hidden,
                    start_pos,
                    slot,
                    source_length=self._mtp_source_length(len(known)),
                )
                target_predictions = [int(logits.argmax(dim=-1).item())]
                accepted_end = start_pos + known_count
                tokens, text = request.append(target_predictions)
                accepted = 0
                proposals = []
                if self.dspark and not request.finish:
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
        initial_prefill_groups = {}
        prefill_groups = {}
        for row in message["scheduled"]:
            request = self.requests[row["request_id"]]
            if row.get("prefill_token_ids") is not None:
                request.history = list(row["prefill_token_ids"])
            token_ids = list(row["token_ids"])
            start_pos = int(row["num_computed_tokens"])
            end_pos = start_pos + len(token_ids)
            if token_ids and end_pos == len(request.history):
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
        prefill_ids = set()
        for rows in initial_prefill_groups.values():
            outputs_by_id.update(self._execute_prefill_group(rows, final=True))
            prefill_ids.update(row["request_id"] for row in rows)
        for rows in prefill_groups.values():
            outputs_by_id.update(self._execute_prefill_group(rows))
            prefill_ids.update(row["request_id"] for row in rows)
        for row in message["scheduled"]:
            if row["request_id"] not in prefill_ids:
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
