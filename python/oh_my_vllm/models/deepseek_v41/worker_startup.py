import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ...worker.trace import Trace
from .adapter import (
    CONFIG_SHA256,
    INFERENCE_CONFIG_SHA256,
    MAX_CONTEXT_TOKENS,
    validate_config,
)
from .cache import DeepSeekCache
from .decode import (
    decode_cycle,
)
from .descriptor import build_descriptor
from .loader import DEFAULT_TILE_ROWS, RankLoader
from .model import ModelArgs, Transformer
from .startup_trace import _StartupTrace
from .tokenizer import Tokenization
from .trace import execution_modes, runtime_identity
from .worker_limits import DEFAULT_CACHE_CONTEXT, prefill_row_limit


class WorkerStartup:
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
        model_strategy = strategy
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
        self._load_model(config, parsed, args, model_strategy)
        self._configure_execution(config, strategy, model_strategy)
        self._initialize_cache(config, args, parsed, raw_config, model_strategy)
        self._initialize_trace(config, args, strategy, model_strategy)
        self.startup_trace.record("ready")

    def _load_model(self, config, parsed, args, model_strategy):
        root = self.root
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

    def _configure_execution(self, config, strategy, model_strategy):
        if strategy == "dep4":
            from .dep4 import Dep4Executor

            self.dep4_executor = Dep4Executor(self.model)
            self.target_call_id_provider = lambda: self.dep4_executor.call
        self.decode_cycle = decode_cycle(self.model)
        modes = execution_modes(model_strategy)
        self.decode_graph_enabled = modes["decode_graph"]
        self.decode_graph_limit = modes["decode_graph_limit"]
        self.proposal_graph_enabled = modes["proposal_graph"]
        self.proposal_graphs = {}
        self.decode_graphs = {}
        self.decode_bindings = {}
        self.decode_engram = {}
        self.batch_decode_enabled = modes["batch_decode"]
        self.routing_trace_enabled = bool(os.environ.get("OH_MY_VLLM_EXECUTION_TRACE"))
        self.target_call = 0
        self.requests = {}
        self.slots = set(range(config["max_num_seqs"]))

    def _initialize_cache(self, config, args, parsed, raw_config, model_strategy):
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
                release_request_slot=self._free_slot,
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

    def _initialize_trace(self, config, args, strategy, model_strategy):
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
                    "runtime": (
                        runtime_identity(
                            self.rank,
                            strategy=self.strategy,
                            batch_decode=self.batch_decode_enabled,
                            decode_graph=self.decode_graph_enabled,
                            decode_graph_limit=self.decode_graph_limit,
                            proposal_graph=self.proposal_graph_enabled,
                        )
                        if os.environ.get("OH_MY_VLLM_EXECUTION_TRACE")
                        else None
                    ),
                },
            )
        if strategy == "dep4":
            if self.routing_trace_enabled:
                from .dep4_trace import Dep4Trace

                self.trace = Dep4Trace(self, self.trace)
                self.dep4_executor.accounting = self.trace
            self.dep4_executor.record_round = self.trace.record
