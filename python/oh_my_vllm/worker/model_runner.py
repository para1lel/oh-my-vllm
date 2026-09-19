"""GPUWorker adapter. Scheduling and logical KV allocation remain in Rust."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


@dataclass
class ScheduledRequest:
    request_id: int
    token_ids: list[int]
    num_computed_tokens: int
    fa_block_table: list[int]
    mamba_block_table: list[int]


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    finished_request_ids: list[int] = field(default_factory=list)
    preempted_request_ids: list[int] = field(default_factory=list)
    num_batched_tokens: int = 0


@dataclass
class RequestOutput:
    request_id: int
    token_ids: list[int]
    num_accepted_draft_tokens: int = 0
    new_draft_token_ids: list[int] = field(default_factory=list)


@dataclass
class WorkerOutput:
    outputs: list[RequestOutput]


class SchedulerAdapter:
    """Translate full Rust block tables into vLLM's incremental request updates."""

    def __init__(self, group_kinds: list[str]):
        self.group_kinds = group_kinds
        self.stride = max(group_kinds.count(kind) for kind in set(group_kinds))
        counts = {}
        self.offsets = []
        for kind in group_kinds:
            self.offsets.append(counts.get(kind, 0))
            counts[kind] = counts.get(kind, 0) + 1
        self.blocks: dict[int, tuple[list[int], ...]] = {}
        self.output_counts: dict[int, int] = {}
        self.preempted: set[int] = set()

    def convert(self, output: SchedulerOutput, prompts: dict, sampling: dict):
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
        from vllm.v1.core.sched.output import SchedulerOutput as VllmOutput

        result = VllmOutput.make_empty()
        result.finished_req_ids = {str(rid) for rid in output.finished_request_ids}
        result.preempted_req_ids = {str(rid) for rid in output.preempted_request_ids}
        self.preempted.update(output.preempted_request_ids)
        for rid in output.finished_request_ids:
            self.blocks.pop(rid, None)
            self.output_counts.pop(rid, None)
            self.preempted.discard(rid)
        cached = CachedRequestData.make_empty()
        for request in output.scheduled:
            rid = request.request_id
            blocks = tuple(
                [
                    0 if block == 0 else block * self.stride + offset
                    for block in (
                        request.fa_block_table
                        if kind == "fa"
                        else request.mamba_block_table
                    )
                ]
                for kind, offset in zip(self.group_kinds, self.offsets, strict=True)
            )
            result.num_scheduled_tokens[str(rid)] = len(request.token_ids)
            if rid not in self.blocks:
                result.scheduled_new_reqs.append(
                    NewRequestData(
                        req_id=str(rid),
                        prompt_token_ids=prompts[rid],
                        mm_features=[],
                        sampling_params=sampling[rid],
                        pooling_params=None,
                        block_ids=blocks,
                        num_computed_tokens=request.num_computed_tokens,
                        lora_request=None,
                    )
                )
                self.output_counts[rid] = 0
            else:
                cached.req_ids.append(str(rid))
                cached.num_computed_tokens.append(request.num_computed_tokens)
                cached.num_output_tokens.append(self.output_counts[rid])
                if rid in self.preempted:
                    cached.resumed_req_ids.add(str(rid))
                    cached.new_block_ids.append(blocks)
                    self.preempted.remove(rid)
                else:
                    old = self.blocks[rid]
                    # Skipped Mamba entries become null but are never read again.
                    # Persistent batch tables need only the appended suffix.
                    for before, after in zip(old, blocks, strict=True):
                        if len(after) < len(before):
                            raise ValueError("block table shrank without preemption")
                    cached.new_block_ids.append(
                        tuple(
                            after[len(before) :]
                            for before, after in zip(old, blocks, strict=True)
                        )
                    )
            self.blocks[rid] = blocks
        result.scheduled_cached_reqs = cached
        result.total_num_scheduled_tokens = sum(result.num_scheduled_tokens.values())
        if result.total_num_scheduled_tokens != output.num_batched_tokens:
            raise ValueError("Rust token budget does not match scheduled tokens")
        result.num_common_prefix_blocks = [0] * len(self.group_kinds)
        return result


def with_config(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.config):
            return method(self, *args, **kwargs)

    return wrapped


class OhMyVllmWorker:
    def __init__(self, vllm_config: VllmConfig, num_speculative_tokens: int = 0):
        from vllm.utils.network_utils import get_open_port
        from vllm.v1.worker.gpu_worker import Worker

        self.config = vllm_config
        self._worker = Worker(
            vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            is_driver_worker=True,
        )
        self.sampling_params_map: dict = {}
        self.prompt_token_ids_map: dict = {}
        self.adapter: SchedulerAdapter | None = None
        self._num_speculative_tokens = num_speculative_tokens

    @with_config
    def init_device(self):
        self._worker.init_device()

    @with_config
    def load_model(self):
        from vllm.platforms import current_platform

        self._worker.load_model()
        # Match the executor's post-load setup before obtaining cache specs.
        # This also sets Mamba block size and page padding, not just FA size.
        current_platform.update_block_size_for_backend(self.config)
        if self.config.cache_config.block_size != 784:
            raise ValueError("Backend changed the required block size of 784")

    @with_config
    def initialize_cache(self, num_gpu_blocks: int):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind

        specs = self._worker.get_kv_cache_spec()
        available = self._worker.determine_available_memory()
        cache = get_kv_cache_configs(self.config, [specs], [available])[0]
        if cache.num_blocks != num_gpu_blocks:
            raise ValueError(
                f"KV capacity mismatch: Rust={num_gpu_blocks}, "
                f"worker={cache.num_blocks}"
            )
        kinds = []
        for group in cache.kv_cache_groups:
            spec = group.kv_cache_spec
            if spec.block_size != 784:
                raise ValueError(
                    f"Physical KV group has incompatible block size: {spec}"
                )
            if get_kv_cache_spec_kind(spec) == KVCacheSpecKind.FULL_ATTENTION:
                kinds.append("fa")
            elif get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA:
                kinds.append("mamba")
            else:
                raise TypeError(f"Unsupported KV group: {spec}")
        if set(kinds) != {"fa", "mamba"}:
            raise ValueError(f"Expected FA and Mamba groups, got {kinds}")
        logger.info("KV physical groups: %s; blocks=%s", kinds, cache.num_blocks)
        self.adapter = SchedulerAdapter(kinds)
        self.logical_num_blocks = cache.num_blocks // self.adapter.stride
        self._worker.initialize_from_config(cache)
        self._worker.compile_or_warm_up_model()

    @with_config
    def shutdown(self):
        from vllm.distributed import cleanup_dist_env_and_memory

        self._worker.shutdown()
        cleanup_dist_env_and_memory()

    def register_request(
        self, request_id: int, prompt_token_ids: list[int], sampling_params=None
    ):
        from vllm.sampling_params import SamplingParams

        if request_id in self.prompt_token_ids_map:
            raise ValueError(f"Duplicate request {request_id}")
        self.prompt_token_ids_map[request_id] = prompt_token_ids
        self.sampling_params_map[request_id] = sampling_params or SamplingParams(
            temperature=0,
            max_tokens=self.config.model_config.max_model_len,
            ignore_eos=True,
        )

    def unregister_request(self, request_id: int):
        self.prompt_token_ids_map.pop(request_id, None)
        self.sampling_params_map.pop(request_id, None)

    @with_config
    def execute_model(self, rust_output: SchedulerOutput) -> WorkerOutput:
        if self.adapter is None:
            raise RuntimeError("KV cache is not initialized")
        vllm_output = self.adapter.convert(
            rust_output,
            self.prompt_token_ids_map,
            self.sampling_params_map,
        )
        output = self._worker.execute_model(vllm_output)
        if rust_output.scheduled and output is None:
            output = self._worker.sample_tokens(None)
        if output is not None and hasattr(output, "get_output"):
            output = output.get_output()
        sampled = (
            {}
            if output is None
            else dict(
                zip(
                    output.req_ids,
                    output.sampled_token_ids,
                    strict=True,
                )
            )
        )
        results = []
        for request in rust_output.scheduled:
            rid = request.request_id
            if str(rid) not in sampled:
                raise RuntimeError(f"Worker omitted scheduled request {rid}")
            tokens = sampled[str(rid)]
            self.adapter.output_counts[rid] += len(tokens)
            results.append(RequestOutput(rid, tokens, max(0, len(tokens) - 1)))
        for rid in rust_output.finished_request_ids:
            self.unregister_request(rid)
        return WorkerOutput(results)
