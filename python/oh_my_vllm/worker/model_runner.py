"""Project-owned single-device GPU execution; Rust retains cache allocations."""

import gc
import logging
import math
import os
from dataclasses import dataclass

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import Qwen
from oh_my_vllm.performance.execution import certify_feedback, record, tracing
from oh_my_vllm.worker.batch_plan import (
    BLOCK,
    PlannedRequest,
    plan_request,
    validate_batch,
)
from oh_my_vllm.worker.errors import (
    WorkerUnavailable,
    failure_kind,
    resource_operation,
    transactional_step,
)
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.graph_capture import validate_graph_allocator
from oh_my_vllm.worker.protocol import RequestOutput, SchedulerOutput, WorkerOutput
from oh_my_vllm.worker.runtime import identity, verify_loaded_modules
from oh_my_vllm.worker.sampler import RequestSampler, greedy_rows, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams
from oh_my_vllm.worker.serving import ServingAdapter
from oh_my_vllm.worker.target_execution import run_target
from oh_my_vllm.worker.tensors import (
    device_tensor,
)

logger = logging.getLogger(__name__)


def _is_device_failure(error: Exception) -> bool:
    return failure_kind(error) in ("oom", "device")


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    max_model_len: int = 65536
    num_gpu_blocks: int = 1024
    speculative_tokens: int = 0
    mamba_blocks: int | None = None
    speculative_mode: str | None = None
    draft_model: str | None = None
    dspark_confidence_threshold: float = 0.2

    def __post_init__(self) -> None:
        mode = self.speculative_mode or ("mtp" if self.speculative_tokens else "none")
        if mode not in ("none", "mtp", "dspark"):
            raise ValueError("unknown speculative mode")
        expected = {"none": 0, "mtp": 4, "dspark": 7}[mode]
        if self.speculative_tokens != expected:
            raise ValueError("speculative token count disagrees with mode")
        if mode == "dspark" and not self.draft_model:
            raise ValueError("DSpark requires a draft checkpoint")
        if not math.isfinite(self.dspark_confidence_threshold) or not (
            0 <= self.dspark_confidence_threshold < 1
        ):
            raise ValueError("DSpark confidence threshold must be in [0,1)")
        object.__setattr__(self, "speculative_mode", mode)

    def target_graph_budget(self) -> tuple[int, dict[str, int]]:
        """Bound target graph budgets independently from physical model caches.

        Adaptive DSpark uses more token/query shapes than fixed-count MTP.
        All budgets retain the graph cache's device-memory headroom checks.
        """
        return (64 if self.speculative_mode == "dspark" else 32), {}

    def target_graph_cooldown(self) -> int:
        """Keep resident DSpark graphs while repeated shape churn cools down.

        A recent recapture starts the same bounded decision interval used by
        MTP's draft family. Existing shapes still replay; other shapes use the
        compiled target path until the interval ends.
        """
        return 32768 if self.speculative_mode == "dspark" else 0

    def decode_graph_key(self, token_counts: list[int], extent: int) -> tuple:
        """Include the per-request query bound in the captured attention shape.

        DSpark confidence stopping changes each request's verification length.
        Equal total rows can still require different Torch/CUDA query bounds.
        Target GQA has six query heads per KV head. Bounds 1/2/5/8 keep groups
        within the native 8/16/32/64-row tiles and limit captured shape variety.
        Ordinary and MTP keep their existing fixed bounds.
        """
        bound = (
            next(value for value in (1, 2, 5, 8) if value >= max(token_counts))
            if self.speculative_mode == "dspark"
            else max(1, self.speculative_tokens + 1)
        )
        return sum(token_counts), len(token_counts), extent, bound


class OhMyVllmWorker:
    def __init__(self, config: RuntimeConfig):
        from oh_my_vllm.ir.execution import execution_policy

        self.execution_policy = execution_policy()
        if config.num_gpu_blocks < 6 or config.max_model_len <= 0:
            raise ValueError("invalid runtime capacity")
        self.config = config
        # Preserve the original scheduler capacity, without vLLM's physical padding.
        self.logical_num_blocks = config.num_gpu_blocks // 3
        self.mamba_blocks = (
            self.logical_num_blocks
            if config.mamba_blocks is None
            else config.mamba_blocks
        )
        if self.mamba_blocks < 2:
            raise ValueError("Mamba capacity needs a null and a writable slot")
        self.histories: dict[int, list[int]] = {}
        self.samplers: dict[int, RequestSampler] = {}
        self.sources: dict[int, int] = {}
        self.computed: dict[int, int] = {}
        self.serving = None
        self.forward_unit = None
        self.logits_unit = None
        self.feature_unit = None
        self.dspark = None
        self._failure = None

    def _is_dspark(self) -> bool:
        return (
            getattr(getattr(self, "config", None), "speculative_mode", None) == "dspark"
        )

    def _proposer(self):
        return getattr(self, "dspark", None) if self._is_dspark() else self.mtp

    def init_device(self) -> None:
        if os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1":
            validate_graph_allocator()
        logger.info("Independent runtime", extra={"fields": identity()})
        torch.cuda.set_device(0)

    @torch.inference_mode()
    def load_model(self) -> None:
        has_dspark = self._is_dspark()
        self.model = Qwen(
            self.config.model,
            mtp=self.config.speculative_mode == "mtp",
            feature_layer_ids=(5, 19, 33, 47, 61) if has_dspark else (),
        )
        if os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") == "1":
            self.forward_unit = self.model.forward
            self.logits_unit = self.model.logits
            self.feature_unit = self.model.forward_features if has_dspark else None
        else:
            self.forward_unit = compile_forward(
                self.model.forward, unit="model_forward"
            )
            self.logits_unit = compile_forward(self.model.logits, unit="model_logits")
            if has_dspark:
                self.feature_unit = compile_forward(
                    self.model.forward_features, unit="model_features"
                )

    @torch.inference_mode()
    def initialize_cache(self) -> None:
        self.caches = []
        for kind in self.model.kinds:
            if kind == "full_attention":
                cache = torch.zeros(
                    self.logical_num_blocks,
                    2,
                    BLOCK,
                    4,
                    256,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            else:
                cache = (
                    torch.zeros(
                        self.mamba_blocks,
                        10240,
                        3,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    torch.zeros(
                        self.mamba_blocks,
                        48,
                        128,
                        128,
                        dtype=torch.bfloat16
                        if self.config.speculative_tokens
                        else torch.float32,
                        device="cuda",
                    ),
                )
            self.caches.append(cache)
        self.attention = PagedAttention()
        capacity, floors = self.config.target_graph_budget()
        self.graph_cache = GraphCache(
            capacity=capacity,
            family_floors=floors,
            churn_cooldown_decisions=self.config.target_graph_cooldown(),
            free_bytes=lambda: torch.cuda.mem_get_info()[0],
            reserved_bytes=lambda: torch.cuda.memory_reserved(),
            release_memory=torch.cuda.empty_cache,
            release_family=self._release_graph_pool,
        )
        # Target outputs feed MTP before the next target replay. Keep target
        # capture scratch separate from the draft and proposal graph families.
        self.graph_pool = None
        self.dspark_graph_pool = None
        self.prefill_graph_pool = None
        self.prefill_dspark_graph_pool = None
        # Large activation captures have a separate, small resident budget and
        # a larger headroom requirement than decode/verification captures.
        self.prefill_graph_cache = GraphCache(
            capacity=8,
            free_bytes=lambda: torch.cuda.mem_get_info()[0],
            reserved_bytes=lambda: torch.cuda.memory_reserved(),
            # Ragged KV, feature outputs and restoration buffers coexist
            # during capture. Keep headroom for the complete transient peak.
            min_capture_free_bytes=32 << 30,
            release_memory=torch.cuda.empty_cache,
            release_family=self._release_graph_pool,
        )
        from oh_my_vllm.worker.mtp import MTP

        self.mtp = (
            MTP(
                self.model,
                self.logical_num_blocks,
                self.config.max_model_len,
            )
            if self.config.speculative_mode == "mtp"
            else None
        )
        if self._is_dspark():
            from oh_my_vllm.worker.dspark import DSpark

            self.dspark = DSpark(
                self.model,
                self.config.draft_model,
                self.logical_num_blocks,
                self.config.max_model_len,
                confidence_threshold=self.config.dspark_confidence_threshold,
            )

    def cache_capacities(self) -> tuple[int, int]:
        """Return the FA and GDN slot counts of the allocated device tensors."""
        fa = {
            cache.shape[0]
            for kind, cache in zip(self.model.kinds, self.caches, strict=True)
            if kind == "full_attention"
        }
        gdn = {
            state.shape[0]
            for kind, cache in zip(self.model.kinds, self.caches, strict=True)
            if kind != "full_attention"
            for state in cache
        }
        if len(fa) != 1 or len(gdn) != 1:
            raise RuntimeError("inconsistent allocated FA or GDN cache capacities")
        return fa.pop(), gdn.pop()

    @resource_operation
    def register_request(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> None:
        if getattr(self, "_failure", None) is not None:
            raise WorkerUnavailable(self._failure)
        if request_id in self.histories:
            raise ValueError("duplicate request")
        if not prompt_token_ids or any(not 0 <= t < 248320 for t in prompt_token_ids):
            raise ValueError("invalid prompt tokens")
        params = sampling_params or SamplingParams(
            self.config.max_model_len,
            temperature=0,
            ignore_eos=True,
            repetition_guard=False,
        )
        sampler = RequestSampler(params, prompt_token_ids, "cuda", vocab_size=248320)
        self.histories[request_id] = list(prompt_token_ids)
        self.samplers[request_id] = sampler

    def prepare_request(self, request_id: int, request: dict) -> list[int]:
        if getattr(self, "_failure", None) is not None:
            raise WorkerUnavailable(self._failure)
        if request_id in self.histories:
            raise ValueError("duplicate request")
        if self.serving is None:
            self.serving = ServingAdapter(
                self.config.model, 248320, self.config.max_model_len
            )
        ids, params = self.serving.prepare(request_id, request)
        try:
            self.register_request(request_id, ids, params)
        except Exception:
            self.serving.generations.pop(request_id, None)
            raise
        return ids

    def unregister_request(self, request_id: int) -> None:
        if self.mtp is not None:
            self.mtp.forget(request_id)
        if getattr(self, "dspark", None) is not None:
            self.dspark.forget(request_id)
        for mapping in (self.histories, self.samplers, self.sources, self.computed):
            mapping.pop(request_id, None)
        if self.serving is not None:
            self.serving.generations.pop(request_id, None)

    @transactional_step
    @torch.inference_mode()
    def execute_model(self, scheduled: SchedulerOutput) -> WorkerOutput:
        plans, registration_errors = self._plan_scheduled_requests(scheduled)
        if not plans:
            return WorkerOutput(registration_errors)
        if tracing():
            record(
                "target",
                requests=[
                    {
                        "request_id": p.request.request_id,
                        "query": len(p.writes),
                        "unique_token_ids": sorted(set(p.request.token_ids)),
                        "fa_block_table": p.request.fa_block_table,
                        "context": p.request.num_computed_tokens,
                        "prefill": p.prefill,
                        "drafts": len(p.drafts),
                        "sample_rows": len(p.sample_indices),
                        "plain_greedy": self.samplers[
                            p.request.request_id
                        ].plain_greedy,
                        "state_source": p.source,
                        "state_destinations": [w for w in p.writes if w > 0],
                        "state_row_writes": p.writes,
                        "mamba_block_table": p.request.mamba_block_table,
                    }
                    for p in plans
                ],
                state_dtype="torch.bfloat16"
                if self.config.speculative_tokens
                else "torch.float32",
            )
        target = run_target(self, plans)
        hidden, features, logits = target.hidden, target.features, target.logits
        starts = target.request_starts
        masks, mask_errors = self._build_masks(plans)
        greedy = None
        if (
            logits is not None
            and not masks
            and all(
                self.samplers.get(p.request.request_id) is not None
                and self.samplers[p.request.request_id].plain_greedy
                for p in plans
                if p.sample_indices and p.request.request_id not in mask_errors
            )
        ):
            greedy = greedy_rows(logits)
        (
            results,
            copies,
            successful_plans,
            successful_starts,
            successful_counts,
            successful_outputs,
        ) = self._commit_plans(plans, starts, logits, greedy, masks, mask_errors)
        if tracing():
            committed_counts = {
                p.request.request_id: count
                for p, count in zip(successful_plans, successful_counts, strict=True)
            }
            record(
                "commit",
                copies=copies,
                outputs=[
                    {
                        "request_id": o.request_id,
                        "kept": len(o.token_ids),
                        "computed": committed_counts.get(o.request_id, 0),
                        "accepted": o.num_accepted_draft_tokens,
                        "finished": o.finish_reason is not None,
                        "state_source": self.sources.get(o.request_id),
                    }
                    for o in results
                ],
            )
        if copies:
            sources, destinations = (
                device_tensor(values, device="cuda")
                for values in zip(*copies, strict=True)
            )
            for kind, cache in zip(self.model.kinds, self.caches, strict=True):
                if kind == "linear_attention":
                    for pool in cache:
                        pool.index_copy_(0, destinations, pool.index_select(0, sources))
        proposer = self._proposer()
        if proposer is not None and successful_plans:
            draft_args = (
                successful_plans,
                successful_starts,
                successful_counts,
                features if self._is_dspark() else hidden,
                self.histories,
                successful_outputs,
            )
            drafts = (
                proposer.propose(
                    *draft_args, samplers=self.samplers, serving=self.serving
                )
                if self._is_dspark()
                else proposer.propose(*draft_args, samplers=self.samplers)
            )
            for output in successful_outputs:
                output.new_draft_token_ids = drafts[output.request_id]
        return WorkerOutput(registration_errors + results)

    def _plan_scheduled_requests(
        self, scheduled: SchedulerOutput
    ) -> tuple[list[PlannedRequest], list[RequestOutput]]:
        for rid in scheduled.finished_request_ids:
            self.unregister_request(rid)
        for rid in scheduled.preempted_request_ids:
            if self.mtp is not None:
                self.mtp.forget(rid)
            if getattr(self, "dspark", None) is not None:
                self.dspark.forget(rid)
            self.sources.pop(rid, None)
            self.computed.pop(rid, None)
        if scheduled.num_batched_tokens != sum(
            len(r.token_ids) for r in scheduled.scheduled
        ):
            raise ValueError("Rust token budget disagrees with scheduled input")
        plans = []
        registration_errors = []
        for request in scheduled.scheduled:
            rid = request.request_id
            if rid not in self.histories:
                if self.serving is None:
                    raise ValueError(f"request {rid} has no registered prompt")
                registration_errors.append(
                    RequestOutput(rid, [], error="request registration failed")
                )
                continue
            if (
                rid in self.computed
                and self.computed[rid] != request.num_computed_tokens
            ):
                raise ValueError(
                    "Rust computed position disagrees with committed state"
                )
            if (
                request.num_computed_tokens + len(request.token_ids)
                > self.config.max_model_len
            ):
                raise ValueError("scheduled input exceeds the model context")
            if any(
                p <= 0 or p >= self.logical_num_blocks for p in request.fa_block_table
            ):
                raise ValueError("FA table exceeds page capacity")
            if any(p < 0 or p >= self.mamba_blocks for p in request.mamba_block_table):
                raise ValueError("Mamba table exceeds state capacity")
            if rid in self.sources and not 0 <= self.sources[rid] < self.mamba_blocks:
                raise ValueError("recurrent source exceeds state capacity")
            plans.append(
                plan_request(
                    request,
                    self.histories[rid],
                    self.sources.get(rid),
                    self.logical_num_blocks,
                    self.mamba_blocks,
                    self.config.speculative_tokens,
                )
            )
        if plans:
            validate_batch(plans)
            plans.sort(key=lambda p: not p.prefill)
        return plans, registration_errors

    def _build_masks(self, plans):
        """Generate grammar rows per request so one invalid grammar is isolated."""
        masks, errors = {}, {}
        if self.serving is None:
            return masks, errors
        for plan in plans:
            if not plan.sample_indices:
                continue
            rid = plan.request.request_id
            try:
                output = self.serving.masks({rid: plan.drafts})
                if output is not None:
                    # ServingAdapter reuses its bitmask buffer on the next call.
                    masks[rid] = output.grammar_bitmask[
                        : len(plan.sample_indices)
                    ].copy()
            except Exception as exc:
                if _is_device_failure(exc):
                    raise
                errors[rid] = str(exc)
        return masks, errors

    def _commit_plans(self, plans, starts, logits, greedy, masks, mask_errors=None):
        """Commit each request independently after the shared GPU forward pass."""
        results, copies, row = [], [], 0
        sampled, sampling_errors = self._draw_batch_rows(
            plans, logits, greedy, masks, mask_errors
        )
        successful_plans, successful_starts, successful_counts, successful_outputs = (
            [],
            [],
            [],
            [],
        )
        for index, plan in enumerate(plans):
            rid = plan.request.request_id
            try:
                if mask_errors and rid in mask_errors:
                    row += len(plan.sample_indices)
                    raise ValueError(mask_errors[rid])
                tokens = []
                if plan.sample_indices:
                    end = row + len(plan.sample_indices)
                    sample_start = row
                    row = end
                    if greedy is not None:
                        tokens = verify_rows(greedy[sample_start:end], plan.drafts)
                    else:
                        if index in sampling_errors:
                            raise sampling_errors[index]
                        tokens = verify_rows(sampled[index], plan.drafts)
                text, finish, reasoning = "", None, 0
                if self.serving is not None and rid in self.serving.generations:
                    generation = self.serving.generations[rid]
                    tokens, text = generation.consume(tokens, self.serving.tokenizer)
                    finish, reasoning = generation.finished, generation.reasoning_tokens
                count, source, checkpoints = plan.commit(len(tokens))
                proposer = self._proposer()
                if proposer is not None:
                    proposer.validate_state(plan, count)
                self.sources[rid] = source
                self.computed[rid] = plan.request.num_computed_tokens + count
                self.samplers[rid].commit(tokens)
                self.histories[rid].extend(tokens)
                output = RequestOutput(
                    rid,
                    tokens,
                    num_accepted_draft_tokens=max(0, len(tokens) - 1),
                    text=text,
                    finish_reason=finish,
                    reasoning_tokens=reasoning,
                )
                successful_plans.append(plan)
                successful_starts.append(starts[index])
                successful_counts.append(count)
                successful_outputs.append(output)
                results.append(output)
                copies.extend(checkpoints)
            except Exception as exc:
                if self.serving is None or _is_device_failure(exc):
                    raise
                logger.exception("request %s failed during step commit", rid)
                self.unregister_request(rid)
                results.append(RequestOutput(rid, [], error=str(exc)))
        return (
            results,
            copies,
            successful_plans,
            successful_starts,
            successful_counts,
            successful_outputs,
        )

    def _draw_batch_rows(self, plans, logits, greedy, masks, mask_errors):
        """Read all non-greedy request draws back in one transfer.

        Sampling transforms may independently synchronize, such as the exact
        top-k/top-p cutoff-tie check.
        """
        if greedy is not None:
            return {}, {}
        drawn, errors = [], {}
        offset = 0
        for index, plan in enumerate(plans):
            count = len(plan.sample_indices)
            rid = plan.request.request_id
            if count and (not mask_errors or rid not in mask_errors):
                try:
                    sampler = self.samplers[rid]
                    if (
                        self._is_dspark()
                        and plan.drafts
                        and sampler.params.temperature > 0
                    ):
                        rows = sampler.draw_speculative_rows(
                            logits[offset : offset + count],
                            plan.drafts,
                            self.dspark.draft_probabilities(rid, plan.drafts),
                            masks.get(rid),
                        )
                    else:
                        rows = sampler.draw_rows(
                            logits[offset : offset + count],
                            drafts=plan.drafts,
                            bitmask=masks.get(rid),
                        )
                    drawn.append((index, rows, count))
                except Exception as exc:
                    if self.serving is None or _is_device_failure(exc):
                        raise
                    errors[index] = exc
            offset += count
        if not drawn:
            return {}, errors
        device_rows = (
            torch.cat([rows for _, rows, _ in drawn]) if len(drawn) > 1 else drawn[0][1]
        )
        host_rows = device_rows.tolist()
        certify_feedback()
        sampled = {}
        offset = 0
        for index, _, count in drawn:
            sampled[index] = host_rows[offset : offset + count]
            offset += count
        return sampled, errors

    def _release_graph_pool(self, family: str) -> None:
        attribute = {
            "target": "graph_pool",
            "target_dspark": "dspark_graph_pool",
            "prefill": "prefill_graph_pool",
            "prefill_dspark": "prefill_dspark_graph_pool",
        }[family]
        setattr(self, attribute, None)

    def resource_snapshot(self) -> dict:
        """Sample live resources only at an explicit, synchronized boundary."""
        if getattr(self, "_failure", None) is not None:
            raise WorkerUnavailable(self._failure)
        from oh_my_vllm.ir.core import compiler_snapshot

        torch.cuda.synchronize()
        graph_caches = {
            "target": self.graph_cache,
            "prefill": self.prefill_graph_cache,
        }
        for name in ("mtp", "dspark"):
            proposer = getattr(self, name, None)
            if proposer is not None:
                graph_caches[name] = proposer.graph_cache
                graph_caches[f"{name}_context"] = proposer.context_graph_cache
        requests = {
            name: len(getattr(self, name))
            for name in ("histories", "samplers", "sources", "computed")
        }
        requests["generations"] = (
            len(self.serving.generations) if self.serving is not None else 0
        )
        for name in ("mtp", "dspark"):
            proposer = getattr(self, name, None)
            requests[f"{name}_next_position"] = len(
                getattr(proposer, "next_position", {})
            )
        for name in ("proposals", "generators"):
            requests[f"dspark_{name}"] = len(getattr(self.dspark, name, {}))
        free, total = torch.cuda.mem_get_info()
        with open("/proc/self/statm", encoding="ascii") as status:
            rss = int(status.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        cpu = {"rss_bytes": rss}
        from oh_my_vllm.ir.executable_diagnostics import (
            EXECUTABLE_DIAGNOSTICS,
        )
        from oh_my_vllm.ir.executable_diagnostics import (
            enabled as executable_diagnostics_enabled,
        )

        if executable_diagnostics_enabled():
            families = {}
            for name, cache in graph_caches.items():
                for (family, _), graph in cache.graphs.items():
                    families.setdefault(f"{name}.{family}", []).append(graph)
            units = {
                name: getattr(self, name, None)
                for name in ("forward_unit", "logits_unit", "feature_unit")
            }
            for name, attributes in (
                ("mtp", ("draft_unit", "context_unit", "logits_unit")),
                (
                    "dspark",
                    ("inject_unit", "backbone_unit", "greedy_unit", "step_unit"),
                ),
            ):
                proposer = getattr(self, name, None)
                if proposer is not None:
                    units.update(
                        (f"{name}.{attribute}", getattr(proposer, attribute, None))
                        for attribute in attributes
                    )
            cpu["compiled_executables"] = EXECUTABLE_DIAGNOSTICS.snapshot(
                families, units
            )
        if os.environ.get("OH_MY_VLLM_HOST_ALLOCATOR_DIAGNOSTICS") == "1":
            from oh_my_vllm.worker.host_allocator import host_allocator_snapshot

            cpu["host_allocator"] = host_allocator_snapshot()
        return {
            "requests": requests,
            "graphs": {name: cache.snapshot() for name, cache in graph_caches.items()},
            "compile": compiler_snapshot(),
            "gpu": {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_bytes": free,
                "total_bytes": total,
            },
            "cpu": cpu,
        }

    def _release_resources(self) -> None:
        """Drop every callable, graph, state, and pool owner after device drain."""
        from oh_my_vllm.ir.core import clear_compiler_caches

        self.histories.clear()
        self.samplers.clear()
        self.sources.clear()
        self.computed.clear()
        if self.serving is not None:
            self.serving.generations.clear()
        self.graph_cache.clear()
        self.prefill_graph_cache.clear()
        for name in ("mtp", "dspark"):
            proposer = getattr(self, name, None)
            if proposer is not None:
                proposer.release()
        self.forward_unit = None
        self.logits_unit = None
        self.feature_unit = None
        for attribute in (
            "graph_pool",
            "dspark_graph_pool",
            "prefill_graph_pool",
            "prefill_dspark_graph_pool",
        ):
            setattr(self, attribute, None)
        self.caches = []
        self.model = None
        self.attention = None
        self.serving = None
        self.mtp = None
        self.dspark = None
        clear_compiler_caches()
        from oh_my_vllm.ir.execution import branch_stream

        branch_stream.cache_clear()
        gc.collect()

    def shutdown(self) -> None:
        if getattr(self, "_failure", None) is not None:
            raise WorkerUnavailable(self._failure)
        torch.cuda.synchronize()
        logger.info(
            "Target graph cache",
            extra={"fields": self.graph_cache.snapshot()},
        )
        logger.info(
            "Prefill graph cache",
            extra={"fields": self.prefill_graph_cache.snapshot()},
        )
        if self.mtp is not None:
            logger.info(
                "MTP graph cache",
                extra={"fields": self.mtp.graph_cache.snapshot()},
            )
            logger.info(
                "MTP context graph cache",
                extra={"fields": self.mtp.context_graph_cache.snapshot()},
            )
        if self.dspark is not None:
            logger.info(
                "DSpark graph cache",
                extra={"fields": self.dspark.graph_cache.snapshot()},
            )
            logger.info(
                "DSpark context graph cache",
                extra={"fields": self.dspark.context_graph_cache.snapshot()},
            )
        logger.info(
            "GPU memory high water",
            extra={
                "fields": {
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            },
        )
        verify_loaded_modules()
        logger.info("Independent runtime module/library audit passed")
        self._release_resources()
        # The bridge exits after shutdown. Process exit releases allocator storage
        # without a second device-wide allocator flush on this terminal path.
