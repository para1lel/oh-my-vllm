"""Project-owned single-device GPU execution; Rust retains cache allocations."""

import logging
import os
from dataclasses import dataclass

import torch

from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import Batch, Qwen
from oh_my_vllm.worker.batch_plan import BLOCK, plan_request, validate_batch
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.protocol import RequestOutput, SchedulerOutput, WorkerOutput
from oh_my_vllm.worker.runtime import identity, verify_loaded_modules
from oh_my_vllm.worker.sampler import RequestSampler, greedy_rows, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams
from oh_my_vllm.worker.serving import ServingAdapter
from oh_my_vllm.worker.tensors import (
    device_page_tables,
    device_tensor,
    device_vectors,
)

logger = logging.getLogger(__name__)


def _is_device_failure(error: Exception) -> bool:
    if isinstance(error, (torch.cuda.OutOfMemoryError, torch.AcceleratorError)):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "device-side assert",
            "illegal memory",
            "launch failure",
        )
    )


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    max_model_len: int = 65536
    num_gpu_blocks: int = 1024
    speculative_tokens: int = 0
    mamba_blocks: int | None = None


class OhMyVllmWorker:
    def __init__(self, config: RuntimeConfig):
        if config.speculative_tokens not in (0, 4):
            raise ValueError("independent runtime supports ordinary or MTP4")
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

    def init_device(self) -> None:
        logger.info("Independent runtime", extra={"fields": identity()})
        torch.cuda.set_device(0)

    @torch.inference_mode()
    def load_model(self) -> None:
        self.model = Qwen(self.config.model, mtp=bool(self.config.speculative_tokens))

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
        self.graph_cache = GraphCache(
            free_bytes=lambda: torch.cuda.mem_get_info()[0],
            reserved_bytes=lambda: torch.cuda.memory_reserved(),
        )
        # Target outputs feed MTP before the next target replay. Keep target
        # capture scratch separate from the draft and proposal graph families.
        self.graph_pool = None
        from oh_my_vllm.worker.mtp import MTP

        self.mtp = (
            MTP(self.model, self.logical_num_blocks, self.config.max_model_len)
            if self.config.speculative_tokens
            else None
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

    def register_request(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> None:
        if request_id in self.histories:
            raise ValueError("duplicate request")
        if not prompt_token_ids or any(not 0 <= t < 248320 for t in prompt_token_ids):
            raise ValueError("invalid prompt tokens")
        params = sampling_params or SamplingParams(
            self.config.max_model_len, temperature=0, ignore_eos=True
        )
        sampler = RequestSampler(params, prompt_token_ids, "cuda", vocab_size=248320)
        self.histories[request_id] = list(prompt_token_ids)
        self.samplers[request_id] = sampler

    def prepare_request(self, request_id: int, request: dict) -> list[int]:
        if self.serving is None:
            self.serving = ServingAdapter(
                self.config.model, 248320, self.config.max_model_len
            )
        ids, params = self.serving.prepare(request_id, request)
        self.register_request(request_id, ids, params)
        return ids

    def unregister_request(self, request_id: int) -> None:
        if self.mtp is not None:
            self.mtp.forget(request_id)
        for mapping in (self.histories, self.samplers, self.sources, self.computed):
            mapping.pop(request_id, None)
        if self.serving is not None:
            self.serving.generations.pop(request_id, None)

    @torch.inference_mode()
    def execute_model(self, scheduled: SchedulerOutput) -> WorkerOutput:
        for rid in scheduled.finished_request_ids:
            self.unregister_request(rid)
        for rid in scheduled.preempted_request_ids:
            if self.mtp is not None:
                self.mtp.forget(rid)
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
        if not plans:
            return WorkerOutput(registration_errors)
        validate_batch(plans)
        plans.sort(key=lambda p: not p.prefill)
        ids, positions, sequence_ids, writes, slots, pages = [], [], [], [], [], []
        starts, page_starts, last_lengths = [0], [0], []
        for sequence, plan in enumerate(plans):
            req = plan.request
            count = len(req.token_ids)
            end = req.num_computed_tokens + count
            ids.extend(req.token_ids)
            positions.extend(range(req.num_computed_tokens, end))
            sequence_ids.extend([sequence] * count)
            writes.extend(plan.writes)
            slots.extend(
                req.fa_block_table[p // BLOCK] * BLOCK + p % BLOCK
                for p in range(req.num_computed_tokens, end)
            )
            starts.append(len(ids))
            pages.extend(req.fa_block_table[: (end + BLOCK - 1) // BLOCK])
            page_starts.append(len(pages))
            last_lengths.append((end - 1) % BLOCK + 1)
        cpu_starts = torch.tensor(starts, dtype=torch.int32)
        extent = min(
            self.config.max_model_len, ((max(positions) + 4096) // 4096) * 4096
        )
        key = (len(ids), len(plans), extent)
        use_graph = (
            not any(p.prefill for p in plans)
            and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1"
            and self.graph_cache.should_use("target", key)
        )
        if not use_graph:
            self.attention.plan(
                cpu_starts,
                torch.tensor(page_starts, dtype=torch.int32),
                torch.tensor(pages, dtype=torch.int32),
                torch.tensor(last_lengths, dtype=torch.int32),
                24,
                4,
                256,
            )
        metadata = device_vectors(
            [
                positions,
                starts,
                sequence_ids,
                [p.source for p in plans],
                writes,
                [p.writes[-1] for p in plans],
                slots,
                ids,
            ]
        )
        batch = Batch(
            positions=metadata[0],
            # FlashInfer prefill consumes int32 offsets; convert once per batch.
            starts=metadata[1].to(torch.int32),
            sequence_ids=metadata[2],
            state_reads=metadata[3],
            state_writes=metadata[4],
            final_state_writes=metadata[5],
            fa_slots=metadata[6],
            attention=self.attention,
            prefill_sequences=sum(p.prefill for p in plans),
            prefill_tokens=sum(len(p.writes) for p in plans if p.prefill),
        )
        token_tensor = metadata[7]
        graph_logits = None
        if use_graph:
            from oh_my_vllm.worker.decode_graph import DecodeGraph

            width = (extent + BLOCK - 1) // BLOCK
            tables = device_page_tables(
                [p.request.fa_block_table for p in plans],
                width,
                counts=[len(p.writes) for p in plans],
                dtype=torch.int32,
                device="cuda",
            )

            def capture():
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                logger.info(
                    "Capture target graph: tokens=%d requests=%d extent=%d", *key
                )
                try:
                    return DecodeGraph(
                        self.model,
                        self.caches,
                        token_tensor,
                        batch,
                        tables,
                        extent,
                        pool=self.graph_pool,
                    )
                except Exception:
                    if not self.graph_cache.has_family("target"):
                        self.graph_pool = None
                    raise

            graph = self.graph_cache.get_or_create("target", key, capture)
            if graph is None:
                self.attention.plan(
                    cpu_starts,
                    torch.tensor(page_starts, dtype=torch.int32),
                    torch.tensor(pages, dtype=torch.int32),
                    torch.tensor(last_lengths, dtype=torch.int32),
                    24,
                    4,
                    256,
                )
            else:
                hidden, graph_logits = graph.replay(token_tensor, batch, tables)
        if graph_logits is None:
            hidden = self.model.forward(token_tensor, batch, self.caches)
        selected = [
            starts[i] + row for i, p in enumerate(plans) for row in p.sample_indices
        ]
        logits = None
        if selected:
            if graph_logits is None:
                logits = self.model.logits(hidden[selected])
            elif selected == list(range(len(ids))):
                logits = graph_logits
            else:
                logits = graph_logits[selected]
        # The grammar adapter only needs these two protocol-independent mappings.
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
        if copies:
            sources, destinations = (
                device_tensor(values, device="cuda")
                for values in zip(*copies, strict=True)
            )
            for kind, cache in zip(self.model.kinds, self.caches, strict=True):
                if kind == "linear_attention":
                    for pool in cache:
                        pool.index_copy_(0, destinations, pool.index_select(0, sources))
        if self.mtp is not None and successful_plans:
            drafts = self.mtp.propose(
                successful_plans,
                successful_starts,
                successful_counts,
                hidden,
                self.histories,
                successful_outputs,
            )
            for output in successful_outputs:
                output.new_draft_token_ids = drafts[output.request_id]
        return WorkerOutput(registration_errors + results)

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
                if self.mtp is not None:
                    self.mtp.validate_state(plan, count)
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
                    rows = self.samplers[rid].draw_rows(
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
        sampled = {}
        offset = 0
        for index, _, count in drawn:
            sampled[index] = host_rows[offset : offset + count]
            offset += count
        return sampled, errors

    def shutdown(self) -> None:
        logger.info(
            "Target graph cache",
            extra={"fields": self.graph_cache.snapshot()},
        )
        if self.mtp is not None:
            logger.info(
                "MTP graph cache",
                extra={"fields": self.mtp.graph_cache.snapshot()},
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
        self.histories.clear()
        self.samplers.clear()
        self.sources.clear()
        self.computed.clear()
        self.graph_cache.clear()
        self.caches = []
        self.model = None
        self.attention = None
        self.serving = None
        self.mtp = None
        torch.cuda.empty_cache()
