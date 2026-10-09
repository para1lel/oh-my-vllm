"""DSpark proposals with target-indexed context and causal confidence stopping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.models.dspark import DSparkModel
from oh_my_vllm.performance.execution import certify_feedback, record, tracing
from oh_my_vllm.worker.batch_plan import BLOCK, PlannedRequest
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.protocol import RequestOutput
from oh_my_vllm.worker.sampler import greedy_rows, probabilities
from oh_my_vllm.worker.sampling import SamplingParams
from oh_my_vllm.worker.tensors import device_page_tables, device_tensor, device_vectors


@dataclass
class Proposal:
    tokens: list[int]
    probabilities: torch.Tensor | None


class DSpark:
    def __init__(
        self,
        target_model,
        path: str | Path,
        capacity: int,
        max_tokens: int,
        *,
        confidence_threshold: float = 0.2,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("DSpark capacity must be a positive integer")
        if type(max_tokens) is not int or not 0 < max_tokens <= 262144:
            raise ValueError("DSpark max_tokens must be in [1,262144]")
        if (
            not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("DSpark confidence threshold must be in [0,1]")
        self.model = DSparkModel(target_model, path)
        self.device = target_model.embedding.device
        self.capacity = capacity
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        config = self.model.config
        self.cache = [
            torch.zeros(
                capacity,
                2,
                BLOCK,
                config.num_key_value_heads,
                config.head_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.next_position: dict[int, int] = {}
        self.proposals: dict[int, Proposal] = {}
        self.generators: dict[int, torch.Generator] = {}
        self.compile_model = (
            self.device.type == "cuda"
            and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1"
        )
        self.inject_unit = self.model.inject
        self.backbone_unit = self.model.backbone
        self.greedy_unit = self.model.greedy
        self.step_unit = self.model.step
        if self.compile_model:
            self.inject_unit = compile_forward(self.inject_unit, unit="dspark_context")
            self.backbone_unit = compile_forward(
                self.backbone_unit, unit="dspark_backbone"
            )
            self.greedy_unit = compile_forward(self.greedy_unit, unit="dspark_greedy")
            self.step_unit = compile_forward(self.step_unit, unit="dspark_markov")
        self.graph_cache = GraphCache(
            capacity=16,
            churn_cooldown_decisions=32768,
            free_bytes=(
                (lambda: torch.cuda.mem_get_info(self.device)[0])
                if self.device.type == "cuda"
                else None
            ),
            reserved_bytes=(
                (lambda: torch.cuda.memory_reserved(self.device))
                if self.device.type == "cuda"
                else None
            ),
        )
        self.graph_pool = None
        # Context commits have no live outputs. Keep their admission budget and
        # temporary storage independent from the read-only proposal graphs.
        self.context_graph_cache = GraphCache(
            # Context graphs support exactly 1..32 rows. Retain each row count
            # so acceptance-driven shape changes cannot churn this finite set.
            capacity=32,
            churn_cooldown_decisions=32768,
            free_bytes=self.graph_cache.free_bytes,
            reserved_bytes=self.graph_cache.reserved_bytes,
        )
        self.context_graph_pool = None

    def forget(self, request_id: int) -> None:
        self.next_position.pop(request_id, None)
        self.proposals.pop(request_id, None)
        self.generators.pop(request_id, None)

    def validate_state(self, plan: PlannedRequest, count: int) -> None:
        request = plan.request
        start = request.num_computed_tokens
        if type(count) is not int or not 0 < count <= len(request.token_ids):
            raise ValueError("DSpark target commit must include kept input rows")
        if start < 0 or start + count > self.max_tokens:
            raise ValueError("DSpark target context exceeds its configured limit")
        expected = self.next_position.get(request.request_id)
        if expected is None:
            if start % BLOCK:
                raise ValueError(
                    "DSpark prefix admission must start at a page boundary"
                )
        elif expected != start:
            raise ValueError("DSpark context position disagrees with target state")
        table = request.fa_block_table
        needed = (start + count + BLOCK - 1) // BLOCK
        if len(table) < needed or any(
            type(page) is not int or not 0 < page < self.capacity
            for page in table[:needed]
        ):
            raise ValueError("DSpark context pages are missing or outside the pool")

    def draft_probabilities(self, request_id: int, drafts: list[int]) -> torch.Tensor:
        proposal = self.proposals.get(request_id)
        if proposal is None or proposal.probabilities is None:
            raise ValueError("DSpark stochastic proposal probabilities are unavailable")
        if (
            len(drafts) > len(proposal.tokens)
            or drafts != proposal.tokens[: len(drafts)]
        ):
            raise ValueError("DSpark verifier drafts differ from the sampled prefix")
        return proposal.probabilities[: len(drafts)]

    def _inject(self, features, positions, slots):
        from oh_my_vllm.worker.dspark_graph import DSparkContextGraph

        key = (len(features),)
        family = "dspark_context"
        if (
            self.compile_model
            and 0 < len(features) <= DSparkContextGraph.MAX_ROWS
            and self.context_graph_cache.should_use(family, key)
        ):

            def capture():
                if self.context_graph_pool is None:
                    self.context_graph_pool = torch.cuda.graph_pool_handle()
                try:
                    return DSparkContextGraph(
                        self.inject_unit,
                        self.cache,
                        features,
                        positions,
                        slots,
                        pool=self.context_graph_pool,
                    )
                except Exception:
                    if not self.context_graph_cache.has_family(family):
                        self.context_graph_pool = None
                    raise

            graph = self.context_graph_cache.get_or_create(family, key, capture)
            if graph is not None:
                return graph.replay(features, positions, slots)
        return self.inject_unit(features, positions, slots, self.cache)

    def _run(self, anchors, positions, tables, lengths, *, greedy: bool):
        config = self.model.config
        tokens = torch.full(
            (len(anchors), config.block_size),
            config.mask_token_id,
            device=self.device,
            dtype=torch.int64,
        )
        tokens[:, 0] = device_tensor(anchors, device=self.device)
        position_tensor = device_tensor(positions, device=self.device)[:, None]
        position_tensor = (
            position_tensor
            + torch.arange(config.block_size, device=self.device)[None, :]
        )
        extent = min(self.max_tokens, ((max(lengths) + 4095) // 4096) * 4096)
        width = max(1, (extent + BLOCK - 1) // BLOCK)
        table_tensor = device_page_tables(tables, width, device=self.device)
        length_tensor = device_tensor(lengths, device=self.device, dtype=torch.int32)
        unit = self.greedy_unit if greedy else self.backbone_unit
        key = (greedy, len(anchors), extent)
        if self.compile_model and self.graph_cache.should_use("dspark", key):
            from oh_my_vllm.worker.dspark_graph import DSparkGraph

            def capture():
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                try:
                    return DSparkGraph(
                        unit,
                        self.cache,
                        tokens,
                        position_tensor,
                        table_tensor,
                        length_tensor,
                        pool=self.graph_pool,
                        key=key,
                    )
                except Exception:
                    if not self.graph_cache.has_family("dspark"):
                        self.graph_pool = None
                    raise

            graph = self.graph_cache.get_or_create("dspark", key, capture)
            if graph is not None:
                return graph.replay(
                    tokens, position_tensor, table_tensor, length_tensor
                )
        return unit(tokens, position_tensor, table_tensor, length_tensor, self.cache)

    def _generator(self, request_id: int, sampler) -> torch.Generator:
        if request_id not in self.generators:
            generator = torch.Generator(device=self.device)
            if sampler.params.seed is None:
                generator.seed()
            else:
                # An independent stream preserves the target sampler's RNG state.
                generator.manual_seed(
                    (sampler.params.seed ^ 0x44535041524B) % (1 << 64)
                )
            self.generators[request_id] = generator
        return self.generators[request_id]

    @torch.inference_mode()
    def propose(
        self,
        plans: list[PlannedRequest],
        starts: list[int],
        counts: list[int],
        features: torch.Tensor,
        histories: dict[int, list[int]],
        outputs: list[RequestOutput],
        samplers=None,
        serving=None,
    ) -> dict[int, list[int]]:
        if (
            len(plans) != len(counts)
            or len(plans) != len(outputs)
            or len(starts) < len(plans)
        ):
            raise ValueError("DSpark request metadata lengths disagree")
        if len({plan.request.request_id for plan in plans}) != len(plans):
            raise ValueError("DSpark request IDs must be unique within a batch")
        config = self.model.config
        if features.ndim != 2 or features.shape[1] != config.feature_size:
            raise ValueError("DSpark target features have the wrong shape")
        if features.dtype != torch.bfloat16 or features.device != self.device:
            raise ValueError("DSpark target features must be BF16 on the model device")
        proposals = {}
        eligible = []
        kept_features, positions, slots = [], [], []
        for index, (plan, count, output) in enumerate(
            zip(plans, counts, outputs, strict=True)
        ):
            self.validate_state(plan, count)
            request = plan.request
            rid = request.request_id
            if output.request_id != rid:
                raise ValueError("DSpark output request ID disagrees with plan")
            first = starts[index]
            if type(first) is not int or first < 0 or first + count > len(features):
                raise ValueError("DSpark kept features are outside the target batch")
            cursor = request.num_computed_tokens + count
            kept_features.append(features[first : first + count])
            positions.extend(range(request.num_computed_tokens, cursor))
            slots.extend(
                request.fa_block_table[position // BLOCK] * BLOCK + position % BLOCK
                for position in range(request.num_computed_tokens, cursor)
            )
            self.proposals.pop(rid, None)
            proposals[rid] = []
            if output.token_ids and output.finish_reason is None:
                history = histories.get(rid)
                if (
                    history is None
                    or len(history) <= cursor
                    or history[cursor] != output.token_ids[-1]
                ):
                    raise ValueError(
                        "DSpark anchor disagrees with committed target history"
                    )
                sampler = None if samplers is None else samplers.get(rid)
                if samplers is not None and sampler is None:
                    raise ValueError("DSpark request sampler is missing")
                remaining = self.max_tokens - cursor - 1
                if sampler is not None:
                    remaining = min(
                        remaining,
                        sampler.params.max_tokens - len(sampler.generated) - 1,
                    )
                limit = min(config.block_size, remaining)
                if limit > 0:
                    generation = (
                        None if serving is None else serving.generations.get(rid)
                    )
                    matcher = None if generation is None else generation.matcher
                    eligible.append(
                        (
                            rid,
                            request.fa_block_table,
                            cursor,
                            history[cursor],
                            limit,
                            sampler,
                            matcher,
                        )
                    )
        if kept_features:
            if tracing():
                record(
                    "dspark_inject",
                    rows=len(positions),
                    requests=[p.request.request_id for p in plans],
                    queries=counts,
                    contexts=[p.request.num_computed_tokens for p in plans],
                )
            position_tensor, slot_tensor = device_vectors(
                [positions, slots], device=self.device
            )
            self._inject(torch.cat(kept_features), position_tensor, slot_tensor)
            for plan, count in zip(plans, counts, strict=True):
                self.next_position[plan.request.request_id] = (
                    plan.request.num_computed_tokens + count
                )
        if not eligible:
            return proposals
        greedy = all(
            matcher is None and (sampler is None or sampler.plain_greedy)
            for _, _, _, _, _, sampler, matcher in eligible
        )
        trace_operation = None
        if tracing():
            trace_operation = record(
                "dspark_backbone",
                rows_per_request=config.block_size,
                anchors=[anchor for _, _, _, anchor, _, _, _ in eligible],
                tables=[table for _, table, _, _, _, _, _ in eligible],
                contexts=[cursor for _, _, cursor, _, _, _, _ in eligible],
                incoming_contexts=[
                    next(
                        p.request.num_computed_tokens
                        for p in plans
                        if p.request.request_id == rid
                    )
                    for rid, _, _, _, _, _, _ in eligible
                ],
                requests=[rid for rid, _, _, _, _, _, _ in eligible],
                greedy=greedy,
            )
        results = self._run(
            [anchor for _, _, _, anchor, _, _, _ in eligible],
            [cursor for _, _, cursor, _, _, _, _ in eligible],
            [table for _, table, _, _, _, _, _ in eligible],
            [cursor for _, _, cursor, _, _, _, _ in eligible],
            greedy=greedy,
        )
        if greedy:
            tokens, confidence, valid = results
            # One readback for all candidates, scores and validity bits.
            rows = torch.stack(
                (tokens.float(), confidence.float(), valid.float()), -1
            ).tolist()
            certify_feedback()
            if trace_operation is not None:
                trace_operation["candidates"] = [
                    [int(item[0]) for item in row] for row in rows
                ]
            for item, row in zip(eligible, rows, strict=True):
                rid, _, _, _, limit, _, _ = item
                survival = 1.0
                for token, score, is_valid in row[:limit]:
                    if not math.isfinite(score):
                        raise RuntimeError("DSpark confidence is not finite")
                    survival *= score
                    if survival < self.confidence_threshold:
                        break
                    if not is_valid:
                        raise RuntimeError(
                            "DSpark reachable draft logits have no finite maximum"
                        )
                    proposals[rid].append(int(token))
                self.proposals[rid] = Proposal(list(proposals[rid]), None)
        else:
            hidden, base_logits = results
            for index, item in enumerate(eligible):
                rid, _, _, anchor, limit, sampler, matcher = item
                proposals[rid] = self._sample(
                    rid,
                    hidden[index],
                    base_logits[index],
                    anchor,
                    limit,
                    sampler,
                    matcher,
                )
        return proposals

    def _sample(self, rid, hidden, base_logits, anchor, limit, sampler, matcher):
        candidates, distributions = [], []
        advanced = 0
        survival = 1.0
        previous = device_tensor([anchor], device=self.device)
        params = (
            sampler.params
            if sampler is not None
            else SamplingParams(max_tokens=7, temperature=0)
        )
        try:
            for position in range(limit):
                logits, confidence = self.step_unit(
                    hidden[position : position + 1],
                    base_logits[position : position + 1],
                    previous,
                )
                score = float(confidence[0])
                if not math.isfinite(score):
                    raise RuntimeError("DSpark confidence is not finite")
                survival *= score
                if survival < self.confidence_threshold:
                    break
                mask = None
                if matcher is not None:
                    if matcher.is_terminated():
                        break
                    import xgrammar as xgr

                    mask = xgr.allocate_token_bitmask(1, self.model.config.vocab_size)
                    matcher.fill_next_token_bitmask(mask, 0)
                if sampler is not None and sampler.plain_greedy and mask is None:
                    token, valid = greedy_rows(logits)[0]
                else:
                    prompt = [] if sampler is None else sampler.prompt
                    generated = [] if sampler is None else sampler.generated
                    distribution = probabilities(
                        logits, params, prompt, [*generated, *candidates], bitmask=mask
                    )
                    if params.temperature == 0:
                        token = int(distribution[0].argmax())
                    else:
                        noise = torch.empty_like(distribution).exponential_(
                            generator=self._generator(rid, sampler)
                        )
                        token = int((distribution / noise).argmax(-1)[0])
                        distributions.append(distribution[0])
                    valid = bool(
                        torch.isfinite(distribution).all() & (distribution.sum() > 0)
                    )
                if not valid:
                    raise RuntimeError(
                        "DSpark reachable proposal distribution has no valid token"
                    )
                if matcher is not None:
                    if not matcher.accept_token(token):
                        raise RuntimeError(
                            "DSpark selected token violates its grammar mask"
                        )
                    advanced += 1
                candidates.append(token)
                previous = device_tensor([token], device=self.device)
        finally:
            if advanced:
                matcher.rollback(advanced)
        q = None
        if params.temperature > 0:
            q = (
                torch.stack(distributions)
                if distributions
                else torch.empty(
                    0,
                    self.model.config.vocab_size,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        self.proposals[rid] = Proposal(list(candidates), q)
        return candidates
