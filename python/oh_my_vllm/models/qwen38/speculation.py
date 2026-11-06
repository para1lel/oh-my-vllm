from collections import OrderedDict

import numpy as np
import torch

from ...operators.mtp.plan import context_plan
from .draft_chain import DraftChain, can_chain
from .draft_pipeline import DraftPipeline, can_pipeline
from .model import MAX_GRAPH_WORKSPACES
from .speculation_ops import Acceptance, Context, Draft


class Speculation:
    def __init__(self, device, weights, cache, media, trace):
        self.device, self.weights, self.cache = device, weights, cache
        self.context_decode = OrderedDict()
        self.context_prefill = None
        self.context_key = None
        self.drafts = {}
        self.chains = {}
        self.pipelines = {}
        self.acceptance = {}
        self.media = media
        self.trace = trace

    def close(self):
        for operation in (
            *self.context_decode.values(),
            *self.drafts.values(),
            *self.chains.values(),
            *self.pipelines.values(),
            *self.acceptance.values(),
        ):
            operation.close()
        if self.context_prefill is not None:
            self.context_prefill.close()

    def accept(self, batch, value, residual, sampler, graph):
        d = self.device
        requests = batch.requests
        indices, owners, prefixes = [], [], []
        for index, request in enumerate(requests):
            first, last = batch.schedule.sequence_rows[index : index + 2]
            for offset, row in enumerate(range(first, last)):
                indices.append(row)
                owners.append(request)
                prefixes.append(batch.tokens[first + 1 : first + offset + 1].tolist())
        _, probability = sampler.select(
            value, residual, indices, owners, graph, prefixes=prefixes, sample=False
        )
        count = len(requests)
        target_rows, drafts, slots, random = [], [], [], []
        for index, request in enumerate(requests):
            first, last = batch.schedule.sequence_rows[index : index + 2]
            length = last - first - 1
            if batch.tokens[first + 1 : last].tolist() != request.drafts[:length]:
                raise ValueError(
                    "scheduled drafts differ from stored proposal probabilities"
                )
            target_rows.extend(
                [*range(first, last), *([last - 1] * (5 - (last - first)))]
            )
            drafts.append(
                [
                    *request.drafts[:length],
                    *([-1] * (4 - length)),
                    request.drafts_greedy,
                ]
            )
            selected = list(batch.writes[first:last])
            for slot in self.cache.private_slots(
                batch.scheduled[index]["request_id"], batch.schedule.read_slots[index]
            ):
                if slot not in selected and slot not in batch.schedule.read_slots:
                    selected.append(slot)
                if len(selected) >= 5:
                    break
            slots.append(selected[:5])
            random.append((request.seed ^ 0xD1B54A32D192ED03, request.accept_offset))
            if not request.drafts_greedy and request.proposals is None:
                raise ValueError("stochastic drafts require proposal probabilities")
        arrays = (
            np.asarray(target_rows, np.int32),
            np.asarray(drafts, np.int64),
            np.asarray(slots, np.int32),
            np.asarray(random, np.uint64),
        )
        if count not in self.acceptance:
            self.acceptance[count] = Acceptance(d, count, arrays)
        else:
            self.acceptance[count].metadata_inputs.update(arrays)
        operation = self.acceptance[count]
        operation.greedy = sampler.greedy and all(
            request.drafts_greedy for request in requests
        )
        for index, request in enumerate(requests):
            if not request.drafts_greedy:
                operation.copy(operation.proposal_rows[index], request.proposals)
        d.module.gather_rows(
            operation.target_token_rows if operation.greedy else operation.target_rows,
            sampler.token_rows if operation.greedy else probability,
            operation.gather,
        )
        operation.capacity = self.cache.slots
        operation.execute(graph)
        kept = d.download(operation.operation.outputs.tokens, None)
        counts = kept[:, 5]
        counters = kept.view(torch.uint64)[:, 6]
        result = []
        for index, request in enumerate(requests):
            request.accept_offset = int(counters[index])
            request.drafts = []
            result.append(kept[index, : int(counts[index])].tolist())
        return result

    def context(self, batch, hidden, graph, final):
        prepared = self._context_rows(batch, final)
        if prepared is None:
            return {}
        tokens, active, media_entries, arrays, starts, lengths = prepared
        plan = context_plan(
            rows=len(tokens),
            batch=len(active),
            table_width=arrays[5].shape[1],
            max_queries=max(np.diff(starts)),
        )
        operation = self._context_operation(batch, hidden, graph, arrays, plan)
        operation.media, indices = self.media.rows(media_entries)
        operation.media_rows = None
        if indices is not None:
            selected = self.device.download(indices, np.int32)
            mapping = np.full(len(tokens), -1, np.int32)
            mapping[selected] = np.arange(len(selected), dtype=np.int32)
            operation.media_rows = self.device.upload(mapping)
        operation.execute(graph)
        self.trace.record(
            "context",
            execution=(
                "cuda_graph"
                if graph and operation.media is None and operation.media_rows is None
                else "eager"
            ),
            requests=[
                {
                    "id": batch.scheduled[index]["request_id"],
                    "start": lengths[starts[local]] - 1,
                    "count": starts[local + 1] - starts[local],
                }
                for local, index in enumerate(active)
            ],
        )
        return {
            index: operation.endpoint_rows[local]
            for local, index in enumerate(active)
            if index in final
        }

    def _context_rows(self, batch, final):
        tokens, hidden_rows, boundaries, positions, slots, lengths, starts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [0],
        )
        active = []
        media_entries = []
        for index, (row, request) in enumerate(
            zip(batch.scheduled, batch.requests, strict=True)
        ):
            start = row["num_computed_tokens"]
            last = (
                request.state_position
                - 1
                + int(
                    index in final
                    and not request.finish
                    and request.generated + 1 < request.limit
                )
            )
            begin = max(1, start, request.mtp_position + 1)
            if begin > last:
                continue
            active.append(index)
            for position in range(begin, last + 1):
                media_entries.append((request, position, len(tokens)))
                tokens.append(request.history[position])
                hidden_rows.append(
                    batch.schedule.sequence_rows[index] + position - start - 1
                    if position > start
                    else -1
                )
                boundaries.append(batch.schedule.read_slots[index])
                positions.append(request.position(position))
                slots.append(
                    row["fa_block_table"][position // 784] * 784 + position % 784
                )
                lengths.append(position + 1)
            starts.append(len(tokens))
            request.mtp_position = last
        if not tokens:
            return None
        arrays = (
            np.asarray(tokens, np.int64),
            np.asarray(hidden_rows, np.int32),
            np.asarray(boundaries, np.int32),
            np.asarray(positions, np.int64).T.copy(),
            np.asarray(slots, np.int32),
            batch.metadata[2][active].copy(),
            np.asarray(starts, np.int32),
            np.asarray(lengths, np.int32),
        )

        return tokens, active, media_entries, arrays, starts, lengths

    def _context_operation(self, batch, hidden, graph, arrays, plan):
        # Prefill retains its target input; only decode needs independent staging.
        staged_decode = graph and batch.schedule.verify
        key = plan.reuse_key if staged_decode else (plan.reuse_key, id(hidden))
        created = False
        if staged_decode:
            if key not in self.context_decode:
                if len(self.context_decode) >= MAX_GRAPH_WORKSPACES:
                    _, evicted = self.context_decode.popitem(last=False)
                    evicted.close()
                self.context_decode[key] = Context(
                    self.device,
                    self.weights,
                    arrays,
                    plan,
                    hidden,
                    self.cache,
                    True,
                    single_upload=plan.batch >= 2,
                )
                created = True
            else:
                self.context_decode.move_to_end(key)
            operation = self.context_decode[key]
        else:
            if self.context_key != key:
                if self.context_prefill is not None:
                    self.context_prefill.close()
                    self.context_prefill = None
                self.context_prefill = Context(
                    self.device,
                    self.weights,
                    arrays,
                    plan,
                    hidden,
                    self.cache,
                    False,
                    single_upload=True,
                )
                self.context_key = key
                created = True
            operation = self.context_prefill
        if not created:
            operation.metadata_inputs.update(arrays)
        if staged_decode:
            operation.copy(self.device.view(operation.hidden, hidden.shape), hidden)
        else:
            operation.hidden = hidden

        return operation

    def record_draft(self, batch, active, stage):
        self.trace.record(
            "draft",
            stage=stage,
            requests=[
                {
                    "id": batch.scheduled[index]["request_id"],
                    "position": batch.requests[index].state_position + stage,
                }
                for index in active
            ],
        )

    def propose(self, batch, endpoints, graph):
        d = self.device
        active = self._prepare_drafts(batch, endpoints)
        if can_chain(batch, active, graph, self.cache.blocks):
            self._propose_chain(batch, active, endpoints)
            return
        previous = {index: endpoints[index][0] for index in active}
        first_stage = 0
        if can_pipeline(batch, active, graph, self.cache.blocks):
            count = len(active)
            if count not in self.pipelines:
                self.pipelines[count] = DraftPipeline(
                    d,
                    [Draft(d, self.weights, count, stage == 0) for stage in range(4)],
                    self.cache.fa[-1],
                )
            operation = self.pipelines[count]
            operation.prepare(batch, active, endpoints)
            first_stage, active, previous = operation.execute(
                batch, active, graph, self.record_draft
            )
        for stage in range(first_stage, 4):
            if not active:
                break
            count = len(active)
            key = (count, stage == 0)
            if key not in self.drafts:
                self.drafts[key] = Draft(d, self.weights, count, stage == 0)
            operation = self.drafts[key]
            arrays, cached_rows, requests = self._draft_metadata(
                batch, active, stage, operation, previous
            )
            operation.metadata_inputs.update(arrays)
            operation.cache = self.cache.fa[-1]
            self.record_draft(batch, active, stage)
            operation.execute(graph)
            chosen, probability = operation.sampling.select_from_logits(
                requests,
                graph,
                prefixes=[request.drafts for request in requests],
                stream="proposal",
            )
            active = self._store_draft_step(
                batch,
                active,
                stage,
                operation,
                cached_rows,
                chosen,
                probability,
                previous,
            )

    def _prepare_drafts(self, batch, endpoints):
        d = self.device
        active = [
            index
            for index in endpoints
            if not batch.requests[index].finish
            and batch.requests[index].generated + 1 < batch.requests[index].limit
        ]
        for index in active:
            request = batch.requests[index]
            request.drafts = []
            request.prepare_draft_guard()
            request.drafts_greedy = bool(request.parameters()[0] == 0)
            if not request.drafts_greedy and request.proposals is None:
                request.proposals = d.scratch((4, 248320), "float32")
                request.proposal_rows = tuple(
                    d.view(request.proposals, (1, 248320), index * 248320)
                    for index in range(4)
                )

        return active

    def _propose_chain(self, batch, active, endpoints):
        d = self.device
        count = len(active)
        greedy = all(batch.requests[index].drafts_greedy for index in active)
        key = (count, greedy)
        if key not in self.chains:
            self.chains[key] = DraftChain(
                d,
                [Draft(d, self.weights, count, stage == 0) for stage in range(4)],
                count,
                greedy,
            )
        operation = self.chains[key]
        operation.prepare(batch, active, endpoints, self.cache)
        for stage in range(4):
            self.record_draft(batch, active, stage)
        operation.execute()
        operation.store([batch.requests[index] for index in active])

    def _draft_metadata(self, batch, active, stage, operation, previous):
        d = self.device
        count = len(active)
        cached_rows = operation.row_views() if count >= 2 else None
        requests = [batch.requests[index] for index in active]
        positions, slots, tables, tokens = [], [], [], []
        for local, index in enumerate(active):
            request, row = batch.requests[index], batch.scheduled[index]
            position = request.state_position + stage
            positions.append(request.position(position))
            slots.append(row["fa_block_table"][position // 784] * 784 + position % 784)
            tables.append(batch.metadata[2][index])
            tokens.append(request.history[-1] if stage == 0 else request.drafts[-1])
            operation.copy(
                (
                    cached_rows[0][local]
                    if cached_rows is not None
                    else d.view(operation.hidden, (1, 5120), local * 5120)
                ),
                previous[index],
            )
        arrays = (
            np.asarray(positions, np.int64).T.copy(),
            np.asarray(slots, np.int32),
            np.asarray(tables, np.int32),
            np.arange(count + 1, dtype=np.int32),
            np.asarray(
                [request.state_position + stage + 1 for request in requests],
                np.int32,
            ),
            np.asarray(tokens, np.int64),
        )

        return arrays, cached_rows, requests

    def _store_draft_step(
        self,
        batch,
        active,
        stage,
        operation,
        cached_rows,
        chosen,
        probability,
        previous,
    ):
        d = self.device
        remaining = []
        for local, index in enumerate(active):
            request = batch.requests[index]
            if not request.drafts_greedy:
                operation.copy(
                    request.proposal_rows[stage],
                    (
                        cached_rows[1][local]
                        if cached_rows is not None
                        else d.view(probability, (1, 248320), local * 248320)
                    ),
                )
            request.drafts.append(int(chosen[local]))
            # The next stage copies all rows before the sampler reuses this buffer.
            previous[index] = (
                cached_rows[2][local]
                if cached_rows is not None
                else d.view(
                    operation.sampling.operation.outputs.hidden,
                    (1, 5120),
                    local * 5120,
                )
            )
            if (
                not request.stops_after(request.drafts)
                and request.generated + len(request.drafts) + 1 < request.limit
            ):
                remaining.append(index)
        return remaining
