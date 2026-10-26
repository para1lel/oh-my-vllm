from contextlib import ExitStack
from threading import get_ident

import numpy as np

from ...operators.fa import api as fa
from ...operators.mtp import api as mtp
from ...operators.mtp.plan import accept_plan, context_plan, draft_plan
from ...operators.mtp.workspace import prepare_accept, prepare_context, prepare_draft
from ...worker.metadata import Metadata
from .draft_chain import DraftChain, can_chain
from .sampling import Sampling


class GraphOperation:
    def __init__(self, device):
        self.device = device
        self.resources = ExitStack()
        self.host_uploads = device.uploads()
        self.resources.callback(self.host_uploads.close)
        self.replay = None
        self.runs = 0

    def close(self):
        self.resources.close()

    def copy(self, destination, source):
        if isinstance(source, np.ndarray):
            self.host_uploads.copy(destination, source)
        else:
            self.device.module.copy_bytes(destination, source)

    def execute(self, graph):
        if not graph:
            result = self.run()
        else:
            if self.replay is None:
                if not self.runs or self.thread != get_ident():
                    self.run()
                self.device.synchronize_compute()
                self.replay = self.resources.enter_context(self.device.graph(self.run))
            self.replay()
            result = None
        self.runs = 1
        self.thread = get_ident()
        return result


class Context(GraphOperation):
    def __init__(self, device, weights, arrays, plan, hidden, cache, graph):
        super().__init__(device)
        self.weights = mtp.Weights(
            weights.mtp_input,
            weights.attention_variant(weights.mtp_layer.attention, "kv_only"),
        )
        self.operation = prepare_context(
            plan,
            device,
            (self.weights,),
            allocate=device.scratch,
        )
        self.metadata_inputs = Metadata(device, arrays)
        self.arrays = self.metadata_inputs.arrays
        self.hidden = (
            device.scratch((cache.parallel * 5, 5120), "bfloat16") if graph else hidden
        )
        self.boundary_hidden = cache.hidden
        self.cache = cache.fa[-1]
        self.media = self.media_rows = None
        self.endpoint_rows = tuple(
            tuple(
                device.view(value, (1, 5120), row * 5120)
                for value in self.operation.outputs
            )
            for row in range(plan.batch)
        )

    def graph_lease(self):
        return self.operation.graph_lease()

    def execute(self, graph):
        if self.media is not None or self.media_rows is not None:
            self.runs = 0
            return self.run()
        return super().execute(graph)

    def close(self):
        super().close()
        self.metadata_inputs.close()
        self.operation.close()

    def run(self):
        (
            tokens,
            hidden_rows,
            boundary_rows,
            positions,
            slots,
            tables,
            starts,
            lengths,
        ) = self.arrays
        inputs = mtp.Inputs(
            mtp.InputView(
                self.hidden,
                tokens,
                hidden_rows,
                self.boundary_hidden,
                boundary_rows,
                self.media,
                self.media_rows,
            ),
            fa.BatchView(positions, starts, lengths),
            fa.CacheView(self.cache, slots, tables),
        )
        return self.operation.run(self.weights, inputs)


class Draft(GraphOperation):
    def __init__(self, device, weights, batch, first, *, table_width=335):
        super().__init__(device)
        self.first = first
        self.weights = mtp.Weights(
            weights.mtp_input,
            weights.attention_variant(weights.mtp_layer.attention, "query_only")
            if first
            else weights.mtp_layer.attention,
            weights.mtp_layer.feed_forward,
            weights.mtp_sample,
        )
        self.operation = prepare_draft(
            draft_plan(rows=batch, first=first, table_width=table_width),
            device,
            (self.weights,),
        )
        self.attention, self.ffn = self.operation.attention, self.operation.ffn
        self.hidden = device.empty((batch, 5120), "bfloat16")
        self.zero = device.zeros((batch, 5120), "bfloat16")
        self.rows = device.upload(np.arange(batch, dtype=np.int32))
        self.boundary_rows = device.upload(np.zeros(batch, dtype=np.int32))
        arrays = tuple(
            np.zeros(shape, dtype=dtype)
            for shape, dtype in (
                ((3, batch), "int64"),
                ((batch,), "int32"),
                ((batch, table_width), "int32"),
                ((batch + 1,), "int32"),
                ((batch,), "int32"),
                ((batch,), "int64"),
            )
        )
        self.metadata_inputs = Metadata(device, arrays)
        self.metadata = self.metadata_inputs.arrays[:5]
        self.tokens = self.metadata_inputs.arrays[5]
        self.sampling = Sampling(
            device,
            weights.mtp_sample,
            batch,
            hidden=self.ffn.outputs,
            operation=self.operation.output,
        )
        self.cache = None
        self.output = None

    def graph_lease(self):
        return self.operation.graph_lease()

    def close(self):
        super().close()
        if self.metadata_inputs is not None:
            self.metadata_inputs.close()
        self.sampling.close()
        self.operation.close()

    def run(self):
        positions, slots, tables, starts, lengths = self.metadata
        inputs = mtp.Inputs(
            mtp.InputView(
                self.hidden, self.tokens, self.rows, self.hidden, self.boundary_rows
            ),
            fa.BatchView(positions, starts, lengths),
            fa.CacheView(self.cache, slots, tables),
            fa.Inputs(self.hidden, self.zero),
        )
        self.operation.project(self.weights, inputs)
        self.output = self.ffn.outputs
        return self.output


class Acceptance(GraphOperation):
    def __init__(self, device, batch):
        super().__init__(device)
        self.operation = prepare_accept(accept_plan(batch=batch), device)
        self.target = device.empty((batch, 5, 248320), "float32")
        self.target_tokens = device.empty((batch, 5), "int64")
        self.proposal = device.empty((batch, 4, 248320), "float32")
        self.drafts = device.empty((batch, 5), "int64")
        self.rng = device.empty((batch, 2), "uint64")
        self.slots = device.empty((batch, 5), "int32")
        self.gather = device.empty((batch * 5,), "int32")
        self.proposal_rows = tuple(
            device.view(self.proposal, (4, 248320), index * 4 * 248320)
            for index in range(batch)
        )
        self.target_rows = device.view(self.target, (batch * 5, 248320))
        self.target_token_rows = device.view(self.target_tokens, (batch * 5, 1))
        self.capacity = None
        self.greedy = False
        self.replays = {}

    def graph_lease(self):
        return self.operation.graph_lease()

    def close(self):
        super().close()
        self.operation.close()

    def run(self):
        return self.operation.run(
            mtp.AcceptInputs(
                self.target_tokens if self.greedy else self.target,
                self.proposal,
                self.drafts,
                self.rng,
                self.slots,
                self.capacity,
            )
        )

    def execute(self, graph):
        if graph:
            if self.greedy not in self.replays:
                self.replays[self.greedy] = self.resources.enter_context(
                    self.device.graph(self.run)
                )
            self.replays[self.greedy]()
        else:
            self.run()


class Speculation:
    def __init__(self, device, weights, cache, media, trace):
        self.device, self.weights, self.cache = device, weights, cache
        self.context_decode = {}
        self.context_prefill = None
        self.context_key = None
        self.drafts = {}
        self.chains = {}
        self.acceptance = {}
        self.media = media
        self.trace = trace

    def close(self):
        for operation in (
            *self.context_decode.values(),
            *self.drafts.values(),
            *self.chains.values(),
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
        if count not in self.acceptance:
            self.acceptance[count] = Acceptance(d, count)
        operation = self.acceptance[count]
        operation.greedy = sampler.greedy and all(
            request.drafts_greedy for request in requests
        )
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
            destination = operation.proposal_rows[index]
            if not request.drafts_greedy:
                if request.proposals is None:
                    raise ValueError("stochastic drafts require proposal probabilities")
                operation.copy(destination, request.proposals)
        operation.copy(operation.gather, np.asarray(target_rows, np.int32))
        d.module.gather_rows(
            operation.target_token_rows if operation.greedy else operation.target_rows,
            sampler.token_rows if operation.greedy else probability,
            operation.gather,
        )
        operation.copy(operation.drafts, np.asarray(drafts, np.int64))
        operation.copy(operation.slots, np.asarray(slots, np.int32))
        operation.copy(operation.rng, np.asarray(random, np.uint64))
        operation.capacity = self.cache.slots
        operation.execute(graph)
        kept = d.download(operation.operation.outputs.tokens, np.int64)
        counts = kept[:, 5]
        counters = kept.view(np.uint64)[:, 6]
        result = []
        for index, request in enumerate(requests):
            request.accept_offset = int(counters[index])
            request.drafts = []
            result.append(kept[index, : counts[index]].tolist())
        return result

    def context(self, batch, hidden, graph, final):
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
            return {}
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
        plan = context_plan(
            rows=len(tokens),
            batch=len(active),
            table_width=arrays[5].shape[1],
            max_queries=max(np.diff(starts)),
        )
        # Prefill retains its target input; only decode needs independent staging.
        staged_decode = graph and batch.schedule.verify
        key = plan.reuse_key if staged_decode else (plan.reuse_key, id(hidden))
        created = False
        if staged_decode:
            if key not in self.context_decode:
                self.context_decode[key] = Context(
                    self.device,
                    self.weights,
                    arrays,
                    plan,
                    hidden,
                    self.cache,
                    True,
                )
                created = True
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
        active = [
            index
            for index in endpoints
            if not batch.requests[index].finish
            and batch.requests[index].generated + 1 < batch.requests[index].limit
        ]
        for index in active:
            request = batch.requests[index]
            request.drafts = []
            request.drafts_greedy = request.parameters()[0] == 0
            if not request.drafts_greedy and request.proposals is None:
                request.proposals = d.scratch((4, 248320), "float32")
                request.proposal_rows = tuple(
                    d.view(request.proposals, (1, 248320), index * 248320)
                    for index in range(4)
                )
        if can_chain(batch, active, graph, self.cache.blocks):
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
            return
        previous = {index: endpoints[index][0] for index in active}
        for stage in range(4):
            if not active:
                break
            count = len(active)
            key = (count, stage == 0)
            if key not in self.drafts:
                self.drafts[key] = Draft(d, self.weights, count, stage == 0)
            operation = self.drafts[key]
            requests = [batch.requests[index] for index in active]
            positions, slots, tables, tokens = [], [], [], []
            for local, index in enumerate(active):
                request, row = batch.requests[index], batch.scheduled[index]
                position = request.state_position + stage
                positions.append(request.position(position))
                slots.append(
                    row["fa_block_table"][position // 784] * 784 + position % 784
                )
                tables.append(batch.metadata[2][index])
                tokens.append(request.history[-1] if stage == 0 else request.drafts[-1])
                operation.copy(
                    d.view(operation.hidden, (1, 5120), local * 5120), previous[index]
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
            remaining = []
            for local, index in enumerate(active):
                request = batch.requests[index]
                if not request.drafts_greedy:
                    operation.copy(
                        request.proposal_rows[stage],
                        d.view(probability, (1, 248320), local * 248320),
                    )
                request.drafts.append(int(chosen[local]))
                # The next stage copies all rows before the sampler reuses this buffer.
                previous[index] = d.view(
                    operation.sampling.operation.outputs.hidden, (1, 5120), local * 5120
                )
                if (
                    not request.stops_after(request.drafts)
                    and request.generated + len(request.drafts) + 1 < request.limit
                ):
                    remaining.append(index)
            active = remaining
