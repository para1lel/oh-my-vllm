from contextlib import ExitStack

import numpy as np
import tvm_ffi

from ..kernels.fa import FA
from ..kernels.ffn import FFN
from ..kernels.mtp import MTPContext, MTPInput
from ..kernels.sample import MTPAccept
from .draft_chain import DraftChain, can_chain
from .metadata import Metadata
from .sampling import Sampling


class GraphOperation:
    def __init__(self, device):
        self.device = device
        self.resources = ExitStack()
        self.replay = None

    def close(self):
        self.resources.close()

    def copy(self, destination, source):
        if isinstance(source, np.ndarray):
            source = tvm_ffi.from_dlpack(np.require(source, requirements=["C", "W"]))
        self.device.module.copy_bytes(destination, source)

    def execute(self, graph):
        if not graph:
            return self.run()
        if self.replay is None:
            self.run()
            self.device.synchronize()
            self.replay = self.resources.enter_context(self.device.graph(self.run))
        self.replay()


class Context(GraphOperation):
    def __init__(self, device, weights, arrays, batch, max_queries, hidden, cache):
        super().__init__(device)
        self.operation = MTPContext(
            weights.mtp_input,
            weights.attention_variant(weights.mtp_layer.attention, "kv_only"),
            len(arrays[0]),
            device.scratch,
            batch=batch,
            max_queries=max_queries,
        )
        self.metadata_inputs = Metadata(device, arrays)
        self.arrays = self.metadata_inputs.arrays
        self.hidden = hidden
        self.boundary_hidden = cache.hidden
        self.cache = cache.fa[-1]
        self.media = self.media_rows = None

    def graph_lease(self):
        return self.operation.graph_lease()

    def execute(self, graph):
        if self.media is not None or self.media_rows is not None:
            return self.run()
        return super().execute(graph)

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
        return self.operation(
            (
                self.hidden,
                tokens,
                hidden_rows,
                self.boundary_hidden,
                boundary_rows,
                self.media,
                self.media_rows,
            ),
            (positions, self.cache, slots, tables, starts, lengths),
        )


class Draft(GraphOperation):
    def __init__(self, device, weights, batch, first):
        super().__init__(device)
        self.first = first
        self.input = MTPInput(weights.mtp_input, batch, device.empty)
        self.attention = FA(
            weights.attention_variant(weights.mtp_layer.attention, "query_only")
            if first
            else weights.mtp_layer.attention,
            batch,
            device.empty,
            variant="query_only" if first else "full",
            max_queries=1,
            first_token=1,
        )
        self.ffn = FFN(weights.mtp_layer.feed_forward, batch, device.empty)
        self.hidden = device.empty((batch, 5120), "bfloat16")
        self.zero = device.zeros((batch, 5120), "bfloat16")
        self.rows = device.upload(np.arange(batch, dtype=np.int32))
        self.boundary_rows = device.upload(np.zeros(batch, dtype=np.int32))
        arrays = tuple(
            np.zeros(shape, dtype=dtype)
            for shape, dtype in (
                ((3, batch), "int64"),
                ((batch,), "int32"),
                ((batch, 335), "int32"),
                ((batch + 1,), "int32"),
                ((batch,), "int32"),
                ((batch,), "int64"),
            )
        )
        self.metadata_inputs = Metadata(device, arrays)
        self.metadata = self.metadata_inputs.arrays[:5]
        self.tokens = self.metadata_inputs.arrays[5]
        self.sampling = Sampling(device, weights.mtp_sample, batch)
        self.cache = None
        self.output = None

    def close(self):
        self.sampling.close()
        super().close()

    def run(self):
        value, residual = (
            (self.hidden, self.zero)
            if self.first
            else self.input(
                self.hidden, self.tokens, self.rows, self.hidden, self.boundary_rows
            )
        )
        value, residual = self.attention(
            value, residual, self.metadata[0], self.cache, *self.metadata[1:]
        )
        self.output = self.ffn(value, residual)
        return self.output


class Acceptance(GraphOperation):
    def __init__(self, device, batch):
        super().__init__(device)
        self.operation = MTPAccept(batch, device.empty)
        self.target = device.empty((batch, 5, 248320), "float32")
        self.proposal = device.empty((batch, 4, 248320), "float32")
        self.drafts = device.empty((batch, 4), "int64")
        self.rng = device.empty((batch, 2), "uint64")
        self.slots = device.empty((batch, 5), "int32")
        self.gather = device.empty((batch * 5,), "int32")
        self.capacity = None

    def run(self):
        return self.operation(
            self.target, self.proposal, self.drafts, self.rng, self.slots, self.capacity
        )

    def execute(self, graph):
        saved = (
            self.device.download(self.rng, np.uint64)
            if graph and self.replay is None
            else None
        )
        if graph and self.replay is None:
            self.run()
            self.copy(self.rng, saved)
            self.replay = self.resources.enter_context(self.device.graph(self.run))
        if graph:
            self.replay()
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
            drafts.append([*request.drafts[:length], *([-1] * (4 - length))])
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
            destination = d.view(operation.proposal, (4, 248320), index * 4 * 248320)
            if request.proposals is None:
                d.module.zero(destination)
            else:
                operation.copy(destination, request.proposals)
        operation.copy(operation.gather, np.asarray(target_rows, np.int32))
        d.module.gather_rows(
            d.view(operation.target, (count * 5, 248320)), probability, operation.gather
        )
        operation.copy(operation.drafts, np.asarray(drafts, np.int64))
        operation.copy(operation.slots, np.asarray(slots, np.int32))
        operation.copy(operation.rng, np.asarray(random, np.uint64))
        operation.capacity = self.cache.slots
        operation.execute(graph)
        kept = d.download(operation.operation.buffers[0], np.int64)
        counts = d.download(operation.operation.buffers[1], np.int32)
        counters = d.download(operation.rng, np.uint64)
        result = []
        for index, request in enumerate(requests):
            request.accept_offset = int(counters[index, 1])
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
        self.trace.record(
            "context",
            requests=[
                {
                    "id": batch.scheduled[index]["request_id"],
                    "start": lengths[starts[local]] - 1,
                    "count": starts[local + 1] - starts[local],
                }
                for local, index in enumerate(active)
            ],
        )
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
        key = (len(tokens), len(active), max(np.diff(starts)), id(hidden))
        if graph:
            if key not in self.context_decode:
                self.context_decode[key] = Context(
                    self.device,
                    self.weights,
                    arrays,
                    len(active),
                    key[2],
                    hidden,
                    self.cache,
                )
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
                    len(active),
                    key[2],
                    hidden,
                    self.cache,
                )
                self.context_key = key
            operation = self.context_prefill
        operation.metadata_inputs.update(arrays)
        operation.media, indices = self.media.rows(media_entries)
        operation.media_rows = None
        if indices is not None:
            selected = self.device.download(indices, np.int32)
            mapping = np.full(len(tokens), -1, np.int32)
            mapping[selected] = np.arange(len(selected), dtype=np.int32)
            operation.media_rows = self.device.upload(mapping)
        operation.execute(graph)
        return {
            index: tuple(
                self.device.view(value, (1, 5120), local * 5120)
                for value in operation.operation.endpoints
            )
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
            if request.proposals is None:
                request.proposals = d.empty((4, 248320), "float32")
        if can_chain(batch, active, graph, self.cache.blocks):
            count = len(active)
            if count not in self.chains:
                self.chains[count] = DraftChain(
                    d,
                    [Draft(d, self.weights, count, stage == 0) for stage in range(4)],
                    count,
                )
            operation = self.chains[count]
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
            chosen, probability = operation.sampling.select(
                *operation.output,
                list(range(count)),
                requests,
                graph,
                prefixes=[request.drafts for request in requests],
                stream="proposal",
            )
            remaining = []
            for local, index in enumerate(active):
                request = batch.requests[index]
                operation.copy(
                    d.view(request.proposals, (1, 248320), stage * 248320),
                    d.view(probability, (1, 248320), local * 248320),
                )
                request.drafts.append(int(chosen[local]))
                # The next stage copies all rows before the sampler reuses this buffer.
                previous[index] = d.view(
                    operation.sampling.operation.buffers[0], (1, 5120), local * 5120
                )
                if (
                    not request.stops_after(request.drafts)
                    and request.generated + len(request.drafts) + 1 < request.limit
                ):
                    remaining.append(index)
            active = remaining
