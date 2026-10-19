from contextlib import ExitStack
from math import prod

import numpy as np

from ..kernels.fa import FA
from ..kernels.ffn import FFN
from ..kernels.gdn import GDN, Schedule
from ..kernels.mtp import MTPContext
from .metadata import Metadata


def metadata_arrays(batch):
    return (
        *batch.metadata,
        batch.tokens,
        np.asarray(batch.schedule.sequence_rows, np.int32),
        np.asarray(batch.schedule.read_slots, np.int32),
        np.asarray(batch.writes, np.int32),
    )


def minimum_context(batch):
    return int(batch.metadata[4][batch.metadata[3][1:] - 1].min())


def scratch_capacity(weights, rows, hidden):
    size = 0

    def count(shape, dtype):
        nonlocal size
        width = {
            "bfloat16": 2,
            "float8_e4m3fn": 1,
            "float32": 4,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
        }[dtype]
        size += ((max(prod(shape) * width, 1) + 255) // 256) * 256

    checkpoints = tuple(range(783, rows, 784))
    if rows - 1 not in checkpoints:
        checkpoints += (rows - 1,)
    schedule = Schedule(
        (0, rows),
        (0,),
        (-1,) * rows,
        checkpoints,
        tuple(range(1, len(checkpoints) + 1)),
        (0,),
        len(checkpoints) + 1,
        False,
    )
    GDN(weights.layers[0].attention, schedule, count, lambda value: None, prepare=False)
    attention = next(
        layer.attention for layer in weights.layers if layer.kind == "full_attention"
    )
    FA(attention, rows, count, max_queries=rows, prepare=False)
    FFN(weights.layers[0].feed_forward, rows, count, prepare=False)
    for _ in range(3 if hidden else 2):
        count((rows, 5120), "bfloat16")
    if hidden:
        MTPContext(
            weights.mtp_input,
            weights.attention_variant(weights.mtp_layer.attention, "kv_only"),
            rows,
            count,
            max_queries=rows,
        )
    return size


class Workspace:
    def __init__(self, device, weights, batch, hidden):
        self.device, self.weights = device, weights
        self.gdn = GDN(
            weights.layers[0].attention, batch.schedule, device.scratch, device.upload
        )
        attention = next(
            layer.attention
            for layer in weights.layers
            if layer.kind == "full_attention"
        )
        self.fa = FA(
            attention,
            batch.rows,
            device.scratch,
            max_queries=batch.max_queries,
            max_context=int(batch.metadata[4].max()),
            min_context=minimum_context(batch),
        )
        self.ffn = FFN(weights.layers[0].feed_forward, batch.rows, device.scratch)
        self.input = device.scratch((batch.rows, 5120), "bfloat16")
        self.zero = device.scratch((batch.rows, 5120), "bfloat16")
        device.module.zero(self.zero)
        self.metadata_inputs = Metadata(device, metadata_arrays(batch))
        self.metadata = self.metadata_inputs.arrays[:5]
        self.tokens = self.metadata_inputs.arrays[5]
        self.gdn.metadata = self.metadata_inputs.arrays[6:]
        self.hidden = device.scratch((batch.rows, 5120), "bfloat16") if hidden else None
        self.resources = ExitStack()
        self.replay = None
        self.media = None
        self.media_rows = None
        self.output = None

    def close(self):
        self.resources.close()
        self.gdn.close()
        self.fa.close()
        self.ffn.close()

    def update(self, batch):
        self.metadata_inputs.update(metadata_arrays(batch))

    def forward(self, cache):
        value, residual = self.input, self.zero
        recurrent = full = 0
        for index, layer in enumerate(self.weights.layers):
            if layer.kind == "full_attention":
                self.fa.weights = layer.attention
                value, residual = self.fa(
                    value,
                    residual,
                    self.metadata[0],
                    cache.fa[full],
                    *self.metadata[1:],
                )
                full += 1
            else:
                self.gdn.weights = layer.attention
                value, residual = self.gdn(
                    value,
                    residual,
                    cache.state[recurrent],
                    cache.convolution[recurrent],
                    self.tokens if index == 0 else None,
                    self.media if index == 0 else None,
                    self.media_rows if index == 0 else None,
                )
                recurrent += 1
            self.ffn.weights = layer.feed_forward
            value, residual = self.ffn(value, residual)
        if self.hidden is not None:
            self.device.module.final_hidden(
                value, residual, self.weights.sample.norm, self.hidden
            )
        self.output = (value, residual)
        return self.output

    def run(self, cache, graph):
        if not graph:
            return self.forward(cache)
        if self.replay is None:
            self.forward(cache)
            self.device.synchronize()
            with ExitStack() as recording:
                for operator in (self.ffn, self.gdn, self.fa):
                    recording.enter_context(operator.graph_lease())
                replay = recording.enter_context(
                    self.device.graph(lambda: self.forward(cache))
                )
                self.resources.enter_context(recording.pop_all())
                self.replay = replay
        self.replay()
        return self.output


class Model:
    def __init__(self, device, weights, cache, mtp, media):
        self.device, self.weights, self.cache, self.mtp = device, weights, cache, mtp
        self.decode = {}
        self.prefill = None
        self.prefill_key = None
        self.media = media

    def close(self):
        for workspace in self.decode.values():
            workspace.close()
        if self.prefill is not None:
            self.prefill.close()

    def forward(self, batch, graph):
        key = (
            batch.rows,
            len(batch.requests),
            batch.max_queries,
            batch.schedule.verify,
            batch.max_queries >= 1024 and int(batch.metadata[4].max()) >= 12544,
            batch.max_queries == 1
            and (
                (len(batch.requests) >= 4 and int(batch.metadata[4].max()) >= 32768)
                or (len(batch.requests) == 2 and int(batch.metadata[4].min()) >= 32768)
            ),
            (34 if batch.max_queries == 2 else 36)
            if len(batch.requests) == 2
            and 2 <= batch.max_queries <= 5
            and batch.rows > batch.max_queries
            and minimum_context(batch) >= 32768
            else 0,
        )
        if batch.schedule.verify:
            if key not in self.decode:
                self.decode[key] = Workspace(self.device, self.weights, batch, self.mtp)
            workspace = self.decode[key]
        else:
            if self.prefill_key != key:
                if self.prefill is not None:
                    self.prefill.close()
                    self.prefill = None
                self.prefill = Workspace(self.device, self.weights, batch, self.mtp)
                self.prefill_key = key
            workspace = self.prefill
        workspace.update(batch)
        workspace.media, workspace.media_rows = self.media.rows(
            [
                (
                    request,
                    row["num_computed_tokens"] + offset,
                    batch.schedule.sequence_rows[index] + offset,
                )
                for index, (row, request) in enumerate(
                    zip(batch.scheduled, batch.requests, strict=True)
                )
                if request.media
                for offset in range(len(row["token_ids"]))
            ]
        )
        result = workspace.run(self.cache, graph)
        if self.mtp:
            self.cache.export_hidden(workspace.hidden, batch)
        return result, workspace.hidden
