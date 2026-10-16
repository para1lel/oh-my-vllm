from contextlib import ExitStack

import numpy as np

from ..kernels.fa import FA
from ..kernels.ffn import FFN
from ..kernels.gdn import GDN
from .metadata import Metadata


def metadata_arrays(batch):
    return (
        *batch.metadata,
        batch.tokens,
        np.asarray(batch.schedule.sequence_rows, np.int32),
        np.asarray(batch.schedule.read_slots, np.int32),
        np.asarray(batch.writes, np.int32),
    )


class Workspace:
    def __init__(self, device, weights, batch, hidden):
        self.device, self.weights = device, weights
        self.gdn = GDN(
            weights.layers[0].attention, batch.schedule, device.empty, device.upload
        )
        attention = next(
            layer.attention
            for layer in weights.layers
            if layer.kind == "full_attention"
        )
        self.fa = FA(attention, batch.rows, device.empty, max_queries=batch.max_queries)
        self.ffn = FFN(weights.layers[0].feed_forward, batch.rows, device.empty)
        self.input = device.empty((batch.rows, 5120), "bfloat16")
        self.zero = device.zeros((batch.rows, 5120), "bfloat16")
        self.metadata_inputs = Metadata(device, metadata_arrays(batch))
        self.metadata = self.metadata_inputs.arrays[:5]
        self.tokens = self.metadata_inputs.arrays[5]
        self.gdn.metadata = self.metadata_inputs.arrays[6:]
        self.hidden = device.empty((batch.rows, 5120), "bfloat16") if hidden else None
        self.resources = ExitStack()
        self.replay = None
        self.media = None
        self.media_rows = None
        self.output = None

    def close(self):
        self.resources.close()

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
            self.replay = self.resources.enter_context(
                self.device.graph(lambda: self.forward(cache))
            )
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
                for offset in range(len(row["token_ids"]))
            ]
        )
        result = workspace.run(self.cache, graph)
        if self.mtp:
            self.cache.export_hidden(workspace.hidden, batch)
        return result, workspace.hidden
