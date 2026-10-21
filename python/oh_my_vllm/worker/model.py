from contextlib import ExitStack
from functools import partial

import numpy as np

from ..operators.fa import api as fa
from ..operators.fa.plan import make_plan as fa_plan
from ..operators.fa.workspace import prepare as prepare_fa
from ..operators.ffn import api as ffn
from ..operators.ffn.plan import make_plan as ffn_plan
from ..operators.ffn.workspace import prepare as prepare_ffn
from ..operators.gdn import api as gdn
from ..operators.gdn.plan import make_plan as gdn_plan
from ..operators.gdn.workspace import prepare as prepare_gdn
from ..operators.mtp.plan import context_plan
from ..operators.output import api as output
from ..operators.output.plan import make_plan as output_plan
from ..operators.output.workspace import prepare as prepare_output
from ..operators.runtime.storage import storage_bytes
from .metadata import Metadata


def metadata_arrays(batch):
    return (
        *batch.metadata,
        batch.tokens,
        np.asarray(batch.schedule.sequence_rows, np.int32),
        np.asarray(batch.schedule.read_slots, np.int32),
        np.asarray(batch.writes, np.int32),
    )


def make_plans(batch, hidden):
    attention = fa_plan(
        rows=batch.rows,
        batch=len(batch.requests),
        table_width=batch.metadata[2].shape[1],
        max_queries=batch.max_queries,
        max_context=int(batch.metadata[4].max()),
        min_context=int(batch.metadata[4][batch.metadata[3][1:] - 1].min()),
    )
    return (
        attention,
        gdn_plan(batch.schedule),
        ffn_plan(rows=batch.rows),
        output_plan(rows=batch.rows, hidden_only=True) if hidden else None,
    )


def scratch_capacity(weights, rows, hidden):
    checkpoints = tuple(range(783, rows, 784))
    if rows - 1 not in checkpoints:
        checkpoints += (rows - 1,)
    schedule = gdn.Schedule(
        (0, rows),
        (0,),
        (-1,) * rows,
        checkpoints,
        tuple(range(1, len(checkpoints) + 1)),
        (0,),
        len(checkpoints) + 1,
        False,
    )
    fields = (
        gdn_plan(schedule).storage
        + fa_plan(rows=rows, batch=1, table_width=335).storage
        + ffn_plan(rows=rows).storage
        + (((rows, 5120), "bfloat16"),) * 2
    )
    if hidden:
        mtp = context_plan(rows=rows)
        fields += (
            output_plan(rows=rows, hidden_only=True).storage
            + mtp.storage
            + mtp.attention.storage
        )
    return storage_bytes(fields)


class Workspace:
    def __init__(self, device, weights, cache, batch, plans):
        self.device, self.weights, self.cache = device, weights, cache
        self.fa = prepare_fa(
            plans[0],
            device,
            tuple(
                layer.attention
                for layer in weights.layers
                if layer.kind == "full_attention"
            ),
            allocate=device.scratch,
        )
        self.gdn = prepare_gdn(
            plans[1],
            device,
            tuple(
                layer.attention
                for layer in weights.layers
                if layer.kind != "full_attention"
            ),
            allocate=device.scratch,
        )
        self.ffn = prepare_ffn(
            plans[2],
            device,
            tuple(layer.feed_forward for layer in weights.layers),
            allocate=device.scratch,
        )
        self.input = device.scratch((batch.rows, 5120), "bfloat16")
        self.zero = device.scratch((batch.rows, 5120), "bfloat16")
        device.module.zero(self.zero)
        self.metadata_inputs = Metadata(device, metadata_arrays(batch))
        metadata = self.metadata_inputs.arrays
        self.tokens = metadata[5]
        self.batch_view = fa.BatchView(metadata[0], metadata[3], metadata[4])
        self.recurrent_view = gdn.BatchView(*metadata[6:])
        self.write_slots = metadata[8]
        self.hidden_operation = (
            prepare_output(plans[3], device, (weights.sample,), allocate=device.scratch)
            if plans[3] is not None
            else None
        )
        self.hidden = (
            self.hidden_operation.outputs.hidden if self.hidden_operation else None
        )
        self.resources = ExitStack()
        self.replay = None
        self.media = self.media_rows = None
        self.calls, self.warmups = [], []
        value = ffn.Inputs(self.input, self.zero)
        recurrent = full = 0
        for index, layer in enumerate(weights.layers):
            if layer.kind == "full_attention":
                state = fa.CacheView(cache.fa[full], metadata[1], metadata[2])
                arguments = (layer.attention, value, self.batch_view, state)
                operation = self.fa
                full += 1
            else:
                state = gdn.StateView(
                    cache.state[recurrent], cache.convolution[recurrent]
                )
                inputs = gdn.Inputs(
                    value.value, value.residual, self.tokens if index == 0 else None
                )
                arguments = (layer.attention, inputs, self.recurrent_view, state)
                operation = self.gdn
                if index == 0:
                    self.first_arguments = arguments
                recurrent += 1
            self.calls.append(partial(operation.run, *arguments))
            self.warmups.append(partial(operation.warmup, *arguments))
            self.calls.append(
                partial(self.ffn.run, layer.feed_forward, operation.outputs)
            )
            self.warmups.append(
                partial(self.ffn.warmup, layer.feed_forward, operation.outputs)
            )
            value = self.ffn.outputs
        self.output = value
        if self.hidden_operation is not None:
            self.calls.append(
                partial(
                    self.hidden_operation.hidden,
                    weights.sample,
                    value,
                    output.HiddenState(cache.hidden, self.write_slots),
                )
            )

    def close(self):
        self.resources.close()
        self.metadata_inputs.close()
        for operation in (self.gdn, self.fa, self.ffn, self.hidden_operation):
            if operation is not None:
                operation.close()

    def update(self, batch):
        self.metadata_inputs.update(metadata_arrays(batch))

    def bind_media(self):
        weight, inputs, batch, state = self.first_arguments
        if inputs.media is self.media and inputs.media_rows is self.media_rows:
            return
        inputs = gdn.Inputs(
            inputs.value, inputs.residual, inputs.tokens, self.media, self.media_rows
        )
        self.first_arguments = (weight, inputs, batch, state)
        self.calls[0] = partial(self.gdn.run, *self.first_arguments)
        self.warmups[0] = partial(self.gdn.warmup, *self.first_arguments)

    def forward(self):
        for operation in self.calls:
            operation()
        return self.output

    def run(self, graph):
        if not graph:
            return self.forward()
        if self.replay is None:
            for warmup in self.warmups:
                warmup()
            self.device.synchronize()
            with ExitStack() as recording:
                for operator in (self.ffn, self.gdn, self.fa, self.hidden_operation):
                    if operator is not None:
                        recording.enter_context(operator.graph_lease())
                replay = recording.enter_context(self.device.graph(self.forward))
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
        plans = make_plans(batch, self.mtp)
        key = tuple(plan.reuse_key for plan in plans if plan is not None)
        if batch.schedule.verify:
            if key not in self.decode:
                self.decode[key] = Workspace(
                    self.device, self.weights, self.cache, batch, plans
                )
            workspace = self.decode[key]
        else:
            if self.prefill_key != key:
                if self.prefill is not None:
                    self.prefill.close()
                    self.prefill = None
                self.prefill = Workspace(
                    self.device, self.weights, self.cache, batch, plans
                )
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
        workspace.bind_media()
        result = workspace.run(graph)
        return result, workspace.hidden
