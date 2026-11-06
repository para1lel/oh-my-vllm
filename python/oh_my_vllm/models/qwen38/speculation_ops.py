from contextlib import ExitStack
from threading import get_ident

import numpy as np
import torch

from ...operators.fa import api as fa
from ...operators.mtp import api as mtp
from ...operators.mtp.plan import accept_plan, draft_plan
from ...operators.mtp.workspace import prepare_accept, prepare_context, prepare_draft
from ...worker.metadata import Metadata
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
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(source)
        if source.device.type == "cpu":
            self.host_uploads.copy(destination, source.contiguous())
        elif (
            isinstance(destination, torch.Tensor)
            and source.device.type == "cuda"
            and destination.device == source.device
            and destination.dtype == source.dtype
            and destination.shape == source.shape
            and destination.is_contiguous()
            and source.is_contiguous()
        ):
            destination.copy_(source, non_blocking=True)
        else:
            self.device.copy(destination, source)

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
    def __init__(
        self, device, weights, arrays, plan, hidden, cache, graph, single_upload
    ):
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
        self.metadata_inputs = Metadata(device, arrays, single_upload=single_upload)
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
            allocate=device.scratch,
        )
        self.attention, self.ffn = self.operation.attention, self.operation.ffn
        self.hidden = device.scratch((batch, 5120), "bfloat16")
        self.zero = device.zero(device.scratch((batch, 5120), "bfloat16"))
        self.rows = device.upload(torch.arange(batch, dtype=torch.int32))
        self.boundary_rows = device.upload(torch.zeros(batch, dtype=torch.int32))
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
        self.metadata_inputs = Metadata(device, arrays, single_upload=batch >= 2)
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
        self._row_bindings = None
        self._row_views = None
        self.previous_hidden = None

    def row_views(self):
        bindings = (
            self.hidden,
            self.sampling.operation.outputs.probabilities,
            self.sampling.operation.outputs.hidden,
        )
        if self._row_bindings is None or any(
            before is not after
            for before, after in zip(self._row_bindings, bindings, strict=True)
        ):
            self._row_views = tuple(
                tuple(
                    self.device.view(value, (1, width), row * width)
                    for row in range(value.shape[0])
                )
                for value, width in zip(bindings, (5120, 248320, 5120), strict=True)
            )
            # Strong source references prevent identity reuse while views are cached.
            self._row_bindings = bindings
        return self._row_views

    def graph_lease(self):
        return self.operation.graph_lease()

    def close(self):
        super().close()
        self._row_views = self._row_bindings = None
        if self.metadata_inputs is not None:
            self.metadata_inputs.close()
        self.sampling.close()
        self.operation.close()

    def run(self):
        if self.previous_hidden is not None:
            self.copy(self.hidden, self.previous_hidden)
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
    def __init__(self, device, batch, arrays):
        super().__init__(device)
        self.operation = prepare_accept(
            accept_plan(batch=batch), device, allocate=device.scratch
        )
        self.target = device.scratch((batch, 5, 248320), "float32")
        self.target_tokens = device.scratch((batch, 5), "int64")
        self.proposal = device.scratch((batch, 4, 248320), "float32")
        self.metadata_inputs = Metadata(device, arrays, single_upload=True)
        self.gather, self.drafts, self.slots, self.rng = self.metadata_inputs.arrays
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
        self.metadata_inputs.close()
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
