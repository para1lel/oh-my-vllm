from contextlib import ExitStack, contextmanager

import numpy as np

from ...worker.metadata import Metadata
from ...worker.repetition import HISTORY_WINDOW, may_repeat


def can_chain(batch, active, graph, blocks):
    if not graph or not active:
        return False
    for index in active:
        request = batch.requests[index]
        parameters = request.parameters()
        generated = request.generated
        if (
            parameters[4] != 0
            or parameters[6] != 1
            or request.thinking
            or request.constraint.matcher is not None
            or request.options.get("stop")
            or generated < request.minimum < generated + 4
            or (not request.ignore_eos and generated + 4 > request.minimum)
            or generated + 4 >= request.limit
        ):
            return False
        if request.options.get("sampling", {}).get("repetition_guard", True):
            history = request.history[
                max(request.prompt_len, len(request.history) - HISTORY_WINDOW) :
            ]
            if may_repeat(history, 3):
                return False
        table = batch.scheduled[index]["fa_block_table"]
        for position in range(request.state_position, request.state_position + 4):
            page = position // 784
            if page < 0 or page >= len(table) or not 0 < table[page] < blocks:
                return False
    return True


class DraftChain:
    def __init__(self, device, stages, count, greedy):
        self.device, self.stages = device, stages
        self.resources = ExitStack()
        self.replay = None
        self.tokens = device.empty((4, count), "int64")
        self.token_rows = [
            device.view(self.tokens, (count,), stage * count) for stage in range(4)
        ]
        self.hidden_rows = tuple(
            device.view(stages[0].hidden, (1, 5120), index * 5120)
            for index in range(count)
        )
        self.metadata_inputs = Metadata(
            device,
            [
                *(
                    value
                    for stage in stages
                    for value in stage.metadata_inputs.host_views[:5]
                ),
                stages[0].metadata_inputs.host_views[5],
            ],
        )
        for index, stage in enumerate(stages):
            stage.metadata = self.metadata_inputs.arrays[index * 5 : index * 5 + 5]
            stage.tokens = (
                stages[index - 1].sampling.operation.outputs.tokens
                if index
                else self.metadata_inputs.arrays[-1]
            )
            stage.metadata_inputs.close()
            stage.metadata_inputs = None
            stage.sampling.greedy = greedy
            stage.sampling.inputs.compact = True
            if index:
                sampler = stages[0].sampling
                stage.sampling.history = sampler.history
                stage.sampling.presence_history = sampler.presence_history
                stage.sampling.mask = sampler.mask
                stage.sampling.parameters = sampler.parameters
                stage.sampling.rng = sampler.rng

    @contextmanager
    def graph_lease(self):
        with ExitStack() as resources:
            for stage in self.stages:
                resources.enter_context(stage.graph_lease())
            yield

    def close(self):
        self.resources.close()
        self.metadata_inputs.close()
        for stage in self.stages:
            stage.close()

    def prepare(self, batch, active, endpoints, cache):
        requests = [batch.requests[index] for index in active]
        count = len(active)
        for local, index in enumerate(active):
            self.stages[0].copy(
                self.hidden_rows[local],
                endpoints[index][0],
            )
        sampler = self.stages[0].sampling
        # Eligibility makes the mask independent of the unknown draft prefix.
        sampler.inputs.update(requests, [[] for _ in requests], "proposal")
        if not sampler.inputs.history_needed:
            sampler.inputs.presence_history.fill(0)
        sampler.uploads.copy(sampler.presence_history, sampler.inputs.presence_history)
        if sampler.inputs.mask_changed:
            sampler.uploads.copy(sampler.mask, sampler.inputs.mask)
        for destination, source in (
            (sampler.parameters, sampler.inputs.parameters),
            (sampler.rng, sampler.inputs.random),
        ):
            sampler.uploads.copy(destination, source)
        metadata = []
        for number, stage in enumerate(self.stages):
            positions = [request.state_position + number for request in requests]
            arrays = (
                np.asarray(
                    [
                        request.position(position)
                        for request, position in zip(requests, positions, strict=True)
                    ],
                    np.int64,
                ).T.copy(),
                np.asarray(
                    [
                        batch.scheduled[index]["fa_block_table"][position // 784] * 784
                        + position % 784
                        for index, position in zip(active, positions, strict=True)
                    ],
                    np.int32,
                ),
                np.asarray([batch.metadata[2][index] for index in active], np.int32),
                np.arange(count + 1, dtype=np.int32),
                np.asarray(positions, np.int32) + 1,
            )
            metadata.extend(arrays)
            stage.cache = cache.fa[-1]
        metadata.append(
            np.asarray([request.history[-1] for request in requests], np.int64)
        )
        self.metadata_inputs.update(metadata)

    def run(self):
        for index, stage in enumerate(self.stages):
            if index:
                stage.copy(
                    stage.hidden,
                    self.stages[index - 1].sampling.operation.outputs.hidden,
                )
            stage.run()
            stage.sampling.draw()
            stage.copy(self.token_rows[index], stage.sampling.operation.outputs.tokens)
            if index < 3:
                stage.sampling.operation.advance_presence(
                    stage.sampling.presence_history
                )

    def execute(self):
        if self.replay is None:
            self.replay = self.resources.enter_context(self.device.graph(self.run))
        self.replay()

    def store(self, requests):
        d = self.device
        for local, request in enumerate(requests):
            if not request.drafts_greedy:
                for stage, operation in enumerate(self.stages):
                    operation.copy(
                        request.proposal_rows[stage],
                        d.view(
                            operation.sampling.operation.outputs.probabilities,
                            (1, 248320),
                            local * 248320,
                        ),
                    )
        tokens = d.download(self.tokens, np.int64)
        for local, request in enumerate(requests):
            request.drafts = tokens[:, local].tolist()
            if not request.drafts_greedy:
                request.proposal_offset = (request.proposal_offset + 4) % (1 << 64)
