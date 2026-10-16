from contextlib import ExitStack

import numpy as np
import tvm_ffi


def can_chain(batch, active, graph, blocks):
    if not graph or not active:
        return False
    for index in active:
        request = batch.requests[index]
        parameters = request.parameters()
        if (
            parameters[0] != 0
            or np.any(parameters[4:6] != 0)
            or parameters[6] != 1
            or request.thinking
            or request.constraint.matcher is not None
            or request.options.get("sampling", {}).get("repetition_guard", True)
            or request.options.get("stop")
            or request.generated + 4 > request.minimum
            or request.generated + 4 >= request.limit
        ):
            return False
        table = batch.scheduled[index]["fa_block_table"]
        for position in range(request.state_position, request.state_position + 4):
            page = position // 784
            if page < 0 or page >= len(table) or not 0 < table[page] < blocks:
                return False
    return True


class DraftChain:
    def __init__(self, device, stages, count):
        self.device, self.stages = device, stages
        self.resources = ExitStack()
        self.replay = None
        self.tokens = device.empty((4, count), "int64")
        self.token_rows = [
            device.view(self.tokens, (count,), stage * count) for stage in range(4)
        ]
        for index, stage in enumerate(stages):
            stage.sampling.value, stage.sampling.residual = stage.ffn.buffers[:2]
            stage.sampling.greedy = True
            if index:
                stage.tokens = stages[index - 1].sampling.operation.buffers[7]
                sampler = stages[0].sampling
                stage.sampling.history = sampler.history
                stage.sampling.mask = sampler.mask
                stage.sampling.parameters = sampler.parameters
                stage.sampling.rng = sampler.rng

    def close(self):
        self.resources.close()
        for stage in self.stages:
            stage.close()

    def prepare(self, batch, active, endpoints, cache):
        d = self.device
        requests = [batch.requests[index] for index in active]
        count = len(active)
        for local, index in enumerate(active):
            self.stages[0].copy(
                d.view(self.stages[0].hidden, (1, 5120), local * 5120),
                endpoints[index][0],
            )
        sampler = self.stages[0].sampling
        # Eligibility makes the mask independent of the unknown draft prefix.
        sampler.inputs.update(requests, [[] for _ in requests], "proposal")
        for destination, source in (
            (sampler.mask, sampler.inputs.mask),
            (sampler.parameters, sampler.inputs.parameters),
            (sampler.rng, sampler.inputs.random),
        ):
            self.stages[0].copy(destination, tvm_ffi.from_dlpack(source))
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
                np.asarray([request.history[-1] for request in requests], np.int64),
            )
            stage.metadata_inputs.update(arrays)
            stage.cache = cache.fa[-1]

    def run(self):
        for index, stage in enumerate(self.stages):
            if index:
                stage.copy(
                    stage.hidden,
                    self.stages[index - 1].sampling.operation.buffers[0],
                )
            stage.run()
            stage.sampling.run()
            stage.copy(self.token_rows[index], stage.sampling.operation.buffers[7])

    def execute(self):
        if self.replay is None:
            self.run()
            self.device.synchronize()
            self.replay = self.resources.enter_context(self.device.graph(self.run))
        self.replay()

    def store(self, requests):
        d = self.device
        tokens = d.download(self.tokens, np.int64)
        for local, request in enumerate(requests):
            request.drafts = tokens[:, local].tolist()
            for index, stage in enumerate(self.stages):
                stage.copy(
                    d.view(request.proposals, (1, 248320), index * 248320),
                    d.view(
                        stage.sampling.operation.buffers[6],
                        (1, 248320),
                        local * 248320,
                    ),
                )
