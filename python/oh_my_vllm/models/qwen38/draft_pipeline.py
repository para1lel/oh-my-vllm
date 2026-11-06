from functools import partial
from threading import get_ident

import numpy as np
import torch

from ...operators.runtime.transfers import TransferEngine
from ...worker.metadata import Metadata


def can_pipeline(batch, active, graph, blocks):
    if not graph or len(active) < 2:
        return False
    for index in active:
        request = batch.requests[index]
        if request.generated + 4 >= request.limit:
            return False
        table = batch.metadata[2][index]
        for position in range(request.state_position, request.state_position + 4):
            page = position // 128
            if page < 0 or page >= len(table) or not 0 < table[page] <= blocks:
                return False
    return True


class DraftPipeline:
    def __init__(self, device, stages, cache):
        self.device, self.stages = device, stages
        self.count = stages[0].hidden.shape[0]
        self.cache = cache
        self.cache_rows = (
            device.view(cache[0], (cache[0].shape[0] * 256, 1024)),
            device.view(cache[1], (cache[1].shape[0] * 256, 4)),
        )
        self.saved = [
            (
                device.empty((self.count * 2, 1024), "uint8"),
                device.empty((self.count * 2, 4), "float32"),
            )
            for _ in range(3)
        ]
        self.indices = None
        self.metadata = Metadata(
            device,
            [
                *(
                    value
                    for stage in stages
                    for value in stage.metadata_inputs.host_views[:5]
                ),
                stages[0].metadata_inputs.host_views[5],
                *(np.zeros(self.count * 2, np.int32) for _ in range(3)),
            ],
            single_upload=True,
        )
        for index, stage in enumerate(stages):
            stage.metadata = self.metadata.arrays[index * 5 : index * 5 + 5]
            stage.tokens = (
                stages[index - 1].sampling.operation.outputs.tokens
                if index
                else self.metadata.arrays[20]
            )
            stage.cache = cache
            stage.metadata_inputs.close()
            stage.metadata_inputs = None
            if index:
                stage.previous_hidden = stages[
                    index - 1
                ].sampling.operation.outputs.hidden
        self.engine = TransferEngine(device)
        self.storage = device.upload_pool.acquire(self.count * 8)
        self.host_tokens = self.storage.tensor[: self.count * 8].view(torch.int64)

    def close(self):
        if self.storage is None:
            return
        self.engine.close()
        for stage in self.stages:
            stage.close()
        self.metadata.close()
        self.host_tokens = None
        self.device.upload_pool.release(self.storage)
        self.storage = None

    def prepare(self, batch, active, endpoints):
        requests = [batch.requests[index] for index in active]
        first_rows = self.stages[0].row_views()[0]
        for local, index in enumerate(active):
            self.stages[0].copy(first_rows[local], endpoints[index][0])
        arrays, indices = [], []
        tables = np.asarray([batch.metadata[2][index] for index in active], np.int32)
        starts = np.arange(self.count + 1, dtype=np.int32)
        for number in range(4):
            positions = [request.state_position + number for request in requests]
            slots = np.asarray(
                [
                    batch.metadata[2][index][position // 128] * 128 + position % 128
                    for index, position in zip(active, positions, strict=True)
                ],
                np.int32,
            )
            arrays.extend(
                (
                    np.asarray(
                        [
                            request.position(position)
                            for request, position in zip(
                                requests, positions, strict=True
                            )
                        ],
                        np.int64,
                    ).T.copy(),
                    slots,
                    tables,
                    starts,
                    np.asarray(positions, np.int32) + 1,
                )
            )
            if number:
                keys = slots // 128 * 256 + slots % 128
                indices.append(np.concatenate((keys, keys + 128)))
        arrays.append(
            np.asarray([request.history[-1] for request in requests], np.int64)
        )
        arrays.extend(indices)
        self.metadata.update(arrays)
        self.indices = indices
        self.snapshots = set()

    def launch(self, number, graph):
        stage = self.stages[number]
        if number:
            for destination, source in zip(
                self.saved[number - 1], self.cache_rows, strict=True
            ):
                self.device.module.gather_rows(
                    destination, source, self.metadata.arrays[20 + number]
                )
            self.snapshots.add(number)
        prepared = stage.runs and stage.thread == get_ident()
        stage.execute(graph if prepared else False)

    def restore(self, number):
        if number not in self.snapshots:
            return
        for saved, cache, width in zip(
            self.saved[number - 1], self.cache_rows, (1024, 4), strict=True
        ):
            for index, row in enumerate(self.indices[number - 1]):
                destination = self.device.view(cache, (1, width), int(row) * width)
                source = self.device.view(saved, (1, width), index * width)
                self.device.copy(destination, source)
        self.snapshots.remove(number)

    def readback(self, value, next_stage, graph):
        self.engine.submit(0, (self.host_tokens,), (value,), "d2h")
        try:
            if next_stage < 4:
                self.launch(next_stage, graph)
        finally:
            complete = self.engine.pending[0].complete
            self.device.checked(self.device.runtime.cudaEventSynchronize(complete))
            if self.engine.poll() != [0]:
                raise RuntimeError("draft token readback did not complete")
        return self.host_tokens

    def execute(self, batch, active, graph, record):
        requests = [batch.requests[index] for index in active]
        self.launch(0, graph)
        for number, stage in enumerate(self.stages):
            offsets = [request.proposal_offset for request in requests]
            prefixes = [request.drafts.copy() for request in requests]
            try:
                chosen, _ = stage.sampling.select_from_logits(
                    requests,
                    graph,
                    prefixes=prefixes,
                    stream="proposal",
                    readback=partial(self.readback, next_stage=number + 1, graph=graph),
                )
                record(batch, active, number)
                previous, remaining = {}, []
                views = stage.row_views()
                for local, index in enumerate(active):
                    request = batch.requests[index]
                    if not request.drafts_greedy:
                        stage.copy(request.proposal_rows[number], views[1][local])
                    request.drafts.append(int(chosen[local]))
                    previous[index] = views[2][local]
                    if (
                        not request.stops_after(request.drafts)
                        and request.generated + number + 2 < request.limit
                    ):
                        remaining.append(index)
                if remaining != active:
                    self.restore(number + 1)
                    return number + 1, remaining, previous
            except BaseException:
                for request, offset, prefix in zip(
                    requests, offsets, prefixes, strict=True
                ):
                    request.proposal_offset = offset
                    request.drafts = prefix
                self.restore(number)
                self.restore(number + 1)
                raise
        return 4, [], {}
