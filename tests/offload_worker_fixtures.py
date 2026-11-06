import codecs
import weakref
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from oh_my_vllm.models.qwen38.cache import Cache
from oh_my_vllm.models.qwen38.inference import Worker


class Tensor:
    def __init__(self, array):
        self.array = array
        self.shape, self.dtype = array.shape, array.dtype

    def view(self, dtype):
        return self.array.view(dtype)

    def copy(self):
        return self.array.copy()


class Transfers:
    def __init__(self):
        self.jobs = {}
        self.completed = []
        self.idle = True
        self.closed = False

    def submit(self, job_id, destinations, sources, direction):
        if direction not in ("h2d", "d2h"):
            raise AssertionError(direction)
        self.jobs[job_id] = [
            (weakref.ref(destination), weakref.ref(source))
            for destination, source in zip(destinations, sources, strict=True)
        ]

    def poll(self):
        result, self.completed = self.completed, []
        for job_id in result:
            for destination, source in self.jobs.pop(job_id):
                destination_value = destination()
                if isinstance(destination_value, torch.Tensor):
                    destination_value = destination_value.view(torch.uint8).numpy()
                elif isinstance(destination_value, np.ndarray):
                    destination_value = destination_value.view(np.uint8)
                else:
                    destination_value = destination_value.view(np.uint8)
                source_value = source()
                if isinstance(source_value, torch.Tensor):
                    source_value = source_value.view(torch.uint8).numpy()
                elif isinstance(source_value, np.ndarray):
                    source_value = source_value.view(np.uint8)
                else:
                    source_value = source_value.view(np.uint8)
                np.copyto(
                    destination_value.reshape(-1),
                    source_value.reshape(-1),
                )
        return result

    def quiescent(self):
        return self.idle and not self.jobs

    def close(self):
        self.completed.extend(self.jobs)
        self.poll()
        self.closed = True


class Pool:
    def __init__(self, engine, limit=1 << 20):
        self.engine, self.limit = engine, limit
        self.used_bytes = self.allocated_bytes = 0
        self.blocked = False
        self.closed = False

    def acquire(self, size):
        if self.blocked or self.used_bytes + size > self.limit:
            return None
        self.used_bytes += size
        self.allocated_bytes = max(self.allocated_bytes, self.used_bytes)
        array = np.empty(size, np.uint8)
        active = True

        def close():
            nonlocal active
            if active:
                self.used_bytes -= size
                active = False

        return SimpleNamespace(array=array, tensor=array, size=size, close=close)

    def close(self):
        if not self.engine.closed or self.used_bytes:
            raise AssertionError("transfer owners were not drained before pool close")
        self.closed = True


def worker_fixture():
    from oh_my_vllm.models.qwen38.offload import Offload

    device = SimpleNamespace(
        view=lambda value, shape, offset=0: value.reshape(-1)[
            offset : offset + int(np.prod(shape))
        ].reshape(shape),
        scratch=lambda shape, dtype: np.empty(shape, dtype),
        module=SimpleNamespace(
            free_bytes=lambda: 0,
            allocated_bytes=lambda: 0,
            reserved_bytes=lambda: 0,
        ),
    )
    cache = Cache.__new__(Cache)
    cache.device = device
    cache.blocks = cache.mamba_blocks = 8
    cache.parallel, cache.private_start, cache.slots = 1, 8, 16
    cache.indices, cache.free_indices = {7: 0}, deque()
    cache.fa = tuple(
        (np.arange(8 * 2 * 3 * 2, dtype=np.float32).reshape(8, 2, 3, 2) + layer,)
        for layer in range(3)
    )
    cache.state = tuple(
        np.arange(16 * 2 * 3, dtype=np.float32).reshape(16, 2, 3) + layer
        for layer in range(2)
    )
    cache.convolution = tuple(
        np.arange(16 * 4 * 3, dtype=np.float32).reshape(16, 4, 3) + layer
        for layer in range(2)
    )
    cache.hidden = np.arange(16 * 5, dtype=np.float32).reshape(16, 5)
    proposals = np.arange(4 * 11, dtype=np.float32).reshape(4, 11)
    media = SimpleNamespace(
        embedding=torch.tensor(
            np.arange(3 * 5, dtype=np.uint16).reshape(3, 5),
            dtype=torch.uint16,
        ),
        pixels=None,
        grid=(2, 2),
        spans=[(5, 3, 0)],
    )
    request = SimpleNamespace(
        state_slot=11,
        state_position=317,
        mtp_position=320,
        proposals=proposals,
        proposal_rows=tuple(proposals[index : index + 1] for index in range(4)),
        media=[media],
        history=[1, 2, 3],
        seed=42,
        random_offset=13,
        proposal_offset=17,
        accept_offset=19,
        drafts=[4, 5, 6, 7],
        drafts_greedy=False,
        constraint=object(),
        decoder=codecs.getincrementaldecoder("utf-8")("replace"),
        counts=np.arange(12),
        buffer="partial",
        rope_positions=np.arange(9).reshape(3, 3),
        rope_delta=9,
    )
    request.decoder.decode(b"\xe4", final=False)
    worker = Worker.__new__(Worker)
    worker.device, worker.cache, worker.requests = device, cache, {7: request}
    engine = Transfers()
    pool = Pool(engine)
    worker.offload = Offload(device, cache, worker.requests, pool.limit, engine, pool)
    return worker, request, engine, pool


def operation(transfer_id, kind, snapshot_id=1, request_id=7, fa=(1, 2), mamba=(3,)):
    return dict(
        transfer_id=transfer_id,
        snapshot_id=snapshot_id,
        kind=kind,
        request_id=request_id,
        fa_blocks=list(fa),
        mamba_blocks=list(mamba),
    )


def exchange(worker, *operations):
    return worker.handle(dict(type="cache", rpc_id=99, operations=list(operations)))
