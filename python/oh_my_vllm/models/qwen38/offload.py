from dataclasses import dataclass
from math import prod

from ...operators.runtime.storage import WIDTHS
from .transfers import completion, validate_operation


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: str

    @classmethod
    def of(cls, tensor):
        return cls(tuple(tensor.shape), str(tensor.dtype))

    @property
    def size(self):
        return prod(self.shape) * WIDTHS[self.dtype]


@dataclass
class Snapshot:
    request_id: int | None
    request: object
    counts: tuple
    specs: tuple
    storage: object
    endpoint: bool
    proposal_rows: tuple | None
    media: tuple
    status: str = "storing"
    cancelled: bool = False
    restored: bool = False

    @property
    def size(self):
        return sum(spec.size for spec in self.specs)

    def host_views(self):
        import tvm_ffi

        offset, views = 0, []
        for spec in self.specs:
            views.append(
                tvm_ffi.from_dlpack(self.storage.array[offset : offset + spec.size])
            )
            offset += spec.size
        return tuple(views)

    def close(self):
        if self.storage is not None:
            self.storage.close()
            self.storage = None


@dataclass
class Pending:
    snapshot_id: int
    kind: str
    sources: tuple
    destinations: tuple
    slot: int | None = None
    proposals: object = None
    proposal_rows: tuple = ()
    media: tuple = ()


class Offload:
    def __init__(self, device, cache, requests, limit, engine=None, pool=None):
        self.device, self.cache, self.requests, self.limit = (
            device,
            cache,
            requests,
            limit,
        )
        if limit and engine is None:
            from ...operators.runtime.transfers import PinnedPool, TransferEngine

            engine = TransferEngine(device)
            try:
                pool = PinnedPool(device, limit, engine.quiescent)
                if capacity := limit // 256 * 256:
                    device.wait_pinned_safe()
                    reservation = pool.acquire(capacity)
                    if reservation is None:
                        raise RuntimeError(
                            "cannot reserve the host cache startup budget"
                        )
                    reservation.close()
            except BaseException:
                engine.close()
                if pool is not None:
                    pool.close()
                raise
        self.engine, self.pool = engine, pool
        self.snapshots, self.pending, self.suspended = {}, {}, {}
        self.seen = set()
        self.store_bytes = self.load_bytes = self.completed_transfers = 0
        self.closed = False

    def stats(self):
        return {
            "host_cache_capacity_bytes": self.limit,
            "host_cache_allocated_bytes": self.pool.allocated_bytes if self.pool else 0,
            "host_cache_used_bytes": self.pool.used_bytes if self.pool else 0,
            "host_cache_snapshots": len(self.snapshots),
            "host_cache_pending_transfers": len(self.pending),
            "host_cache_store_bytes": self.store_bytes,
            "host_cache_load_bytes": self.load_bytes,
            "host_cache_completed_transfers": self.completed_transfers,
        }

    def check_execute(self, request_ids):
        if any(request_id in self.suspended for request_id in request_ids):
            raise ValueError("request is not GPU resident during offload")

    def check_register(self, request_id):
        if request_id in self.requests or request_id in self.suspended:
            raise ValueError("request identifier is already owned by the worker")

    def cancel(self, request_id):
        for snapshot in self.snapshots.values():
            if snapshot.request_id == request_id:
                snapshot.cancelled = True
        snapshot_id = self.suspended.get(request_id)
        pending = any(
            pending.snapshot_id == snapshot_id for pending in self.pending.values()
        )
        if not pending:
            self.suspended.pop(request_id, None)
        return pending

    def request_sources(self, request):
        import numpy as np

        values, endpoint = [], request.state_slot is not None
        if endpoint:
            values.extend(self.cache.state_views(request.state_slot))
        rows = None
        if request.proposals is not None:
            values.append(request.proposals)
            rows = tuple(tuple(row.shape) for row in request.proposal_rows)
            if sum(prod(shape) for shape in rows) != prod(request.proposals.shape):
                raise ValueError("proposal rows do not cover their storage")
        media = []
        for index, item in enumerate(request.media):
            if item.embedding is not None and not isinstance(
                item.embedding, np.ndarray
            ):
                media.append(index)
                values.append(item.embedding)
        return tuple(values), endpoint, rows, tuple(media)

    def submit(self, transfer_id, pending):
        self.pending[transfer_id] = pending
        if not pending.sources:
            return self.finish(transfer_id)
        self.engine.submit(
            transfer_id,
            pending.destinations,
            pending.sources,
            "d2h" if pending.kind == "store" else "h2d",
        )
        return None

    def store(self, operation):
        transfer_id, snapshot_id = operation["transfer_id"], operation["snapshot_id"]
        if snapshot_id in self.snapshots:
            raise ValueError("cache snapshot identifier repeated")
        sources = self.cache.transfer_views(
            operation["fa_blocks"], operation["mamba_blocks"]
        )
        request_id, request = operation["request_id"], None
        endpoint, rows, media = False, None, ()
        if request_id is not None:
            self.check_execute([request_id])
            request = self.requests[request_id]
            extra, endpoint, rows, media = self.request_sources(request)
            sources += extra
        specs = tuple(TensorSpec.of(value) for value in sources)
        size = sum(spec.size for spec in specs)
        reserved = (size + 255) // 256 * 256
        if not self.limit or reserved > self.limit - self.pool.used_bytes:
            return completion(transfer_id, "capacity", size)
        storage = self.pool.acquire(size) if size else None
        if size and storage is None:
            status = "capacity" if self.engine.quiescent() else "deferred"
            return completion(transfer_id, status, size)
        snapshot = Snapshot(
            request_id,
            request,
            (len(operation["fa_blocks"]), len(operation["mamba_blocks"])),
            specs,
            storage,
            endpoint,
            rows,
            media,
        )
        self.snapshots[snapshot_id] = snapshot
        if request_id is not None:
            self.suspended[request_id] = snapshot_id
        return self.submit(
            transfer_id,
            Pending(snapshot_id, "store", sources, snapshot.host_views()),
        )

    def load_targets(self, snapshot, operation, destinations):
        slot, proposals, rows, media = None, None, (), []
        if snapshot.request_id is not None:
            if snapshot.endpoint:
                slot = self.cache.private_slots(snapshot.request_id, None)[0]
                destinations += self.cache.state_views(slot)
            offset = len(destinations)
            if snapshot.proposal_rows is not None:
                spec = snapshot.specs[offset]
                proposals = self.device.scratch(spec.shape, spec.dtype)
                offset += 1
                row_offset, views = 0, []
                for shape in snapshot.proposal_rows:
                    views.append(self.device.view(proposals, shape, row_offset))
                    row_offset += prod(shape)
                rows = tuple(views)
                destinations += (proposals,)
            for index in snapshot.media:
                spec = snapshot.specs[offset]
                value = self.device.scratch(spec.shape, spec.dtype)
                destinations += (value,)
                media.append((index, value))
                offset += 1
        return Pending(
            operation["snapshot_id"],
            "load",
            snapshot.host_views(),
            destinations,
            slot,
            proposals,
            rows,
            tuple(media),
        )

    def load(self, operation, snapshot):
        if snapshot.status != "host":
            raise ValueError("cache snapshot has a pending transfer")
        if operation["request_id"] != snapshot.request_id:
            raise ValueError("cache snapshot request does not match")
        if (
            len(operation["fa_blocks"]),
            len(operation["mamba_blocks"]),
        ) != snapshot.counts:
            raise ValueError("cache load block counts do not match")
        destinations = self.cache.transfer_views(
            operation["fa_blocks"], operation["mamba_blocks"]
        )
        if snapshot.request_id is not None:
            if (
                snapshot.cancelled
                or snapshot.restored
                or self.requests.get(snapshot.request_id) is not snapshot.request
            ):
                raise ValueError("cache request is cancelled or already restored")
            if snapshot.request_id in self.cache.indices:
                raise ValueError("cache load request already has a private state group")
            if snapshot.endpoint and not self.cache.free_indices:
                return completion(operation["transfer_id"], "deferred", snapshot.size)
        pending = self.load_targets(snapshot, operation, destinations)
        if (
            tuple(TensorSpec.of(value) for value in pending.destinations)
            != snapshot.specs
        ):
            raise ValueError("cache load tensor layout does not match")
        snapshot.status = "loading"
        return self.submit(operation["transfer_id"], pending)

    def finish(self, transfer_id):
        pending = self.pending.pop(transfer_id)
        snapshot = self.snapshots[pending.snapshot_id]
        request, request_id = snapshot.request, snapshot.request_id
        if request is not None:
            if pending.kind == "store":
                self.cache.release(request_id)
                request.state_slot = None
                if snapshot.proposal_rows is not None:
                    request.proposals, request.proposal_rows = None, ()
                for index in snapshot.media:
                    request.media[index].embedding = None
            elif snapshot.cancelled or self.requests.get(request_id) is not request:
                self.cache.release(request_id)
            else:
                request.state_slot = pending.slot
                request.proposals, request.proposal_rows = (
                    pending.proposals,
                    pending.proposal_rows,
                )
                for index, value in pending.media:
                    request.media[index].embedding = value
                snapshot.restored = True
                self.suspended.pop(request_id, None)
        snapshot.status = "host"
        if pending.kind == "store":
            self.store_bytes += snapshot.size
        else:
            self.load_bytes += snapshot.size
        self.completed_transfers += 1
        return completion(transfer_id, "complete", snapshot.size)

    def handle(self, message):
        if self.closed:
            raise RuntimeError("host cache is closed")
        completed = (
            [self.finish(job_id) for job_id in self.engine.poll()]
            if self.engine
            else []
        )
        for operation in message["operations"]:
            validate_operation(operation)
            transfer_id, snapshot_id = (
                operation["transfer_id"],
                operation["snapshot_id"],
            )
            if transfer_id in self.seen:
                raise ValueError("cache transfer identifier repeated")
            self.seen.add(transfer_id)
            if operation["kind"] == "store":
                result = self.store(operation)
            else:
                snapshot = self.snapshots[snapshot_id]
                if operation["kind"] == "load":
                    result = self.load(operation, snapshot)
                else:
                    if snapshot.status != "host":
                        raise ValueError(
                            "cannot drop a snapshot with a pending transfer"
                        )
                    size = snapshot.size
                    snapshot.close()
                    del self.snapshots[snapshot_id]
                    if self.suspended.get(snapshot.request_id) == snapshot_id and (
                        snapshot.cancelled
                        or snapshot.restored
                        or self.requests.get(snapshot.request_id)
                        is not snapshot.request
                    ):
                        self.suspended.pop(snapshot.request_id)
                    result = completion(transfer_id, "complete", size)
            if result is not None:
                completed.append(result)
        return {
            "type": "cache_result",
            "rpc_id": message["rpc_id"],
            "completed": completed,
            "host_cache_bytes": self.limit,
        }

    def close(self):
        if self.closed:
            return
        if self.engine is not None:
            self.engine.close()
        for snapshot in self.snapshots.values():
            snapshot.cancelled = True
        for job_id in list(self.pending):
            self.finish(job_id)
        for snapshot in self.snapshots.values():
            snapshot.close()
        self.snapshots.clear()
        self.suspended.clear()
        if self.pool is not None:
            self.pool.close()
        self.closed = True
