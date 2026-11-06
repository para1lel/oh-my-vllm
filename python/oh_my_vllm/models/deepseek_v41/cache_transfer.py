import torch

from .cache_types import _FaPage, _MambaPage, _operation, _Snapshot


class CacheTransfers:
    @staticmethod
    def _host_tensor(value):
        result = value.detach().to(device="cpu", copy=True)
        if value.device.type == "cuda":
            result = result.pin_memory()
        return result

    @staticmethod
    def _device_tensor(value, reference):
        return value.to(device=reference.device, dtype=reference.dtype, copy=True)

    def _clone_fa(self, page):
        return _FaPage(
            [self._host_tensor(value) for value in page.compressed],
            [self._host_tensor(value) for value in page.index],
            [self._host_tensor(value) for value in page.engram],
            page.valid_tokens,
        )

    def _clone_mamba(self, page):
        return _MambaPage(
            [self._host_tensor(value) for value in page.window],
            [self._host_tensor(value) for value in page.mtp_window],
            [self._host_tensor(value) for value in page.engram],
            [
                (self._host_tensor(kv), self._host_tensor(score))
                for kv, score in page.tails
            ],
            page.valid_tokens,
            page.endpoint,
        )

    @staticmethod
    def _tensor_size(value):
        return value.numel() * value.element_size()

    @staticmethod
    def _shape_size(shape, dtype):
        size = 1
        for dimension in shape:
            size *= dimension
        return size * torch.empty((), dtype=dtype).element_size()

    def _fa_page_bytes(self):
        return sum(value.page_bytes(self.block_tokens) for value in self.histories)

    def _mamba_page_bytes(self):
        return (
            sum(
                self._tensor_size(value[0])
                for value in (*self._window, *self._mtp_window)
            )
            + sum(
                self._shape_size((self.block_tokens,), value.dtype)
                for value in self._engram
            )
            + sum(
                self._tensor_size(kv[0]) + self._tensor_size(score[0])
                for kv, score in self._tails
            )
        )

    def _snapshot_size(self, fa, mamba):
        return (
            sum(
                self._tensor_size(value)
                for page in fa
                for value in (*page.compressed, *page.index, *page.engram)
            )
            + sum(
                self._tensor_size(value)
                for page in mamba
                for value in (*page.window, *page.mtp_window, *page.engram)
            )
            + sum(
                self._tensor_size(kv) + self._tensor_size(score)
                for page in mamba
                for kv, score in page.tails
            )
        )

    def _release_request_slot(self, request_id):
        if self.requests is None or self.free_slots is None:
            return
        request = self.requests.get(request_id)
        slot = request.cache_slot if request is not None else None
        if slot is not None:
            if self.release_request_slot is not None:
                self.release_request_slot(request)
            else:
                self.clear_slot(slot)
                self.free_slots.add(slot)
                request.cache_slot = None

    def _validate_store(self, operation):
        request_id = operation.get("request_id")
        if request_id is not None and self.requests is not None:
            if request_id not in self.requests:
                raise ValueError("DeepSeek cache store request is unknown")
            request = self.requests[request_id]
            if request.cache_slot is None:
                raise ValueError("DeepSeek cache store request is not resident")
        for block in operation["fa_blocks"]:
            self._validate_block_id(block, self.fa_capacity, "FA")
        for block in operation["mamba_blocks"]:
            self._validate_block_id(block, self.mamba_capacity, "mamba")
        if any(block not in self.fa_pages for block in operation["fa_blocks"]) or any(
            block not in self.mamba_pages for block in operation["mamba_blocks"]
        ):
            raise ValueError("DeepSeek store references an unknown physical page")
        return (
            len(operation["fa_blocks"]) * self._fa_page_bytes()
            + len(operation["mamba_blocks"]) * self._mamba_page_bytes()
        )

    def _store(self, operation):
        snapshot_id = operation["snapshot_id"]
        if snapshot_id in self.snapshots:
            raise ValueError("DeepSeek cache snapshot identifier repeated")
        request_id = operation.get("request_id")
        fa_blocks = operation["fa_blocks"]
        mamba_blocks = operation["mamba_blocks"]
        size = self._validate_store(operation)
        if self.host_limit == 0 or self.used_bytes + size > self.host_limit:
            return "capacity", size
        fa = [self._clone_fa(self.fa_pages[block]) for block in fa_blocks]
        mamba = [self._clone_mamba(self.mamba_pages[block]) for block in mamba_blocks]
        self.snapshots[snapshot_id] = _Snapshot(
            fa,
            mamba,
            size,
            request_id,
            "deepseek_v41_packed_fa",
            1,
        )
        self.used_bytes += size
        self.store_bytes += size
        self.completed_transfers += 1
        self._release_request_slot(operation.get("request_id"))
        return "complete", size

    def _load(self, operation, snapshot):
        self._validate_load(operation, snapshot)
        for block, source in zip(operation["fa_blocks"], snapshot.fa, strict=True):
            destination = self._fa_page(block)
            for targets, values in (
                (destination.compressed, source.compressed),
                (destination.index, source.index),
                (destination.engram, source.engram),
            ):
                for target, value in zip(targets, values, strict=True):
                    target.copy_(self._device_tensor(value, target))
            destination.valid_tokens = source.valid_tokens
        for block, source in zip(
            operation["mamba_blocks"], snapshot.mamba, strict=True
        ):
            destination = self._mamba_page(block)
            for targets, values in (
                (destination.window, source.window),
                (destination.mtp_window, source.mtp_window),
                (destination.engram, source.engram),
            ):
                for target, value in zip(targets, values, strict=True):
                    target.copy_(self._device_tensor(value, target))
            for (target_kv, target_score), (value_kv, value_score) in zip(
                destination.tails, source.tails, strict=True
            ):
                target_kv.copy_(self._device_tensor(value_kv, target_kv))
                target_score.copy_(self._device_tensor(value_score, target_score))
            destination.valid_tokens = source.valid_tokens
            destination.endpoint = source.endpoint
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.load_bytes += snapshot.bytes
        self.completed_transfers += 1
        return "complete", snapshot.bytes

    def _validate_load(self, operation, snapshot):
        if (
            snapshot.layout_id != "deepseek_v41_packed_fa"
            or snapshot.layout_version != 1
        ):
            raise ValueError(
                "DeepSeek host snapshot layout does not match the physical pool"
            )
        for block in operation["fa_blocks"]:
            self._validate_block_id(block, self.fa_capacity, "FA")
        for block in operation["mamba_blocks"]:
            self._validate_block_id(block, self.mamba_capacity, "mamba")
        request_id = operation.get("request_id")
        if (
            request_id is not None
            and self.requests is not None
            and (request_id not in self.requests)
        ):
            raise ValueError("DeepSeek cache load request is unknown")
        if snapshot.request_id is not None and request_id != snapshot.request_id:
            raise ValueError("DeepSeek cache load request does not match snapshot")
        if snapshot.request_id is not None and self.requests is not None:
            request = self.requests.get(snapshot.request_id)
            if request is None:
                raise ValueError("DeepSeek cache load request is unknown")
            if request.cache_slot is not None:
                raise ValueError("DeepSeek cache load request is already resident")
        if len(operation["fa_blocks"]) != len(snapshot.fa) or len(
            operation["mamba_blocks"]
        ) != len(snapshot.mamba):
            raise ValueError("DeepSeek cache load block counts do not match")

    def handle(self, message):
        if self.unusable:
            raise RuntimeError("DeepSeek cache is unusable after a failed batch")
        operations = message["operations"]
        seen = set(self.seen)
        snapshots = set(self.snapshots)
        for operation in operations:
            _operation(operation)
            transfer_id = operation["transfer_id"]
            if transfer_id in seen:
                raise ValueError("DeepSeek cache transfer identifier repeated")
            seen.add(transfer_id)
            request_id = operation.get("request_id")
            if (
                request_id is not None
                and self.requests is not None
                and (request_id not in self.requests)
            ):
                raise ValueError("DeepSeek cache operation request is unknown")
            snapshot_id = operation["snapshot_id"]
            if operation["kind"] == "store":
                if snapshot_id in snapshots:
                    raise ValueError("DeepSeek cache snapshot identifier repeated")
                size = self._validate_store(operation)
                if self.host_limit and self.used_bytes + size <= self.host_limit:
                    snapshots.add(snapshot_id)
            else:
                if snapshot_id not in snapshots:
                    raise ValueError("unknown DeepSeek cache snapshot")
                if operation["kind"] == "load":
                    snapshot = self.snapshots.get(snapshot_id)
                    if snapshot is None:
                        raise ValueError(
                            "DeepSeek cache batch cannot load a new snapshot"
                        )
                    self._validate_load(operation, snapshot)
            if operation["kind"] == "drop":
                snapshots.remove(snapshot_id)
        completed = []
        try:
            for operation in operations:
                transfer_id = operation["transfer_id"]
                kind = operation["kind"]
                snapshot_id = operation["snapshot_id"]
                if kind == "store":
                    status, size = self._store(operation)
                elif kind == "load":
                    status, size = self._load(operation, self.snapshots[snapshot_id])
                else:
                    snapshot = self.snapshots.pop(snapshot_id)
                    self.used_bytes -= snapshot.bytes
                    status, size = "complete", snapshot.bytes
                completed.append(
                    {
                        "transfer_id": transfer_id,
                        "status": status,
                        "bytes": size,
                    }
                )
        except BaseException:
            self.unusable = True
            raise
        self.seen.update(operation["transfer_id"] for operation in operations)
        return completed
