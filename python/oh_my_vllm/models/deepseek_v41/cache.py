"""Paged physical state for the DeepSeek-V4.1 model adapter."""

from dataclasses import dataclass
from math import ceil

import torch

from .state import StateExports
from .storage import release_sequence_slot, reserve_sequence_slot


def _blocks(value):
    if not isinstance(value, list) or any(
        type(block) is not int or not 0 < block <= 0xFFFFFFFF for block in value
    ):
        raise ValueError("invalid DeepSeek cache block list")
    if len(value) != len(set(value)):
        raise ValueError("DeepSeek cache blocks must be unique")
    return value


def _table(value):
    if not isinstance(value, list) or any(
        type(block) is not int or not 0 <= block <= 0xFFFFFFFF for block in value
    ):
        raise ValueError("invalid DeepSeek cache table")
    return value


def _operation(operation):
    for name in ("transfer_id", "snapshot_id"):
        if (
            type(operation.get(name)) is not int
            or not 0 <= operation[name] <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError(f"invalid DeepSeek cache {name}")
    if operation.get("request_id") is not None and (
        type(operation["request_id"]) is not int
        or not 0 <= operation["request_id"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("invalid DeepSeek cache request_id")
    if operation.get("kind") not in ("store", "load", "drop"):
        raise ValueError("invalid DeepSeek cache operation")
    _blocks(operation.get("fa_blocks"))
    _blocks(operation.get("mamba_blocks"))
    if operation["kind"] == "drop" and (
        operation["fa_blocks"] or operation["mamba_blocks"]
    ):
        raise ValueError("DeepSeek cache drop cannot contain blocks")


@dataclass
class _FaPage:
    compressed: list[torch.Tensor]
    index: list[torch.Tensor]
    engram: list[torch.Tensor]
    valid_tokens: int = 0


@dataclass
class _MambaPage:
    window: list[torch.Tensor]
    mtp_window: list[torch.Tensor]
    engram: list[torch.Tensor]
    tails: list[tuple[torch.Tensor, torch.Tensor]]
    valid_tokens: int = 0
    endpoint: int = 0


@dataclass
class _Snapshot:
    fa: list[_FaPage]
    mamba: list[_MambaPage]
    bytes: int
    request_id: int | None


class DeepSeekCache:
    """Keep model state in logical pages and support Rust cache transfers.

    The model uses per-request staging slots. Logical pages make a request slot
    replaceable after preemption, host offload, or prefix reuse.
    """

    def __init__(
        self,
        model,
        *,
        block_tokens=128,
        max_seq_len=None,
        host_limit=0,
        fa_capacity=None,
        mamba_capacity=None,
        requests=None,
        free_slots=None,
    ):
        if type(block_tokens) is not int or block_tokens < 1:
            raise ValueError("DeepSeek cache block size must be positive")
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("DeepSeek cache context is too small")
        if type(host_limit) is not int or host_limit < 0:
            raise ValueError("DeepSeek host cache limit must be nonnegative")
        self.model = model
        self.block_tokens = block_tokens
        self.max_seq_len = max_seq_len
        self.host_limit = host_limit
        self.fa_capacity = fa_capacity
        self.mamba_capacity = mamba_capacity
        self.requests = requests
        self.free_slots = free_slots
        self.fa_pages = {}
        self.mamba_pages = {}
        self.snapshots = {}
        self.seen = set()
        self.used_bytes = 0
        self.store_bytes = 0
        self.load_bytes = 0
        self.completed_transfers = 0
        self.unusable = False
        self._window = []
        self._mtp_window = []
        self._compressed = []
        self._index = []
        self._engram = []
        self._tails = []
        self.exports = StateExports()
        model._state_export = self.exports
        for layer in getattr(model, "layers", ()):
            attention = getattr(layer, "attn", None)
            if attention is None:
                continue
            window = getattr(attention, "window_kv_cache", None)
            if window is not None:
                self._window.append(window)
                attention._state_export = self.exports
        for layer in getattr(model, "mtp", ()):
            attention = getattr(layer, "attn", None)
            window = getattr(attention, "window_kv_cache", None)
            if window is not None:
                self._mtp_window.append(window)
                attention._state_export = self.exports
        for layer in getattr(model, "layers", ()):
            attention = getattr(layer, "attn", None)
            if attention is None:
                continue
            compressed = getattr(attention, "compress_kv_cache", None)
            if compressed is not None:
                self._compressed.append(
                    (compressed, int(getattr(attention, "compress_ratio", 1)))
                )
            indexer = getattr(attention, "indexer", None)
            index = getattr(indexer, "k_cache", None)
            if index is not None:
                self._index.append(
                    (index, int(getattr(attention, "compress_ratio", 1)))
                )
            compressor = getattr(attention, "compressor", None)
            kv_state = getattr(compressor, "kv_state", None)
            score_state = getattr(compressor, "score_state", None)
            if kv_state is not None and score_state is not None:
                self._tails.append((kv_state, score_state))
                compressor._state_export = self.exports
        engram = getattr(getattr(model, "engram_hash", None), "cache", None)
        if engram is not None:
            self._engram.append(engram)

    @property
    def host_allocated_bytes(self):
        return self.used_bytes

    def stats(self):
        staging = [
            value
            for value in (
                *(value for value, _ in self._compressed),
                *(value for value, _ in self._index),
                *self._engram,
            )
            if hasattr(value, "_paged_storage")
        ]
        mapped = 0
        if staging:
            from ...operators.runtime.paged import mapped_bytes

            mapped = mapped_bytes()
        return {
            "sequence_staging_mapped_bytes": mapped,
            "sequence_staging_logical_bytes": sum(
                value.numel() * value.element_size() for value in staging
            ),
            "gpu_logical_page_bytes": (
                len(self.fa_pages) * self._fa_page_bytes()
                + len(self.mamba_pages) * self._mamba_page_bytes()
            ),
            "host_cache_capacity_bytes": self.host_limit,
            "host_cache_allocated_bytes": self.used_bytes,
            "host_cache_used_bytes": self.used_bytes,
            "host_cache_snapshots": len(self.snapshots),
            "host_cache_pending_transfers": 0,
            "host_cache_store_bytes": self.store_bytes,
            "host_cache_load_bytes": self.load_bytes,
            "host_cache_completed_transfers": self.completed_transfers,
        }

    def _new_fa_page(self):
        return _FaPage(
            [
                torch.zeros(
                    (ceil(self.block_tokens / ratio), *value.shape[2:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value, ratio in self._compressed
            ],
            [
                torch.zeros(
                    (ceil(self.block_tokens / ratio), *value.shape[2:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value, ratio in self._index
            ],
            [
                torch.zeros(
                    (self.block_tokens,), dtype=value.dtype, device=value.device
                )
                for value in self._engram
            ],
        )

    def _new_mamba_page(self):
        return _MambaPage(
            [torch.zeros_like(value[0]) for value in self._window],
            [torch.zeros_like(value[0]) for value in self._mtp_window],
            [
                torch.zeros(
                    (self.block_tokens,), dtype=value.dtype, device=value.device
                )
                for value in self._engram
            ],
            [
                (torch.zeros_like(kv[0]), torch.full_like(score[0], -torch.inf))
                for kv, score in self._tails
            ],
        )

    def _fa_page(self, block):
        page = self.fa_pages.get(block)
        if page is None:
            page = self._new_fa_page()
            self.fa_pages[block] = page
        return page

    def _mamba_page(self, block):
        page = self.mamba_pages.get(block)
        if page is None:
            page = self._new_mamba_page()
            self.mamba_pages[block] = page
        return page

    def _validate_block_id(self, block, capacity, name):
        if (
            type(block) is not int
            or block < 0
            or (capacity is not None and block >= capacity)
        ):
            raise ValueError(f"invalid DeepSeek {name} block")

    def _validate_row(self, row, *, end=None):
        fa = _table(row.get("fa_block_table"))
        mamba = _table(row.get("mamba_block_table"))
        for block in fa:
            self._validate_block_id(block, self.fa_capacity, "FA")
        for block in mamba:
            self._validate_block_id(block, self.mamba_capacity, "mamba")
        if end is not None and end > 0:
            last = (end - 1) // self.block_tokens
            if len(fa) <= last or len(mamba) <= last:
                raise ValueError("DeepSeek cache table is shorter than the request")
            if any(block == 0 for block in fa[: last + 1]):
                raise ValueError("DeepSeek FA table has an unusable block")
        return fa, mamba

    def ensure_row(self, row):
        fa, mamba = self._validate_row(row)
        for block in fa:
            if block:
                self._fa_page(block)
        for block in mamba:
            if block:
                self._mamba_page(block)

    def ensure_mamba_row(self, row):
        _fa, mamba = self._validate_row(row)
        for block in mamba:
            if block:
                self._mamba_page(block)

    def clear_slot(self, slot):
        self.exports.discard(slot)
        for value in self._window:
            value[slot].zero_()
        for value in self._mtp_window:
            value[slot].zero_()
        for value, _ratio in self._compressed:
            release_sequence_slot(value, slot)
        for value, _ratio in self._index:
            release_sequence_slot(value, slot)
        for value in self._engram:
            release_sequence_slot(value, slot)
        for kv, score in self._tails:
            kv[slot].zero_()
            score[slot].fill_(-torch.inf)

    def reserve_slot(self, slot, end):
        for value, ratio in (*self._compressed, *self._index):
            reserve_sequence_slot(value, slot, end // ratio)
        for value in self._engram:
            reserve_sequence_slot(value, slot, end)

    def _state_destinations(self, row, start_pos, end_pos, mamba_blocks):
        destinations = {}
        for index, block in enumerate(mamba_blocks):
            endpoint = (index + 1) * self.block_tokens
            if block and start_pos < endpoint <= end_pos:
                destinations[block] = endpoint
        tail = mamba_blocks[(end_pos - 1) // self.block_tokens]
        if not tail:
            raise ValueError("DeepSeek accepted endpoint has no state page")
        destinations[tail] = end_pos
        checkpoint = row.get("cache_checkpoint")
        if checkpoint is not None:
            if (
                not isinstance(checkpoint, (list, tuple))
                or len(checkpoint) != 2
                or type(checkpoint[0]) is not int
                or not 0 < checkpoint[0] <= self.max_seq_len
            ):
                raise ValueError("invalid DeepSeek cache checkpoint")
            endpoint, block = checkpoint
            self._validate_block_id(block, self.mamba_capacity, "checkpoint")
            if not block:
                raise ValueError("DeepSeek checkpoint must name a physical page")
            if start_pos < endpoint <= end_pos:
                if block in destinations and destinations[block] != endpoint:
                    raise ValueError("DeepSeek state page has conflicting endpoints")
                destinations[block] = endpoint
        return destinations

    def begin_step(self, row, slot, *, start_pos, end_pos, minimum_end=None):
        if not 0 <= start_pos < end_pos <= self.max_seq_len:
            raise ValueError("DeepSeek state export range is out of bounds")
        _fa, mamba = self._validate_row(row, end=end_pos)
        minimum_end = end_pos if minimum_end is None else minimum_end
        if type(minimum_end) is not int or not start_pos < minimum_end <= end_pos:
            raise ValueError("invalid DeepSeek minimum accepted endpoint")
        if end_pos - minimum_end > 5:
            raise ValueError("DeepSeek verification exceeds five draft tokens")
        endpoints = set(
            self._state_destinations(row, start_pos, end_pos, mamba).values()
        )
        endpoints.update(range(minimum_end, end_pos + 1))
        self.exports.begin(slot, start_pos, endpoints, minimum_end=minimum_end)

    def mtp_checkpoint_ends(self, slots, first, last, final_end):
        return sorted(
            {
                end
                for slot in slots
                for end in self.exports.endpoints(slot)
                if first < end <= last
                and end < final_end
                and end < self.exports.steps[slot].minimum_end
            }
        )

    def _captured_state(self, slot, endpoint, accepted_end, include_mtp):
        if slot in self.exports.steps:
            return (
                [
                    self.exports.window_state(slot, endpoint, value)
                    for value in self._window
                ],
                [
                    self.exports.window_state(slot, endpoint, value)
                    for value in self._mtp_window
                ]
                if include_mtp
                else (),
                [
                    self.exports.tail_state(slot, endpoint, kv)
                    for kv, _score in self._tails
                ],
            )
        if endpoint != accepted_end:
            raise ValueError("DeepSeek intermediate endpoint was not exported")
        return (
            [value[slot] for value in self._window],
            [value[slot] for value in self._mtp_window] if include_mtp else (),
            [(kv[slot], score[slot]) for kv, score in self._tails],
        )

    def commit(
        self, request_id, row, slot, *, start_pos=0, accepted_end, include_mtp=False
    ):
        if type(start_pos) is not int or type(accepted_end) is not int:
            raise ValueError("DeepSeek cache positions must be integers")
        if not 0 <= start_pos <= accepted_end <= self.max_seq_len:
            raise ValueError("DeepSeek cache positions are out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=accepted_end)
        if accepted_end == 0:
            self.exports.discard(slot)
            return
        destinations = self._state_destinations(
            row, start_pos, accepted_end, mamba_blocks
        )
        prefix_state = row.get("prefix_state_slot")
        if prefix_state is not None and start_pos:
            self._validate_block_id(prefix_state, self.mamba_capacity, "prefix")
            prefix = self.mamba_pages.get(prefix_state)
            if prefix is None or prefix.endpoint != start_pos:
                raise ValueError(
                    "DeepSeek prefix state endpoint differs from the schedule"
                )
            if prefix_state in destinations and destinations[prefix_state] != start_pos:
                raise ValueError(
                    "DeepSeek prefix state requires a private continuation"
                )
        states = {
            end: self._captured_state(slot, end, accepted_end, include_mtp)
            for end in set(destinations.values())
        }
        first_page = start_pos // self.block_tokens
        last_page = (accepted_end - 1) // self.block_tokens
        for page_index in range(first_page, last_page + 1):
            page_start = page_index * self.block_tokens
            page_end = min(accepted_end, page_start + self.block_tokens)
            block = fa_blocks[page_index]
            if not block:
                continue
            page = self._fa_page(block)
            for sources, targets in (
                (self._compressed, page.compressed),
                (self._index, page.index),
            ):
                for (value, ratio), destination in zip(sources, targets, strict=True):
                    first = max(start_pos // ratio, page_start // ratio)
                    last = min(accepted_end // ratio, ceil(page_end / ratio))
                    if first < last:
                        offset = page_start // ratio
                        destination[first - offset : last - offset].copy_(
                            value[slot, first:last]
                        )
            for value, destination in zip(self._engram, page.engram, strict=True):
                first = max(start_pos, page_start)
                if first < page_end:
                    destination[first - page_start : page_end - page_start].copy_(
                        value[slot, first:page_end]
                    )
            page.valid_tokens = max(page.valid_tokens, page_end - page_start)
        for block, endpoint in destinations.items():
            page = self._mamba_page(block)
            window, mtp_window, tails = states[endpoint]
            for source, destination in zip(window, page.window, strict=True):
                destination.copy_(source)
            if include_mtp:
                for source, destination in zip(
                    mtp_window, page.mtp_window, strict=True
                ):
                    destination.copy_(source)
            for (source_kv, source_score), (target_kv, target_score) in zip(
                tails, page.tails, strict=True
            ):
                target_kv.copy_(source_kv)
                target_score.copy_(source_score)
            page_start = (endpoint - 1) // self.block_tokens * self.block_tokens
            for source, destination in zip(self._engram, page.engram, strict=True):
                destination.zero_()
                destination[: endpoint - page_start].copy_(
                    source[slot, page_start:endpoint]
                )
        for block, endpoint in destinations.items():
            page = self.mamba_pages[block]
            page.endpoint = endpoint
            page.valid_tokens = (
                endpoint - (endpoint - 1) // self.block_tokens * self.block_tokens
            )
        self.exports.discard(slot)

    def restore(self, request_id, slot, row, *, start_pos, use_prefix_state=True):
        if type(start_pos) is not int or not 0 <= start_pos <= self.max_seq_len:
            raise ValueError("DeepSeek restore position is out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=start_pos)
        state_source = row.get("prefix_state_slot") if use_prefix_state else None
        if state_source is not None:
            self._validate_block_id(state_source, self.mamba_capacity, "prefix")
            if state_source == 0:
                raise ValueError("DeepSeek prefix state must name a physical page")
        state = None
        if start_pos:
            last_page = (start_pos - 1) // self.block_tokens
            if state_source is None:
                state_source = next(
                    (
                        block
                        for block in reversed(mamba_blocks[: last_page + 1])
                        if block
                    ),
                    None,
                )
            state = self.mamba_pages.get(state_source)
            if state is None:
                raise ValueError("DeepSeek restore state page is unavailable")
            if state.endpoint != start_pos:
                raise ValueError(
                    "DeepSeek restore state endpoint differs from the schedule"
                )
            for index, block in enumerate(fa_blocks[: last_page + 1]):
                if block:
                    page = self.fa_pages.get(block)
                    required = min(
                        self.block_tokens, start_pos - index * self.block_tokens
                    )
                    if page is None or page.valid_tokens < required:
                        raise ValueError(
                            "DeepSeek restore FA page is unavailable or incomplete"
                        )
        self.clear_slot(slot)
        if not start_pos:
            return
        self.reserve_slot(slot, start_pos)
        for page_index, block in enumerate(fa_blocks):
            page_start = page_index * self.block_tokens
            if page_start >= start_pos:
                break
            if not block:
                continue
            page = self.fa_pages[block]
            page_end = min(start_pos, page_start + self.block_tokens)
            for targets, sources in (
                (self._compressed, page.compressed),
                (self._index, page.index),
            ):
                for (value, ratio), source in zip(targets, sources, strict=True):
                    first = page_start // ratio
                    last = min(start_pos // ratio, ceil(page_end / ratio))
                    if first < last:
                        value[slot, first:last].copy_(source[: last - first])
            for value, source in zip(self._engram, page.engram, strict=True):
                value[slot, page_start:page_end].copy_(source[: page_end - page_start])
        for targets, sources in (
            (self._window, state.window),
            (self._mtp_window, state.mtp_window),
        ):
            for value, source in zip(targets, sources, strict=True):
                value[slot].copy_(source)
        for (kv, score), (source_kv, source_score) in zip(
            self._tails, state.tails, strict=True
        ):
            kv[slot].copy_(source_kv)
            score[slot].copy_(source_score)
        page_start = (start_pos - 1) // self.block_tokens * self.block_tokens
        for value, source in zip(self._engram, state.engram, strict=True):
            value[slot, page_start:start_pos].copy_(source[: start_pos - page_start])

    def copy_prefix(self, copies):
        for source, destination, length in copies:
            if (
                type(source) is not int
                or type(destination) is not int
                or type(length) is not int
                or source <= 0
                or destination <= 0
                or source == destination
                or not 0 < length <= self.block_tokens
            ):
                raise ValueError("invalid DeepSeek FA prefix copy")
            self._validate_block_id(source, self.fa_capacity, "FA source")
            self._validate_block_id(destination, self.fa_capacity, "FA destination")
            source_page = self.fa_pages.get(source)
            if source_page is None or source_page.valid_tokens < length:
                raise ValueError(
                    "DeepSeek prefix copy references an unknown or incomplete page"
                )
            destination_page = self._fa_page(destination)
            for spec, sources, targets in (
                (self._compressed, source_page.compressed, destination_page.compressed),
                (self._index, source_page.index, destination_page.index),
            ):
                for (_value, ratio), source_value, target in zip(
                    spec, sources, targets, strict=True
                ):
                    count = min(target.shape[0], ceil(length / ratio))
                    target[:count].copy_(source_value[:count])
            for source_value, target in zip(
                source_page.engram, destination_page.engram, strict=True
            ):
                target[:length].copy_(source_value[:length])
            destination_page.valid_tokens = length

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
        return sum(
            self._shape_size(
                (ceil(self.block_tokens / ratio), *value.shape[2:]), value.dtype
            )
            for value, ratio in (*self._compressed, *self._index)
        ) + sum(
            self._shape_size((self.block_tokens,), value.dtype)
            for value in self._engram
        )

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
        slot = getattr(request, "cache_slot", None) if request is not None else None
        if slot is not None:
            self.clear_slot(slot)
            self.free_slots.add(slot)
            request.cache_slot = None

    def _validate_store(self, operation):
        request_id = operation.get("request_id")
        if request_id is not None and self.requests is not None:
            if request_id not in self.requests:
                raise ValueError("DeepSeek cache store request is unknown")
            request = self.requests[request_id]
            if getattr(request, "cache_slot", None) is None:
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
            if getattr(request, "cache_slot", None) is not None:
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

    def close(self):
        self.exports.clear()
        for value in (
            *(value for value, _ in self._compressed),
            *(value for value, _ in self._index),
            *self._engram,
        ):
            owner = getattr(value, "_paged_storage", None)
            if owner is not None:
                for slot in range(value.shape[0]):
                    owner.release(slot)
        self.snapshots.clear()
        self.fa_pages.clear()
        self.mamba_pages.clear()
        self.used_bytes = 0
