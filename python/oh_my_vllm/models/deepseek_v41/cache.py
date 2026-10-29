"""Paged physical state for the DeepSeek-V4.1 model adapter."""

from dataclasses import dataclass
from math import ceil

import torch

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
    window: list[torch.Tensor]
    mtp_window: list[torch.Tensor]
    compressed: list[torch.Tensor]
    index: list[torch.Tensor]
    engram: list[torch.Tensor]
    tails: list[tuple[torch.Tensor, torch.Tensor]]
    valid_tokens: int = 0


@dataclass
class _MambaPage:
    engram: list[torch.Tensor]
    tails: list[tuple[torch.Tensor, torch.Tensor]]
    valid_tokens: int = 0


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
        for layer in getattr(model, "layers", ()):
            attention = getattr(layer, "attn", None)
            if attention is None:
                continue
            window = getattr(attention, "window_kv_cache", None)
            if window is not None:
                self._window.append(window)
        for layer in getattr(model, "mtp", ()):
            attention = getattr(layer, "attn", None)
            window = getattr(attention, "window_kv_cache", None)
            if window is not None:
                self._mtp_window.append(window)
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
                    (self.block_tokens, value.shape[-1]),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value in self._window
            ],
            [
                torch.zeros(
                    (self.block_tokens, value.shape[-1]),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value in self._mtp_window
            ],
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
                    (self.block_tokens,),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value in self._engram
            ],
            [
                (
                    torch.zeros_like(kv[0]),
                    torch.full_like(score[0], -torch.inf),
                )
                for kv, score in self._tails
            ],
        )

    def _new_mamba_page(self):
        return _MambaPage(
            [
                torch.zeros(
                    (self.block_tokens,),
                    dtype=value.dtype,
                    device=value.device,
                )
                for value in self._engram
            ],
            [
                (
                    torch.zeros_like(kv[0]),
                    torch.full_like(score[0], -torch.inf),
                )
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

    def commit(
        self,
        request_id,
        row,
        slot,
        *,
        start_pos=0,
        accepted_end,
        include_mtp=False,
    ):
        if type(start_pos) is not int or type(accepted_end) is not int:
            raise ValueError("DeepSeek cache positions must be integers")
        if not 0 <= start_pos <= accepted_end <= self.max_seq_len:
            raise ValueError("DeepSeek cache positions are out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=accepted_end)
        if accepted_end == 0:
            return
        first_page = start_pos // self.block_tokens
        last_page = (accepted_end - 1) // self.block_tokens
        for page_index in range(first_page, last_page + 1):
            page_start = page_index * self.block_tokens
            page_end = min(accepted_end, page_start + self.block_tokens)
            block = fa_blocks[page_index]
            if block == 0:
                continue
            page = self._fa_page(block)
            page.valid_tokens = max(page.valid_tokens, page_end - page_start)
            for value, destination in zip(self._window, page.window, strict=True):
                positions = torch.arange(
                    max(start_pos, page_start),
                    page_end,
                    device=value.device,
                )
                offsets = positions - page_start
                source = value[slot, positions.remainder(value.shape[1])]
                destination[offsets] = source
            if include_mtp:
                for value, destination in zip(
                    self._mtp_window, page.mtp_window, strict=True
                ):
                    positions = torch.arange(
                        max(start_pos, page_start),
                        page_end,
                        device=value.device,
                    )
                    offsets = positions - page_start
                    source = value[slot, positions.remainder(value.shape[1])]
                    destination[offsets] = source
            for (value, ratio), destination in zip(
                self._compressed, page.compressed, strict=True
            ):
                first_row = max(start_pos // ratio, page_start // ratio)
                last_row = min(accepted_end // ratio, (page_end + ratio - 1) // ratio)
                if first_row < last_row:
                    source = value[slot, first_row:last_row]
                    destination[
                        first_row - page_start // ratio : last_row - page_start // ratio
                    ] = source
            for (value, ratio), destination in zip(
                self._index, page.index, strict=True
            ):
                first_row = max(start_pos // ratio, page_start // ratio)
                last_row = min(accepted_end // ratio, (page_end + ratio - 1) // ratio)
                if first_row < last_row:
                    source = value[slot, first_row:last_row]
                    destination[
                        first_row - page_start // ratio : last_row - page_start // ratio
                    ] = source
            for value, destination in zip(self._engram, page.engram, strict=True):
                first = max(start_pos, page_start)
                last = min(accepted_end, page_end)
                if first < last:
                    destination[first - page_start : last - page_start] = value[
                        slot, first:last
                    ]
        prefix_state = row.get("prefix_state_slot")
        if prefix_state is not None and start_pos:
            self._validate_block_id(prefix_state, self.mamba_capacity, "prefix")
            if prefix_state == 0:
                raise ValueError("DeepSeek prefix state must name a physical page")
            prefix_page = self.mamba_pages.get(prefix_state)
            if prefix_page is None:
                raise ValueError("DeepSeek prefix state page is unavailable")
            page_index = (start_pos - 1) // self.block_tokens
            destination_block = mamba_blocks[page_index]
            if destination_block == 0:
                raise ValueError("DeepSeek prefix destination has no physical page")
            destination_page = self._mamba_page(destination_block)
            page_start = page_index * self.block_tokens
            prefix_length = start_pos - page_start
            if prefix_page.valid_tokens < prefix_length:
                raise ValueError("DeepSeek prefix state page is incomplete")
            for source, destination in zip(
                prefix_page.engram, destination_page.engram, strict=True
            ):
                destination[:prefix_length].copy_(source[:prefix_length])
            for (source_kv, source_score), (
                destination_kv,
                destination_score,
            ) in zip(prefix_page.tails, destination_page.tails, strict=True):
                destination_kv.copy_(source_kv)
                destination_score.copy_(source_score)
            destination_page.valid_tokens = max(
                destination_page.valid_tokens, prefix_length
            )
        for page_index in range(first_page, last_page + 1):
            page_start = page_index * self.block_tokens
            page_end = min(accepted_end, page_start + self.block_tokens)
            block = mamba_blocks[page_index]
            if block == 0:
                continue
            page = self._mamba_page(block)
            page.valid_tokens = max(page.valid_tokens, page_end - page_start)
            for value, destination in zip(self._engram, page.engram, strict=True):
                first = max(start_pos, page_start)
                last = min(accepted_end, page_end)
                if first < last:
                    destination[first - page_start : last - page_start] = value[
                        slot, first:last
                    ]
        tail_block = next(
            (
                block
                for block in reversed(
                    mamba_blocks[: (accepted_end - 1) // self.block_tokens + 1]
                )
                if block
            ),
            None,
        )
        checkpoint = row.get("cache_checkpoint")
        checkpoint_block = None
        if checkpoint is not None:
            if (
                not isinstance(checkpoint, (list, tuple))
                or len(checkpoint) != 2
                or type(checkpoint[0]) is not int
                or not 0 <= checkpoint[0] <= self.max_seq_len
            ):
                raise ValueError("invalid DeepSeek cache checkpoint")
            checkpoint_block = checkpoint[1]
            self._validate_block_id(checkpoint_block, self.mamba_capacity, "checkpoint")
            if checkpoint_block == 0:
                raise ValueError("DeepSeek checkpoint must name a physical page")
            if checkpoint[0] > accepted_end:
                checkpoint_block = None
        for block in (tail_block, checkpoint_block):
            if block is None:
                continue
            tail_page = self._mamba_page(block)
            for (kv, score), (destination_kv, destination_score) in zip(
                self._tails, tail_page.tails, strict=True
            ):
                destination_kv.copy_(kv[slot])
                destination_score.copy_(score[slot])
        tail_fa_block = next(
            (
                block
                for block in reversed(
                    fa_blocks[: (accepted_end - 1) // self.block_tokens + 1]
                )
                if block
            ),
            None,
        )
        if tail_fa_block is not None:
            tail_fa_page = self._fa_page(tail_fa_block)
            for (kv, score), (destination_kv, destination_score) in zip(
                self._tails, tail_fa_page.tails, strict=True
            ):
                destination_kv.copy_(kv[slot])
                destination_score.copy_(score[slot])

    def restore(self, request_id, slot, row, *, start_pos, use_prefix_state=True):
        if type(start_pos) is not int or not 0 <= start_pos <= self.max_seq_len:
            raise ValueError("DeepSeek restore position is out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=start_pos)
        state_source = row.get("prefix_state_slot") if use_prefix_state else None
        if state_source is not None:
            self._validate_block_id(state_source, self.mamba_capacity, "prefix")
            if state_source == 0:
                raise ValueError("DeepSeek prefix state must name a physical page")
        if start_pos:
            last_page = (start_pos - 1) // self.block_tokens
            source_index = last_page
            if state_source is None:
                source_index = next(
                    (
                        index
                        for index in range(last_page, -1, -1)
                        if mamba_blocks[index]
                    ),
                    None,
                )
                if source_index is None:
                    raise ValueError("DeepSeek restore has no accepted state page")
                state_source = mamba_blocks[source_index]
            source_page = self.mamba_pages.get(state_source)
            if source_page is None:
                raise ValueError("DeepSeek restore state page is unavailable")
            source_start = source_index * self.block_tokens
            required_tokens = (
                min(start_pos, source_start + self.block_tokens) - source_start
            )
            if source_page.valid_tokens < required_tokens:
                raise ValueError("DeepSeek restore state page is incomplete")
        self.clear_slot(slot)
        if start_pos == 0:
            return
        self.reserve_slot(slot, start_pos)
        for page_index, block in enumerate(fa_blocks):
            page_start = page_index * self.block_tokens
            if page_start >= start_pos:
                break
            page_end = min(start_pos, page_start + self.block_tokens)
            if block == 0:
                continue
            page = self.fa_pages.get(block)
            if page is None:
                raise ValueError("DeepSeek restore references an unknown FA page")
            for value, source in zip(self._window, page.window, strict=True):
                for position in range(page_start, page_end):
                    value[slot, position % value.shape[1]] = source[
                        position - page_start
                    ]
            for value, source in zip(self._mtp_window, page.mtp_window, strict=True):
                for position in range(page_start, page_end):
                    value[slot, position % value.shape[1]] = source[
                        position - page_start
                    ]
            for (value, ratio), source in zip(
                self._compressed, page.compressed, strict=True
            ):
                first_row = page_start // ratio
                last_row = min(start_pos // ratio, (page_end + ratio - 1) // ratio)
                if first_row < last_row:
                    value[slot, first_row:last_row] = source[: last_row - first_row]
            for (value, ratio), source in zip(self._index, page.index, strict=True):
                first_row = page_start // ratio
                last_row = min(start_pos // ratio, (page_end + ratio - 1) // ratio)
                if first_row < last_row:
                    value[slot, first_row:last_row] = source[: last_row - first_row]
            for value, source in zip(self._engram, page.engram, strict=True):
                value[slot, page_start:page_end] = source[: page_end - page_start]
        for page_index, block in enumerate(fa_blocks):
            page_start = page_index * self.block_tokens
            if page_start >= start_pos:
                break
            if block == 0:
                continue
            page = self.fa_pages.get(block)
            if page is None:
                continue
            for (kv, score), (source_kv, source_score) in zip(
                self._tails, page.tails, strict=True
            ):
                if page_start + page.valid_tokens >= start_pos:
                    kv[slot].copy_(source_kv)
                    score[slot].copy_(source_score)
        for page_index, block in enumerate(mamba_blocks):
            page_start = page_index * self.block_tokens
            if page_start >= start_pos:
                break
            page_end = min(start_pos, page_start + self.block_tokens)
            if block == 0:
                continue
            page = self.mamba_pages.get(block)
            if page is None:
                continue
            for value, source in zip(self._engram, page.engram, strict=True):
                value[slot, page_start:page_end] = source[: page_end - page_start]
        if state_source:
            source_page = self.mamba_pages.get(state_source)
            source_start = (start_pos - 1) // self.block_tokens * self.block_tokens
            if source_page is None:
                raise ValueError("DeepSeek prefix state page is unavailable")
            source_end = min(start_pos, source_start + self.block_tokens)
            for value, source in zip(self._engram, source_page.engram, strict=True):
                value[slot, source_start:source_end] = source[
                    : source_end - source_start
                ]
            for (kv, score), (source_kv, source_score) in zip(
                self._tails, source_page.tails, strict=True
            ):
                kv[slot].copy_(source_kv)
                score[slot].copy_(source_score)
        tail_page = None
        if not state_source:
            for block in reversed(mamba_blocks[: ceil(start_pos / self.block_tokens)]):
                tail_page = self.mamba_pages.get(block)
                if tail_page is not None:
                    break
        if tail_page is not None:
            for (kv, score), (source_kv, source_score) in zip(
                self._tails, tail_page.tails, strict=True
            ):
                kv[slot].copy_(source_kv)
                score[slot].copy_(source_score)

    def copy_prefix(self, copies):
        for source, destination, length in copies:
            if (
                type(source) is not int
                or type(destination) is not int
                or type(length) is not int
                or source <= 0
                or destination <= 0
                or source == destination
                or length <= 0
                or length > self.block_tokens
            ):
                raise ValueError("invalid DeepSeek FA prefix copy")
            self._validate_block_id(source, self.fa_capacity, "FA source")
            self._validate_block_id(destination, self.fa_capacity, "FA destination")
            source_page = self.fa_pages.get(source)
            if source_page is None:
                raise ValueError("DeepSeek prefix copy references an unknown page")
            destination_page = self._fa_page(destination)
            destination_page.valid_tokens = max(destination_page.valid_tokens, length)
            for source_value, destination_value in zip(
                source_page.window, destination_page.window, strict=True
            ):
                destination_value[:length].copy_(source_value[:length])
            for source_value, destination_value in zip(
                source_page.mtp_window, destination_page.mtp_window, strict=True
            ):
                destination_value[:length].copy_(source_value[:length])
            for (_value, ratio), source_value, destination_value in zip(
                self._compressed,
                source_page.compressed,
                destination_page.compressed,
                strict=True,
            ):
                count = min(destination_value.shape[0], ceil(length / ratio))
                destination_value[:count].copy_(source_value[:count])
            for (_value, ratio), source_value, destination_value in zip(
                self._index,
                source_page.index,
                destination_page.index,
                strict=True,
            ):
                count = min(destination_value.shape[0], ceil(length / ratio))
                destination_value[:count].copy_(source_value[:count])
            for source_value, destination_value in zip(
                source_page.engram, destination_page.engram, strict=True
            ):
                destination_value[:length].copy_(source_value[:length])
            for (source_kv, source_score), (destination_kv, destination_score) in zip(
                source_page.tails, destination_page.tails, strict=True
            ):
                destination_kv.copy_(source_kv)
                destination_score.copy_(source_score)

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
            [self._host_tensor(value) for value in page.window],
            [self._host_tensor(value) for value in page.mtp_window],
            [self._host_tensor(value) for value in page.compressed],
            [self._host_tensor(value) for value in page.index],
            [self._host_tensor(value) for value in page.engram],
            [
                (self._host_tensor(kv), self._host_tensor(score))
                for kv, score in page.tails
            ],
            page.valid_tokens,
        )

    def _clone_mamba(self, page):
        return _MambaPage(
            [self._host_tensor(value) for value in page.engram],
            [
                (self._host_tensor(kv), self._host_tensor(score))
                for kv, score in page.tails
            ],
            page.valid_tokens,
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
        total = sum(
            self._shape_size((self.block_tokens, value.shape[-1]), value.dtype)
            for value in self._window
        )
        total += sum(
            self._shape_size((self.block_tokens, value.shape[-1]), value.dtype)
            for value in self._mtp_window
        )
        total += sum(
            self._shape_size(
                (ceil(self.block_tokens / ratio), *value.shape[2:]), value.dtype
            )
            for value, ratio in self._compressed
        )
        total += sum(
            self._shape_size(
                (ceil(self.block_tokens / ratio), *value.shape[2:]), value.dtype
            )
            for value, ratio in self._index
        )
        total += sum(
            self._shape_size((self.block_tokens,), value.dtype)
            for value in self._engram
        )
        total += sum(
            self._tensor_size(kv[0]) + self._tensor_size(score[0])
            for kv, score in self._tails
        )
        return total

    def _mamba_page_bytes(self):
        total = sum(
            self._shape_size((self.block_tokens,), value.dtype)
            for value in self._engram
        )
        total += sum(
            self._tensor_size(kv[0]) + self._tensor_size(score[0])
            for kv, score in self._tails
        )
        return total

    def _snapshot_size(self, fa, mamba):
        total = 0
        for page in fa:
            for value in (
                *page.window,
                *page.mtp_window,
                *page.compressed,
                *page.index,
            ):
                total += self._tensor_size(value)
            for value in page.engram:
                total += self._tensor_size(value)
            for kv, score in page.tails:
                total += self._tensor_size(kv) + self._tensor_size(score)
        for page in mamba:
            for value in page.engram:
                total += self._tensor_size(value)
            for kv, score in page.tails:
                total += self._tensor_size(kv) + self._tensor_size(score)
        return total

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
            for target, value in zip(destination.window, source.window, strict=True):
                target.copy_(self._device_tensor(value, target))
            for target, value in zip(
                destination.mtp_window, source.mtp_window, strict=True
            ):
                target.copy_(self._device_tensor(value, target))
            for target, value in zip(
                destination.compressed, source.compressed, strict=True
            ):
                target.copy_(self._device_tensor(value, target))
            for target, value in zip(destination.index, source.index, strict=True):
                target.copy_(self._device_tensor(value, target))
            for target, value in zip(destination.engram, source.engram, strict=True):
                target.copy_(self._device_tensor(value, target))
            for (target_kv, target_score), (value_kv, value_score) in zip(
                destination.tails, source.tails, strict=True
            ):
                target_kv.copy_(self._device_tensor(value_kv, target_kv))
                target_score.copy_(self._device_tensor(value_score, target_score))
            destination.valid_tokens = source.valid_tokens
        for block, source in zip(
            operation["mamba_blocks"], snapshot.mamba, strict=True
        ):
            destination = self._mamba_page(block)
            for target, value in zip(destination.engram, source.engram, strict=True):
                target.copy_(self._device_tensor(value, target))
            for (target_kv, target_score), (value_kv, value_score) in zip(
                destination.tails, source.tails, strict=True
            ):
                target_kv.copy_(self._device_tensor(value_kv, target_kv))
                target_score.copy_(self._device_tensor(value_score, target_score))
            destination.valid_tokens = source.valid_tokens
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
