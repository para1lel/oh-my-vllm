from math import ceil

import torch

from .cache_state import CacheState
from .cache_transfer import CacheTransfers
from .cache_types import _FaPage, _MambaPage, _table
from .packed_cache import IndexWorkspace, PageTable
from .state import StateExports


class DeepSeekCache(CacheState, CacheTransfers):
    """Keep model state in logical pages and support Rust cache transfers.

    Active slots reference physical packed history through a block table.
    Accepted state pages retain the windows and incomplete compressor groups.
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
        release_request_slot=None,
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
        self.release_request_slot = release_request_slot
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
        self._bind_model_state(model)
        reference = self._window[0]
        self.fa_capacity = fa_capacity or (
            1 + reference.size(0) * ceil(max_seq_len / block_tokens)
        )
        self.page_table = PageTable(
            reference.size(0),
            max_seq_len,
            block_tokens,
            self.fa_capacity,
            reference.device,
        )
        self.index_workspace = None
        if self._index:
            index = self._index[0][0]
            self.index_workspace = IndexWorkspace(
                reference.size(0),
                max_seq_len,
                index.shape[-1],
                index.device,
                index.lazy,
                index.dtype,
            )
        index_owners = {id(value) for value, _ratio in self._index}
        for value in self.histories:
            value.bind(
                self.page_table,
                self.index_workspace if id(value) in index_owners else None,
            )
        model._physical_cache = self

    def _bind_model_state(self, model):
        model._state_export = self.exports
        for layer in model.layers:
            attention = layer.attn
            self._window.append(attention.window_kv_cache)
            attention._state_export = self.exports
            if attention.is_kv_source:
                self._compressed.append(
                    (attention.compress_kv_cache, attention.compress_ratio)
                )
                if attention.compress_ratio > 1:
                    compressor = attention.compressor
                    self._tails.append((compressor.kv_state, compressor.score_state))
                    compressor._state_export = self.exports
            if attention.indexer is not None and attention.indexer.owns_k:
                self._index.append(
                    (attention.indexer.k_cache, attention.compress_ratio)
                )
        for layer in model.mtp:
            self._mtp_window.append(layer.attn.window_kv_cache)
            layer.attn._state_export = self.exports
        if model.engram_hash is not None:
            self._engram.append(model.engram_hash.cache)
        self.histories = (
            *(value for value, _ in self._compressed),
            *(value for value, _ in self._index),
            *self._engram,
        )

    def stats(self):
        return {
            "gpu_logical_page_bytes": (
                len(self.fa_pages) * self._fa_page_bytes()
                + len(self.mamba_pages) * self._mamba_page_bytes()
            ),
            "packed_pool_capacity_bytes": self.fa_capacity * self._fa_page_bytes(),
            "packed_pool_generation": self.page_table.generation,
            "packed_pool_graph_leases": self.page_table.leases,
            "index_workspace_reserved_bytes": (
                self.index_workspace.values.numel()
                * self.index_workspace.values.element_size()
                if self.index_workspace is not None
                else 0
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

    def _new_fa_page(self, block):
        for value in self.histories:
            value.reserve_page(block)
        return _FaPage(
            [value.page(block) for value, _ratio in self._compressed],
            [value.page(block) for value, _ratio in self._index],
            [value.page(block) for value in self._engram],
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
            page = self._new_fa_page(block)
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

    def _ensure_pages(self, fa, mamba):
        for block in fa:
            if block:
                self._fa_page(block)
        for block in mamba:
            if block:
                self._mamba_page(block)

    def activate_row(self, row, slot, *, end_pos):
        """Bind the scheduler's logical pages before physical model writes."""
        if type(end_pos) is not int or not 0 <= end_pos <= self.max_seq_len:
            raise ValueError("DeepSeek active endpoint is out of bounds")
        fa, mamba = self._validate_row(row, end=end_pos)
        self._ensure_pages(fa, mamba)
        self.page_table.bind(slot, fa)

    def require_active_slots(self, slots):
        if any(slot not in self.page_table.active for slot in slots):
            raise ValueError("bind a DeepSeek cache row before model execution")

    def clear_slot(self, slot):
        self.exports.discard(slot)
        self.page_table.bind(slot, [])
        for value in self._window:
            value[slot].zero_()
        for value in self._mtp_window:
            value[slot].zero_()
        for kv, score in self._tails:
            kv[slot].zero_()
            score[slot].fill_(-torch.inf)

    def reserve_workspace(self, batch, end):
        if self.index_workspace is not None:
            self.index_workspace.reserve(batch * min(end, self.max_seq_len))

    def reset(self):
        """Start an empty cache epoch without replacing physical pool addresses."""
        if self.page_table.closed or self.page_table.leases:
            raise RuntimeError("close captured graphs before resetting the packed pool")
        if self.page_table.values.is_cuda:
            torch.cuda.synchronize(self.page_table.values.device)
        for history in self.histories:
            for block in history.initialized:
                history.discard_after(block, 0)
        self.page_table.generation += 1
        slots = self.page_table.values.size(0)
        for slot in range(slots):
            self.clear_slot(slot)
        self.exports.clear()
        self.snapshots.clear()
        self.fa_pages.clear()
        self.mamba_pages.clear()
        self.seen.clear()
        self.used_bytes = 0
        self.unusable = False

    def close(self):
        self.page_table.close()
        self.release_request_slot = None
        for value in self.histories:
            value.close()
        if self.index_workspace is not None:
            self.index_workspace.close()
        self.exports.clear()
        self.snapshots.clear()
        self.fa_pages.clear()
        self.mamba_pages.clear()
        self.used_bytes = 0
