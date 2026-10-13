"""MTP attention over token-indexed pages, with cache position zero absent."""

from itertools import pairwise

import torch

from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.worker.tensors import device_page_tables


class MTPAttention:
    def __init__(self, max_tokens: int) -> None:
        from oh_my_vllm.ir.attention import register_plan

        self.max_tokens = max_tokens
        self.workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
        self._ir_handle = register_plan(self)

    def plan(
        self, starts: list[int], pages: list[list[int]], positions: list[int]
    ) -> None:
        counts = [b - a for a, b in pairwise(starts)]
        if len(counts) != len(pages) or not counts or min(counts) <= 0:
            raise ValueError("MTP attention requires nonempty sequences")
        if starts[0] != 0 or starts[-1] != len(positions):
            raise ValueError("MTP query offsets do not match positions")
        self._ir_reference_plan = starts, pages, positions
        ends = []
        for i, count in enumerate(counts):
            query = positions[starts[i] : starts[i + 1]]
            if query[0] < 1 or query != list(range(query[0], query[0] + count)):
                raise ValueError(
                    "MTP positions must be contiguous and start after zero"
                )
            end = query[-1] + 1
            if end > self.max_tokens or len(pages[i]) * 784 < end:
                raise ValueError("MTP queries exceed allocated pages")
            ends.append(end)
        self.decode_mode = max(counts) <= 5
        if self.decode_mode:
            self.extent = min(self.max_tokens, ((max(positions) + 4096) // 4096) * 4096)
            width = (self.extent + 783) // 784
            self.tables = device_page_tables(
                pages, width, counts=counts, device="cuda", dtype=torch.int32
            )
            self.lengths = torch.tensor(
                [p + 1 for p in positions], dtype=torch.int32, device="cuda"
            )
        else:
            slots, offsets = [], [0]
            for table, end in zip(pages, ends, strict=True):
                slots.extend(table[p // 784] * 784 + p % 784 for p in range(1, end))
                offsets.append(len(slots))
            self.slots = torch.tensor(slots, device="cuda")
            self.query_offsets = starts
            self.kv_offsets = offsets

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        from oh_my_vllm.ir.attention import (
            MTP_DECODE,
            MTP_PREFILL,
            planned_attention,
        )

        route = MTP_DECODE if self.decode_mode else MTP_PREFILL
        tables = self.tables if self.decode_mode else None
        lengths = self.lengths if self.decode_mode else None
        return planned_attention(query, cache, self._ir_handle, route, tables, lengths)

    def _run(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        tables: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decode_mode:
            if tables is None or lengths is None:
                raise ValueError("MTP decode needs explicit tables and lengths")
            return decode(
                query,
                cache,
                tables,
                lengths,
                first=1,
                max_tokens=self.extent,
            )
        pages, offsets = self.slots // 784, self.slots % 784
        k, v = cache[pages, 0, offsets], cache[pages, 1, offsets]
        from .ragged_attention import causal_attention

        return causal_attention(query, k, v, self.query_offsets, self.kv_offsets)
