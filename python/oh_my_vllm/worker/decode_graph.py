"""Reusable decode graphs with transactional capture of mutable model caches."""

from dataclasses import fields

import torch

from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.models.qwen import AttentionBatch, Batch, Qwen


class DecodeAttention:
    def __init__(self, tables: torch.Tensor, lengths: torch.Tensor, extent: int):
        self.tables = tables
        self.lengths = lengths
        self.extent = extent
        self.first = 0
        self.starts = None

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return decode(
            query,
            cache,
            self.tables,
            self.lengths,
            first=self.first,
            max_tokens=self.extent,
            starts=self.starts,
        )


class DecodeGraph:
    """A fixed token/request shape; all request addresses remain dynamic inputs.

    Warmup and capture execute kernels that mutate the KV and recurrent pools.
    Save every destination before either operation and restore in a finally block.
    Thus graph creation has the same externally visible cache effect as one replay,
    including when a recurrent destination is also the committed source.
    """

    @torch.inference_mode()
    def __init__(
        self,
        model: Qwen,
        caches: list,
        tokens: torch.Tensor,
        batch: Batch,
        tables: torch.Tensor,
        extent: int,
    ) -> None:
        if batch.prefill_sequences:
            raise ValueError("decode graphs cannot capture prefill")
        self.tokens = tokens.clone()
        self.attention = DecodeAttention(
            tables.clone(), batch.positions.clone() + 1, extent
        )
        self.batch = Batch(
            **{
                f.name: value.clone() if isinstance(value, torch.Tensor) else value
                for f in fields(batch)
                if f.name != "attention"
                for value in [getattr(batch, f.name)]
            },
            attention=self.attention,
        )
        if tokens.numel() > batch.starts.numel() - 1:
            self.attention.starts = self.batch.starts
        self.graph = torch.cuda.CUDAGraph()
        state_slots = batch.state_writes.unique()
        state_slots = state_slots[state_slots >= 0]
        fa_pages, fa_offsets = batch.fa_slots // 784, batch.fa_slots % 784
        saved = []
        for cache in caches:
            if isinstance(cache, tuple):
                saved.append(tuple(pool[state_slots].clone() for pool in cache))
            else:
                saved.append(cache[fa_pages, :, fa_offsets].clone())

        def restore() -> None:
            for cache, backup in zip(caches, saved, strict=True):
                if isinstance(cache, tuple):
                    for pool, data in zip(cache, backup, strict=True):
                        pool[state_slots] = data
                else:
                    cache[fa_pages, :, fa_offsets] = backup

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            hidden = model.forward(self.tokens, self.batch, caches)
            return hidden, model.logits(hidden)

        try:
            # JIT compilation and library initialization must precede capture.
            run()
            restore()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self.hidden, self.logits = run()
        finally:
            restore()

    def replay(
        self, tokens: torch.Tensor, batch: Batch, tables: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.tokens.copy_(tokens)
        self.attention.tables.copy_(tables)
        self.attention.lengths.copy_(batch.positions + 1)
        for f in fields(batch):
            value = getattr(batch, f.name)
            if isinstance(value, torch.Tensor):
                getattr(self.batch, f.name).copy_(value)
        self.graph.replay()
        return self.hidden, self.logits


class DraftGraph:
    """Capture the single MTP layer; preserve overwritten FA rows during capture."""

    @torch.inference_mode()
    def __init__(self, model, cache, tokens, hidden, batch, tables, extent):
        self.tokens, self.input_hidden = tokens.clone(), hidden.clone()
        self.attention = DecodeAttention(tables.clone(), batch.positions + 1, extent)
        self.attention.first = 1
        self.batch = AttentionBatch(
            batch.positions.clone(), batch.fa_slots.clone(), self.attention
        )
        self.graph = torch.cuda.CUDAGraph()
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        saved = cache[pages, :, offsets].clone()

        def run():
            return model.draft(self.tokens, self.input_hidden, self.batch, cache)

        try:
            run()
            cache[pages, :, offsets] = saved
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self.hidden = run()
        finally:
            cache[pages, :, offsets] = saved

    def replay(self, tokens, hidden, batch, tables):
        self.tokens.copy_(tokens)
        self.input_hidden.copy_(hidden)
        self.batch.positions.copy_(batch.positions)
        self.batch.fa_slots.copy_(batch.fa_slots)
        self.attention.tables.copy_(tables)
        self.attention.lengths.copy_(batch.positions + 1)
        self.graph.replay()
        return self.hidden
