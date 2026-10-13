"""Reusable decode graphs with transactional capture of mutable model caches."""

from dataclasses import fields

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.models.qwen import AttentionBatch, Batch, Qwen
from oh_my_vllm.worker.graph_capture import capture_graph


class DecodeAttention:
    def __init__(
        self,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        extent: int,
        max_query_len: int = 5,
    ):
        from oh_my_vllm.ir.attention import register_plan

        self.tables = tables
        self.lengths = lengths
        self.extent = extent
        self.max_query_len = max_query_len
        self.first = 0
        self.starts = None
        self._ir_handle = register_plan(self)

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        from oh_my_vllm.ir.attention import TARGET_OWNED_DECODE, planned_attention

        return planned_attention(
            query,
            cache,
            self._ir_handle,
            TARGET_OWNED_DECODE,
            self.tables,
            self.lengths,
        )

    def _run(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        tables: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tables is None or lengths is None:
            raise ValueError("target decode needs explicit tables and lengths")
        return decode(
            query,
            cache,
            tables,
            lengths,
            first=self.first,
            max_tokens=self.extent,
            starts=self.starts,
            max_query_len=self.max_query_len,
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
        pool=None,
        compile_model: bool = False,
        capture_features: bool = False,
        max_query_len: int = 5,
    ) -> None:
        if batch.prefill_sequences:
            raise ValueError("decode graphs cannot capture prefill")
        self.tokens = tokens.clone()
        self.attention = DecodeAttention(
            tables.clone(),
            batch.positions.clone() + 1,
            extent,
            max_query_len=max_query_len,
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

        def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            if capture_features:
                hidden, features = model.forward_features(
                    self.tokens, self.batch, caches
                )
            else:
                hidden = model.forward(self.tokens, self.batch, caches)
                features = None
            return hidden, model.logits(hidden), features

        unit = compile_forward(run, unit="target_graph") if compile_model else run
        try:
            # JIT compilation and library initialization must precede capture.
            unit()
            restore()
            torch.cuda.synchronize()
            with capture_graph(self.graph, pool=pool):
                self.hidden, self.logits, self.features = unit()
            self.execution_owners = tuple(getattr(unit, "execution_owners", ()))
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
    def __init__(
        self,
        model,
        cache,
        tokens,
        hidden,
        batch,
        tables,
        extent,
        starts=None,
        pool=None,
        compile_model: bool = False,
    ):
        self.tokens, self.input_hidden = tokens.clone(), hidden.clone()
        self.attention = DecodeAttention(tables.clone(), batch.positions + 1, extent)
        self.attention.first = 1
        if starts is not None and tokens.numel() > starts.numel() - 1:
            self.attention.starts = starts.clone()
        self.batch = AttentionBatch(
            batch.positions.clone(), batch.fa_slots.clone(), self.attention
        )
        self.graph = torch.cuda.CUDAGraph()
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        saved = cache[pages, :, offsets].clone()

        def run():
            return model.draft(self.tokens, self.input_hidden, self.batch, cache)

        unit = compile_forward(run, unit="draft_graph") if compile_model else run
        try:
            unit()
            cache[pages, :, offsets] = saved
            torch.cuda.synchronize()
            with capture_graph(self.graph, pool=pool):
                self.hidden = unit()
            self.execution_owners = tuple(getattr(unit, "execution_owners", ()))
        finally:
            cache[pages, :, offsets] = saved

    def replay(self, tokens, hidden, batch, tables, starts=None):
        if self.attention.starts is not None:
            self.attention.starts.copy_(starts)
        self.tokens.copy_(tokens)
        # The prior draft graph's output may share this family's pool. Queue its
        # copy on the replay stream before this graph can overwrite that pool.
        self.input_hidden.copy_(hidden)
        self.batch.positions.copy_(batch.positions)
        self.batch.fa_slots.copy_(batch.fa_slots)
        self.attention.tables.copy_(tables)
        self.attention.lengths.copy_(batch.positions + 1)
        self.graph.replay()
        return self.hidden


class ProposalGraph:
    """Four greedy proposals with no intermediate device-to-host synchronization."""

    @torch.inference_mode()
    def __init__(
        self,
        model,
        cache,
        hidden,
        positions,
        tables,
        extent,
        pool=None,
        compile_model: bool = False,
    ):
        self.model, self.cache = model, cache
        self.hidden = hidden.clone()
        self.positions = positions.clone()
        self.tables = tables.clone()
        # Allocated outside capture: retain it for every later replay, too.
        self.rows = torch.arange(len(hidden), device=hidden.device)
        steps = torch.arange(1, 4, device=hidden.device)
        all_positions = positions[:, None] + steps
        pages = tables.gather(1, all_positions // 784)
        offsets = all_positions % 784
        saved = cache[pages, :, offsets].clone()
        self.graph = torch.cuda.CUDAGraph()
        self.proposal_attentions = []
        for step in range(1, 4):
            attention = DecodeAttention(self.tables, self.positions + step + 1, extent)
            attention.first = 1
            self.proposal_attentions.append(attention)

        def run():
            current = self.hidden
            token = model.logits(current).argmax(-1)
            outputs = [token]
            for step in range(1, 4):
                position = self.positions + step
                page = self.tables[self.rows, position // 784]
                attention = self.proposal_attentions[step - 1]
                attention.lengths = position + 1
                batch = AttentionBatch(position, page * 784 + position % 784, attention)
                current = model.draft(token, current, batch, cache)
                token = model.logits(current).argmax(-1)
                outputs.append(token)
            return torch.stack(outputs, -1)

        unit = compile_forward(run, unit="proposal_graph") if compile_model else run
        try:
            unit()
            cache[pages, :, offsets] = saved
            torch.cuda.synchronize()
            with capture_graph(self.graph, pool=pool):
                self.tokens = unit()
            self.execution_owners = tuple(getattr(unit, "execution_owners", ()))
        finally:
            cache[pages, :, offsets] = saved

    def replay(self, hidden, positions, tables):
        self.hidden.copy_(hidden)
        self.positions.copy_(positions)
        self.tables.copy_(tables)
        self.graph.replay()
        return self.tokens
