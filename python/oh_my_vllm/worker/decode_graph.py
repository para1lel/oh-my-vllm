"""Reusable decode graphs with transactional capture of mutable model caches."""

from dataclasses import fields

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.decode_attention import decode
from oh_my_vllm.models.qwen import AttentionBatch, Batch, Qwen


class DecodeAttention:
    def __init__(
        self,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        extent: int,
        workspace=None,
        max_query_len: int = 5,
    ):
        from oh_my_vllm.ir.attention import register_plan

        self.tables = tables
        self.lengths = lengths
        self.extent = extent
        self.max_query_len = max_query_len
        self.first = 0
        self.starts = None
        self.workspace = workspace
        self.native_metadata = None
        self._ir_handle = register_plan(self)

    def prepare(self) -> None:
        """Translate logical 784-token pages once per target model execution."""
        if self.first:
            return
        if self.workspace is None:
            self.workspace = torch.empty(
                128 << 20, device=self.tables.device, dtype=torch.uint8
            )
        if self.starts is None:
            tables, lengths = self.tables, self.lengths
            starts = torch.arange(
                len(lengths) + 1, device=lengths.device, dtype=torch.int32
            )
        else:
            tables = self.tables.index_select(0, self.starts[:-1])
            lengths = self.lengths.index_select(0, self.starts[1:] - 1)
            starts = self.starts.to(torch.int32)
        offsets = torch.arange(49, device=tables.device, dtype=torch.int32)
        subpages = (tables[:, :, None] * 98 + offsets).flatten(1).to(torch.int32)
        self.native_metadata = subpages, lengths.to(torch.int32), starts

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        from oh_my_vllm.ir.attention import (
            TARGET_NATIVE_DECODE,
            TARGET_OWNED_DECODE,
            planned_attention,
        )

        route = (
            TARGET_NATIVE_DECODE
            if self.first == 0 and self.native_metadata is not None
            else TARGET_OWNED_DECODE
        )
        native = self.native_metadata if route == TARGET_NATIVE_DECODE else (None,) * 3
        return planned_attention(
            query,
            cache,
            self._ir_handle,
            route,
            self.tables,
            self.lengths,
            *native,
        )

    def _run(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        tables: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        native_tables: torch.Tensor | None = None,
        native_lengths: torch.Tensor | None = None,
        native_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tables is None or lengths is None:
            raise ValueError("target decode needs explicit tables and lengths")
        if self.first == 0 and native_tables is not None:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache

            tables, lengths, starts = native_tables, native_lengths, native_starts
            # Every logical page stores 49 K subpages followed by 49 V subpages.
            # Offset views share the same indices; holes are never addressed.
            blocks = cache.view(-1, 16, *cache.shape[-2:])
            return trtllm_batch_decode_with_kv_cache(
                query,
                (blocks[:-49], blocks[49:]),
                self.workspace,
                tables,
                lengths,
                self.extent,
                bmm1_scale=query.shape[-1] ** -0.5,
                kv_layout="NHD",
                backend="trtllm-gen",
                q_len_per_req=None,
                max_q_len=self.max_query_len if self.starts is not None else 1,
                cum_seq_lens_q=starts,
            )
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
            workspace=getattr(batch.attention, "workspace", None),
            max_query_len=max_query_len,
        )
        if self.attention.workspace is None:
            self.attention.workspace = torch.empty(
                128 << 20, device=tables.device, dtype=torch.uint8
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
            self.attention.prepare()
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
            with torch.cuda.graph(self.graph, pool=pool):
                self.hidden, self.logits, self.features = unit()
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
            with torch.cuda.graph(self.graph, pool=pool):
                self.hidden = unit()
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
            with torch.cuda.graph(self.graph, pool=pool):
                self.tokens = unit()
        finally:
            cache[pages, :, offsets] = saved

    def replay(self, hidden, positions, tables):
        self.hidden.copy_(hidden)
        self.positions.copy_(positions)
        self.tables.copy_(tables)
        self.graph.replay()
        return self.tokens
