"""Bounded persistent MTP context captures with selected endpoint outputs."""

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.models.qwen import AttentionBatch
from oh_my_vllm.worker.decode_graph import DecodeAttention


class MTPContextGraph:
    @torch.inference_mode()
    def __init__(
        self, model, cache, tokens, hidden, batch, indices, attention, *, pool=None
    ):
        self.tokens, self.hidden = tokens.clone(), hidden.clone()
        self.indices = indices.clone()
        self.batch = AttentionBatch(
            batch.positions.clone(), batch.fa_slots.clone(), None
        )
        self.attention = None
        if attention is not None:
            self.attention = DecodeAttention(
                attention.tables.clone(), attention.lengths.clone(), attention.extent
            )
            self.attention.first = 1
        self.graph = torch.cuda.CUDAGraph()
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        saved = cache[pages, :, offsets].clone()

        def run():
            return model.draft_context(
                self.tokens,
                self.hidden,
                self.batch,
                cache,
                self.indices,
                self.attention,
            )

        unit = compile_forward(run, unit="mtp_context_graph")
        try:
            unit()
            cache[pages, :, offsets] = saved
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, pool=pool):
                self.output = unit()
        finally:
            cache[pages, :, offsets] = saved

    @torch.inference_mode()
    def replay(self, tokens, hidden, batch, indices, attention):
        pairs = [
            (self.tokens, tokens),
            (self.hidden, hidden),
            (self.indices, indices),
            (self.batch.positions, batch.positions),
            (self.batch.fa_slots, batch.fa_slots),
        ]
        if attention is not None:
            if (
                self.attention is None
                or attention.first != 1
                or self.attention.extent != attention.extent
            ):
                raise ValueError("MTP context attention shape changed after capture")
            pairs.extend(
                (
                    (self.attention.tables, attention.tables),
                    (self.attention.lengths, attention.lengths),
                )
            )
        elif self.attention is not None:
            raise ValueError("MTP context lost selected attention")
        if any(
            dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device
            for dst, src in pairs
        ):
            raise ValueError("MTP context tensor metadata changed after capture")
        for destination, source in pairs:
            destination.copy_(source)
        self.graph.replay()
        return self.output
