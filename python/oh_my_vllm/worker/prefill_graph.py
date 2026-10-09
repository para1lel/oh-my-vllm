"""Bounded target prefill captures with owned, updateable attention metadata."""

from dataclasses import fields

import torch

from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import Batch


class PrefillGraph:
    """Capture a fixed query/context shape while keeping page addresses dynamic.

    This family captures the ragged or native prefill attention path. Short FA2
    plans retain their existing path. Independent graphs share scratch only when
    their complete fork/join executions occur sequentially on the origin stream.
    """

    @torch.inference_mode()
    def __init__(
        self, model, caches, tokens, batch, plan, *, pool=None, capture_features=False
    ):
        self.tokens = tokens.clone()
        self.attention = PagedAttention(workspace=batch.attention.workspace)
        self.attention.plan(*plan, 24, 4, 256)
        if not (self.attention.native or self.attention.ragged):
            raise ValueError("prefill graph requires native or ragged attention")
        self.batch = Batch(
            **{
                f.name: value.clone() if isinstance(value, torch.Tensor) else value
                for f in fields(batch)
                if f.name != "attention"
                for value in [getattr(batch, f.name)]
            },
            attention=self.attention,
        )
        self.graph = torch.cuda.CUDAGraph()
        state_slots = batch.state_writes.unique()
        state_slots = state_slots[state_slots >= 0]
        fa_pages, fa_offsets = batch.fa_slots // 784, batch.fa_slots % 784
        saved = [
            tuple(state[state_slots].clone() for state in cache)
            if isinstance(cache, tuple)
            else cache[fa_pages, :, fa_offsets].clone()
            for cache in caches
        ]

        def restore():
            for cache, backup in zip(caches, saved, strict=True):
                if isinstance(cache, tuple):
                    for state, data in zip(cache, backup, strict=True):
                        state[state_slots] = data
                else:
                    cache[fa_pages, :, fa_offsets] = backup

        def run():
            if capture_features:
                return model.forward_features(self.tokens, self.batch, caches)
            return model.forward(self.tokens, self.batch, caches), None

        unit = compile_forward(run, unit="target_prefill_graph")
        try:
            unit()
            restore()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, pool=pool):
                self.hidden, self.features = unit()
        finally:
            restore()

    @torch.inference_mode()
    def replay(self, tokens, batch, plan):
        self.tokens.copy_(tokens)
        for f in fields(batch):
            value = getattr(batch, f.name)
            if isinstance(value, torch.Tensor):
                getattr(self.batch, f.name).copy_(value)
        # Planning creates new addresses. Copy its dynamic values into the
        # tensors captured by the graph instead of replacing their storage.
        names = (
            "query_starts",
            "kv_starts",
            "kv_lengths",
            "subpages",
            "key_slots",
            "value_slots",
        )
        static = {
            name: getattr(self.attention, name)
            for name in names
            if hasattr(self.attention, name)
        }
        # A failed plan must leave all captured storage alive and bound. Check
        # every replacement before updating any captured tensor.
        try:
            self.attention.plan(*plan, 24, 4, 256)
            dynamic = {name: getattr(self.attention, name) for name in static}
            if any(
                tensor.shape != dynamic[name].shape
                or tensor.dtype != dynamic[name].dtype
                for name, tensor in static.items()
            ):
                raise ValueError("prefill attention shape changed after capture")
        finally:
            for name, tensor in static.items():
                setattr(self.attention, name, tensor)
        for name, tensor in static.items():
            tensor.copy_(dynamic[name])
        self.graph.replay()
        return self.hidden, self.features
