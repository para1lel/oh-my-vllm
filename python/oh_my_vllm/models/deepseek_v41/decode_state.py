"""Capture-owned recurrent endpoints for a batched target call."""

from contextlib import contextmanager

import torch


@contextmanager
def redirect_state_exports(model, replacement):
    saved = []
    for owner in model.modules():
        if hasattr(owner, "_state_export"):
            saved.append((owner, owner._state_export))
            owner._state_export = replacement
    try:
        yield
    finally:
        for owner, export in saved:
            owner._state_export = export


class GraphStateExports:
    def __init__(self, bindings, start_pos):
        self.bindings = bindings
        self.start_pos = start_pos
        self.steps = {}
        self.windows = {}
        self.tails = {}

    def window(self, cache, values, start_pos):
        batch, length, width = values.shape
        window = cache.size(1)
        key = id(cache)
        if key not in self.windows:
            self.windows[key] = torch.empty(
                (batch, length, window, width), dtype=cache.dtype, device=cache.device
            )
        snapshots = self.windows[key]
        previous = cache.index_select(0, self.bindings.cache_slots)
        rows = torch.arange(batch, device=cache.device)[:, None]
        positions = self.bindings.position.reshape(-1, 1)
        if start_pos < window:
            padding = torch.arange(window, device=cache.device) >= positions
            previous.masked_fill_(padding.unsqueeze(-1), 0)
        for offset in range(length):
            snapshots[:, offset].copy_(previous)
            ring = (
                positions + torch.arange(offset + 1, device=cache.device)
            ).remainder(window)
            snapshots[rows, offset, ring] = values[:, : offset + 1]

    def tail(self, kv, score, values, scores, start_pos):
        ratio, width = kv.shape[1:]
        batch, length = self.bindings.batch, self.bindings.length
        key = id(kv)
        if key not in self.tails:
            shape = (batch, length, ratio, width)
            self.tails[key] = (
                torch.empty(shape, dtype=kv.dtype, device=kv.device),
                torch.empty(shape, dtype=score.dtype, device=score.device),
            )
        snapshots, score_snapshots = self.tails[key]
        snapshots.zero_()
        score_snapshots.fill_(-torch.inf)
        prefix = start_pos % ratio
        for offset in range(length):
            remainder = (start_pos + offset + 1) % ratio
            if remainder:
                last = prefix + offset + 1
                snapshots[:, offset, :remainder].copy_(
                    values[:, last - remainder : last]
                )
                score_snapshots[:, offset, :remainder].copy_(
                    scores[:, last - remainder : last]
                )

    def publish(self, export, *, preserve_existing=False):
        for index, (slot, start_pos) in enumerate(
            zip(self.bindings.cache_slots_cpu, self.bindings.positions_cpu, strict=True)
        ):
            step = export.steps.get(slot)
            if step is None:
                continue
            for end in step.endpoints:
                offset = end - start_pos - 1
                if not 0 <= offset < self.bindings.length:
                    continue
                for key, values in self.windows.items():
                    if not preserve_existing or (end, key) not in step.windows:
                        step.windows[end, key] = values[index, offset]
                for key, (kv, score) in self.tails.items():
                    if not preserve_existing or (end, key) not in step.tails:
                        step.tails[end, key] = kv[index, offset], score[index, offset]
