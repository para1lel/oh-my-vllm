"""Private endpoint state captured before recurrent buffers are overwritten."""

from dataclasses import dataclass, field

import torch


@dataclass
class _Step:
    endpoints: tuple[int, ...]
    minimum_end: int
    windows: dict = field(default_factory=dict)
    tails: dict = field(default_factory=dict)


class StateExports:
    def __init__(self):
        self.steps = {}
        self.batch = ()

    def begin(self, slot, start_pos, endpoints, *, minimum_end=None):
        endpoints = tuple(sorted(set(endpoints)))
        if (
            type(slot) is not int
            or slot < 0
            or type(start_pos) is not int
            or start_pos < 0
            or not endpoints
            or any(type(end) is not int or end <= start_pos for end in endpoints)
        ):
            raise ValueError("invalid DeepSeek state export endpoints")
        if slot in self.steps:
            raise ValueError("DeepSeek slot already has a state export step")
        minimum_end = max(endpoints) if minimum_end is None else minimum_end
        if type(minimum_end) is not int or minimum_end not in endpoints:
            raise ValueError("invalid DeepSeek minimum accepted endpoint")
        self.steps[slot] = _Step(endpoints, minimum_end)

    def bind(self, slots):
        slots = tuple(slots)
        if len(set(slots)) != len(slots) or any(
            type(slot) is not int or slot < 0 for slot in slots
        ):
            raise ValueError("invalid DeepSeek state export batch slots")
        self.batch = slots

    def _selected(self, batch_size, start_pos, end_pos):
        if not self.steps:
            return
        if len(self.batch) != batch_size:
            raise ValueError("DeepSeek state export batch is not bound")
        for index, slot in enumerate(self.batch):
            step = self.steps.get(slot)
            if step is not None:
                for end in step.endpoints:
                    if start_pos < end <= end_pos:
                        yield index, slot, step, end

    def window(self, cache, values, start_pos):
        width = cache.size(1)
        for index, slot, step, end in self._selected(
            values.size(0), start_pos, start_pos + values.size(1)
        ):
            key = end, id(cache)
            if key in step.windows:
                continue
            snapshot = torch.zeros_like(cache[slot])
            first = max(0, end - width)
            previous_end = min(start_pos, end)
            if first < previous_end:
                positions = torch.arange(first, previous_end, device=cache.device)
                offsets = positions.remainder(width)
                snapshot[offsets] = cache[slot, offsets]
            first = max(first, start_pos)
            positions = torch.arange(first, end, device=cache.device)
            snapshot[positions.remainder(width)] = values[
                index, first - start_pos : end - start_pos
            ]
            step.windows[key] = snapshot

    def tail(self, kv, score, values, scores, start_pos):
        ratio = kv.size(1)
        origin = start_pos - start_pos % ratio
        for index, _slot, step, end in self._selected(
            values.size(0), start_pos, origin + values.size(1)
        ):
            key = end, id(kv)
            if key in step.tails:
                continue
            snapshot_kv = torch.zeros_like(kv[0])
            snapshot_score = torch.full_like(score[0], -torch.inf)
            remainder = end % ratio
            if remainder:
                first, last = end - remainder - origin, end - origin
                snapshot_kv[:remainder].copy_(values[index, first:last])
                snapshot_score[:remainder].copy_(scores[index, first:last])
            step.tails[key] = snapshot_kv, snapshot_score

    def window_state(self, slot, endpoint, cache):
        try:
            return self.steps[slot].windows[endpoint, id(cache)]
        except KeyError:
            raise ValueError("missing DeepSeek window state at endpoint") from None

    def tail_state(self, slot, endpoint, kv):
        try:
            return self.steps[slot].tails[endpoint, id(kv)]
        except KeyError:
            raise ValueError("missing DeepSeek compressor state at endpoint") from None

    def endpoints(self, slot):
        step = self.steps.get(slot)
        return () if step is None else step.endpoints

    def discard(self, slot):
        self.steps.pop(slot, None)

    def clear(self):
        self.steps.clear()
        self.batch = ()
