"""Concurrent Eager dense owners with lockstep expert-only empty ranks."""

import torch
import torch.distributed as dist

from .dispatch import ExpertExchange

_DONE = 0
_TARGET = 1
_PREFILL = 2
_PROPOSAL = 3


class Dep4Executor:
    """Align whole target/proposal calls without taking request ownership."""

    def __init__(self, model, *, group=None):
        self.model = model
        self.group = group
        self.exchange = ExpertExchange(group=group)
        self.active = False
        self.call = 0
        self.target_rounds = 0
        self.target_active_ranks = 0
        self.record_round = None
        self.accounting = None
        self.target = model.forward
        self.proposal = model.forward_spec
        model.forward = self.forward
        model.forward_spec = self.forward_spec
        for layer in (*model.layers, *model.mtp):
            layer.ffn.dep4_exchange = self.exchange

    def _pending(self, kind):
        device = next(self.model.parameters()).device
        local = torch.tensor([kind], dtype=torch.int32, device=device)
        gathered = [
            torch.empty_like(local) for _ in range(dist.get_world_size(self.group))
        ]
        dist.all_gather(gathered, local, group=self.group)
        pending = [int(value.item()) for value in gathered]
        selected = min((value for value in pending if value != _DONE), default=_DONE)
        return selected, pending.count(selected)

    def _empty(self, kind):
        layers = self.model.layers if kind in (_TARGET, _PREFILL) else self.model.mtp
        device = next(self.model.parameters()).device
        for layer in layers:
            moe = layer.ffn
            rows = torch.empty((0, moe.dim), dtype=torch.bfloat16, device=device)
            ids = torch.empty(
                (0, moe.n_activated_experts), dtype=torch.int32, device=device
            )
            weights = torch.empty_like(ids, dtype=torch.float32)
            self.exchange(moe, rows, (weights, ids))

    def _join(self, kind, function=None):
        if not self.active:
            raise RuntimeError("DEP4 model calls require a Rust-owned execution step")
        while True:
            selected, active_ranks = self._pending(kind)
            if selected == _DONE:
                return None
            target = selected in (_TARGET, _PREFILL)
            if target:
                self.call += 1
                self.target_rounds += 1
                self.target_active_ranks += active_ranks
                if self.accounting is not None:
                    self.accounting.begin(
                        self.call, "prefill" if selected == _PREFILL else "decode"
                    )
            if self.record_round is not None:
                self.record_round(
                    "dep4_round",
                    call=self.call,
                    phase="target" if target else "proposal",
                    active_ranks=active_ranks,
                )
            if selected == kind:
                return function()
            self._empty(selected)

    def forward(self, *args, **kwargs):
        if self.accounting is None:
            return self._join(_TARGET, lambda: self.target(*args, **kwargs))
        from .trace import target_call_id

        phase, options, payload = self.accounting.prepare(args, kwargs)
        kind = _PREFILL if phase == "prefill" else _TARGET
        result = self._join(kind, lambda: self.target(*args, **options))
        call = target_call_id(self.accounting.worker, self.call)
        self.accounting.capture(call, payload)
        return result

    def forward_spec(self, *args, **kwargs):
        start = kwargs.get("start_pos", args[2] if len(args) > 2 else 0)
        if start == 0:
            return self.proposal(*args, **kwargs)
        return self._join(_PROPOSAL, lambda: self.proposal(*args, **kwargs))

    @torch.inference_mode()
    def execute(self, operation):
        if self.active:
            raise RuntimeError("DEP4 execution steps cannot overlap")
        self.active = True
        try:
            result = operation()
            self._join(_DONE)
            if self.accounting is not None:
                self.accounting.flush()
            # Rank acknowledgements must follow all NCCL and cache writes.
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return result
        finally:
            self.active = False
