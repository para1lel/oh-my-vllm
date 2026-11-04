"""Concurrent dense owners with lockstep expert-only empty ranks."""

import torch
import torch.distributed as dist

from .dispatch import DispatchStatus, ExpertExchange

_DONE = 0
_TARGET = 1
_PREFILL = 2
_PROPOSAL = 3


class Dep4Executor:
    """Align whole target/proposal calls without taking request ownership."""

    def __init__(
        self,
        model,
        *,
        group=None,
        padded=False,
        check_round_status=None,
        workspace_budget_bytes=512 << 20,
    ):
        from .model import Transformer

        self.model = model
        self.group = group
        self.device = next(model.parameters()).device
        self.check_round_status = (
            self.device.type == "cuda"
            if check_round_status is None
            else check_round_status
        )
        self.sparse_control = isinstance(model, Transformer)
        self.padded = padded and self.device.type == "cuda"
        self.status = DispatchStatus(self.device) if self.check_round_status else None
        self.exchange = ExpertExchange(
            group=group, status=self.status, budget_bytes=workspace_budget_bytes
        )
        self.eligibility = {}
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

    def _padded_ready(self):
        if not self.padded or self.status is None:
            return False
        for layer in (*self.model.layers, *self.model.mtp):
            moe = layer.ffn
            state = getattr(moe, "_routed_state", None)
            if state is None:
                return False
            previous = self.eligibility.get(id(moe))
            if (
                previous is not None
                and previous[0] is state
                and previous[1] == state.generation
            ):
                if not previous[2]:
                    return False
                continue
            if state.weights is None:
                moe.prepare_routed_experts()
            weights = state.weights
            supported = weights is not False and (
                weights.gate[0][0].is_cuda and weights.gate[0][0].size(0) in (576, 2304)
            )
            self.eligibility[id(moe)] = (state, state.generation, supported)
            if not supported:
                return False
        return True

    def _pending(self, kind, source_rows):
        local = torch.tensor(
            [kind, source_rows, int(self._padded_ready())],
            dtype=torch.int32,
            device=self.device,
        )
        gathered = torch.empty(
            (dist.get_world_size(self.group), 3), dtype=torch.int32, device=self.device
        )
        dist.all_gather_into_tensor(gathered.flatten(), local, group=self.group)
        metadata = gathered.cpu().tolist()
        pending = [row[0] for row in metadata]
        selected = min((value for value in pending if value != _DONE), default=_DONE)
        maximum = max((row[1] for row in metadata if row[0] == selected), default=0)
        return (
            selected,
            pending.count(selected),
            maximum,
            all(row[2] for row in metadata),
        )

    def _empty(self, kind):
        layers = self.model.layers if kind in (_TARGET, _PREFILL) else self.model.mtp
        for layer in layers:
            moe = layer.ffn
            rows = torch.empty((0, moe.dim), dtype=torch.bfloat16, device=self.device)
            ids = torch.empty(
                (0, moe.n_activated_experts), dtype=torch.int32, device=self.device
            )
            weights = torch.empty_like(ids, dtype=torch.float32)
            self.exchange(moe, rows, (weights, ids))

    def _round_status(self):
        from .kernel import _disable_sparse_native, _sparse_status_accumulator

        sparse = (
            _sparse_status_accumulator(self.device).to(torch.int32).reshape(1)
            if self.sparse_control
            else torch.zeros_like(self.status.tensor)
        )
        flags = torch.cat((self.status.tensor.ne(0).to(torch.int32), sparse))
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.group)
        self.status.tensor.bitwise_or_(flags[:1])
        rejected, sparse_rejected = flags.cpu().tolist()
        if rejected or sparse_rejected:
            reasons = []
            if rejected:
                reasons.append("invalid padded expert metadata")
            if sparse_rejected:
                _disable_sparse_native()
                reasons.append("native sparse attention failure")
            raise RuntimeError(
                f"DEP4 round rejected: {', '.join(reasons)}; no cache commit or result"
            )

    def _join(self, kind, function=None, *, source_rows=0):
        if not self.active:
            raise RuntimeError("DEP4 model calls require a Rust-owned execution step")
        while True:
            selected, active_ranks, maximum, fixed = self._pending(kind, source_rows)
            if selected == _DONE:
                return None
            self.exchange.prepare_round(maximum, fixed=fixed)
            if self.check_round_status and self.sparse_control:
                from .kernel import reset_sparse_status

                reset_sparse_status()
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
            result = function() if selected == kind else self._empty(selected)
            if self.check_round_status:
                self._round_status()
            if selected == kind:
                return result

    def _source_rows(self, args, kwargs, *, proposal=False):
        inputs = args[0] if args else kwargs["input_ids"]
        count = inputs.numel()
        if inputs.is_floating_point():
            count //= inputs.size(-1)
        if proposal:
            count *= getattr(self.model.mtp[0], "block_size", 1)
        return count

    def _target(self, args, kwargs):
        if self.check_round_status and self.sparse_control:
            kwargs = {**kwargs, "check_status": False}
        return self.target(*args, **kwargs)

    def forward(self, *args, **kwargs):
        rows = self._source_rows(args, kwargs)
        if self.accounting is None:
            return self._join(
                _TARGET, lambda: self._target(args, kwargs), source_rows=rows
            )
        from .trace import target_call_id

        phase, options, payload = self.accounting.prepare(args, kwargs)
        kind = _PREFILL if phase == "prefill" else _TARGET
        result = self._join(kind, lambda: self._target(args, options), source_rows=rows)
        call = target_call_id(self.accounting.worker, self.call)
        self.accounting.capture(call, payload)
        return result

    def forward_spec(self, *args, **kwargs):
        start = kwargs.get("start_pos", args[2] if len(args) > 2 else 0)
        if start == 0:
            return self.proposal(*args, **kwargs)
        options = (
            {**kwargs, "check_status": False}
            if self.check_round_status and self.sparse_control
            else kwargs
        )
        return self._join(
            _PROPOSAL,
            lambda: self.proposal(*args, **options),
            source_rows=self._source_rows(args, kwargs, proposal=True),
        )

    @torch.inference_mode()
    def execute(self, operation):
        if self.active:
            raise RuntimeError("DEP4 execution steps cannot overlap")
        self.active = True
        try:
            if self.status is not None:
                self.status.begin_execute()
            result = operation()
            self._join(_DONE)
            if self.accounting is not None:
                self.accounting.flush()
            # Rank acknowledgements must follow all NCCL and cache writes.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return result
        finally:
            self.active = False
