"""Credit owner routes locally and flush them after all DEP4 rounds finish."""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .trace import RoutingTrace


@dataclass
class _Call:
    phase: str
    requests: list = field(default_factory=list)
    routes: RoutingTrace | None = None
    explicit: bool = False
    credited: dict = field(default_factory=dict)
    counters: tuple | None = None


class Dep4Trace:
    def __init__(self, worker, output):
        self.worker = worker
        self.output = output
        self.calls = {}
        self.frontiers = {}

    def begin(self, call, phase):
        if call in self.calls:
            raise ValueError("DEP4 trace has a duplicate global target call")
        self.calls[call] = _Call(phase)

    def prepare(self, args, kwargs):
        inputs = args[0] if args else kwargs["input_ids"]
        slots = kwargs.get("cache_slots_cpu")
        positions = kwargs.get("cache_positions_cpu")
        start = kwargs.get("start_pos", args[1] if len(args) > 1 else 0)
        if slots is None or len(slots) != inputs.size(0):
            raise ValueError("DEP4 trace needs registered request cache slots")
        if positions is None:
            positions = (start,) * len(slots)
        owners = {
            request.cache_slot: (identity, request)
            for identity, request in self.worker.requests.items()
            if getattr(request, "cache_slot", None) is not None
        }
        requests, phases = [], set()
        for index, (slot, position) in enumerate(zip(slots, positions, strict=True)):
            if slot not in owners:
                raise ValueError("DEP4 target call has no registered request")
            identity, request = owners[slot]
            end = position + inputs.size(1)
            credited_start = max(position, self.frontiers.get(identity, 0))
            phase = "prefill" if credited_start < request.prompt_len else "decode"
            if credited_start < end:
                if phase == "prefill" and end > request.prompt_len:
                    raise ValueError("DEP4 target call crosses the prefill boundary")
                phases.add(phase)
            requests.append(
                dict(
                    id=identity,
                    start=position,
                    end=end,
                    prompt=request.prompt_len,
                    first=index * inputs.size(1),
                    logits=kwargs.get("compute_logits", True),
                )
            )
        if len(phases) > 1:
            raise ValueError("DEP4 target call mixes prefill and decode")
        options = dict(kwargs)
        routes = options.get("routing_trace")
        explicit = routes is not None
        if routes is None:
            routes = RoutingTrace(
                len(self.worker.model.layers),
                inputs.numel(),
                rank=self.worker.rank,
                strategy="dep4",
                device=inputs.device,
            )
            options["routing_trace"] = routes
        routes.deferred = True
        return next(iter(phases), "prefill"), options, (requests, routes, explicit)

    def capture(self, call, payload):
        requests, routes, explicit = payload
        record = self.calls[call]
        if record.routes is not None:
            raise ValueError("DEP4 target call was captured twice")
        snapshot = RoutingTrace(
            routes.indices.size(0),
            routes.indices.size(1),
            rank=self.worker.rank,
            strategy="dep4",
            device=routes.indices.device,
        )
        snapshot.indices.copy_(routes.indices)
        record.requests, record.routes, record.explicit = requests, snapshot, explicit

    def _credit(self, record, request, row):
        identity = request["id"]
        if identity in record.credited:
            raise ValueError("DEP4 trace credits a request twice in one call")
        if row is None:
            record.credited[identity] = None
            return
        start, count = row["start"], row["count"]
        if (
            row["id"] != identity
            or count < 1
            or not request["start"] <= start < start + count <= request["end"]
            or start != self.frontiers.get(identity, 0)
        ):
            raise ValueError("DEP4 trace credit is not a contiguous executed range")
        if record.phase == "prefill":
            expected_logits = int(
                start + count == request["prompt"] and request["logits"]
            )
            if (
                row["accepted"]
                or row["proposed"]
                or row["logit_rows"] != expected_logits
            ):
                raise ValueError("DEP4 prefill credit has invalid draft or logit rows")
        elif (
            count != row["accepted"] + 1
            or not 0 <= row["accepted"] <= row["proposed"] <= 5
            or row["logit_rows"] != count
        ):
            raise ValueError("DEP4 decode credit has an invalid accepted mask")
        record.credited[identity] = dict(row)
        self.frontiers[identity] = start + count

    def _legacy(self, values):
        for event in values["requests"]:
            identity = event["id"]
            request = self.worker.requests[identity]
            for record in self.calls.values():
                if record.explicit:
                    continue
                for executed in record.requests:
                    if executed["id"] != identity or identity in record.credited:
                        continue
                    start = max(executed["start"], self.frontiers.get(identity, 0))
                    end = min(executed["end"], request.state_position)
                    row = None
                    if end > start:
                        count = end - start
                        prefill = record.phase == "prefill"
                        row = dict(
                            id=identity,
                            start=start,
                            count=count,
                            accepted=0 if prefill else count - 1,
                            proposed=0 if prefill else values.get("proposed", 0),
                            logit_rows=(
                                int(end == executed["prompt"] and executed["logits"])
                                if prefill
                                else count
                            ),
                        )
                    self._credit(record, executed, row)

    def _explicit(self, values):
        call = values["call"]
        record = self.calls.get(call)
        if record is None or not record.explicit or record.counters is not None:
            raise ValueError("DEP4 accounting call is missing or duplicated")
        if values["rank"] != self.worker.rank or values["phase"] != record.phase:
            raise ValueError("DEP4 accounting rank or target phase differs")
        rows = {row["id"]: row for row in values["request_rows"]}
        if len(rows) != len(values["request_rows"]) or set(rows) != {
            request["id"] for request in record.requests
        }:
            raise ValueError("DEP4 accounting rows differ from executed requests")
        for request in record.requests:
            self._credit(record, request, rows[request["id"]])
        assignments, dispatch, presence, mask = values["_deferred_routes"]
        valid = (mask == self._credit_mask(record)).all()
        record.counters = (torch.where(valid, assignments, -1), dispatch, presence)

    @staticmethod
    def _credit_mask(record):
        mask = torch.zeros(
            record.routes.indices.size(1),
            dtype=torch.bool,
            device=record.routes.indices.device,
        )
        for executed in record.requests:
            credited = record.credited[executed["id"]]
            if credited is not None:
                first = executed["first"] + credited["start"] - executed["start"]
                mask[first : first + credited["count"]] = True
        return mask

    def record(self, operation, **values):
        if operation == "accounting_call":
            self._explicit(values)
            return
        if operation == "target":
            self._legacy(values)
        self.output.record(operation, **values)

    @staticmethod
    def readback(packed):
        return packed.cpu().tolist()

    def flush(self):
        if not self.calls:
            return
        device = next(self.worker.model.parameters()).device
        layers = len(self.worker.model.layers)
        local, presences, rows = [], [], []
        for call, record in sorted(self.calls.items()):
            if set(record.credited) != {request["id"] for request in record.requests}:
                raise ValueError("DEP4 target call has no accepted-state credit")
            request_rows = [row for row in record.credited.values() if row is not None]
            rows.append((call, record.phase, request_rows))
            if record.routes is None:
                assignments = torch.zeros((layers, 4), dtype=torch.int32, device=device)
                dispatch = torch.zeros_like(assignments)
                presence = torch.zeros((layers, 384), dtype=torch.uint8, device=device)
            elif record.counters is not None:
                assignments, dispatch, presence = record.counters
            else:
                assignments, dispatch, presence = record.routes.device_counters(
                    self._credit_mask(record)
                )
            local.append((assignments, dispatch))
            presences.append(presence)
        presence = torch.stack(presences)
        dist.all_reduce(
            presence, op=dist.ReduceOp.MAX, group=self.worker.dep4_executor.group
        )
        rank = self.worker.rank
        touched = presence[:, :, rank * 96 : (rank + 1) * 96].sum(-1, dtype=torch.int32)
        packed = torch.stack(
            [
                torch.cat((assignments, dispatch, touched[index, :, None]), dim=1)
                for index, (assignments, dispatch) in enumerate(local)
            ]
        )
        for (call, phase, request_rows), counters in zip(
            rows, self.readback(packed), strict=True
        ):
            credited = sum(row["count"] for row in request_rows)
            if any(sum(layer[:4]) != credited * 6 for layer in counters):
                raise ValueError(
                    "DEP4 trace has incomplete routing layers or accepted rows"
                )
            self.output.record(
                "accounting_call",
                rank=rank,
                call=call,
                phase=phase,
                request_rows=request_rows,
                expert_assignments=[layer[:4] for layer in counters],
                dispatch_rows=[layer[4:8] for layer in counters],
                touched_experts=[layer[8] for layer in counters],
            )
        self.calls.clear()

    def close(self):
        self.output.close()
