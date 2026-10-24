"""Prepared communication at the staged block boundaries.

The caller agrees on counts and owner-major row identities before preparation,
and binds the communicator to the block's four-rank placement. TP4/TEP4 rows
are identical on all ranks. DEP4 rows are contiguous owner intervals.
All operations use the caller stream; Eager waits for host enqueue, not device
completion. Drain the communicator before Eager reuse or prepare_capture().
The caller owns completion, communicator abort, and all-rank status acceptance.
On any capture error, terminate the rank without unwinding CUDA resources.
Retain this workspace and its block through Eager completion or graph release.
"""

import time
from contextlib import contextmanager

from ...operators.routing.api import EncodedDispatchInputs
from ...operators.runtime.collectives import Route
from ...operators.transform import api as transform
from ...operators.transform.plan import make_plan
from ...operators.transform.workspace import prepare as prepare_transform
from .block import Outputs
from .workspace import Workspace as RankWorkspace
from .workspace import _span


class Workspace(RankWorkspace):
    def __init__(self, block, communicator, counts, allocate, byte_limit):
        block.check_open()
        self.block, self.communicator = block, communicator
        self.rank = block.owner.plan.rank
        self.split = block.owner.plan.strategy == "DEP4"
        counts = tuple(counts)
        if (
            len(counts) != 4
            or any(type(count) is not int or not 0 <= count <= 384 for count in counts)
            or counts[self.rank] != block.plan.rows
            or (not self.split and len(set(counts)) != 1)
        ):
            raise ValueError("collective row counts differ from the block placement")
        total = sum(counts) if self.split else counts[0]
        if total != block.moe.dispatch.plan.rows:
            raise ValueError("collective routed rows differ from the block")
        reads = [*block._allocations, *block.owner.tensors.values()]
        reads.extend(
            value
            for weight in block.owner.packed.values()
            for value in (weight.encoded, weight.scales)
        )
        super().__init__(block.owner, reads, allocate, byte_limit)
        self.prepared, self.exchanges, self.copies = [], [], []
        self.widen_status = ()
        quantized, routes = (
            block.moe.input_quantization.outputs,
            block.moe.route.outputs,
        )
        self.local_packet = EncodedDispatchInputs(
            quantized.encoded, quantized.scales, quantized.status, *routes
        )
        self.global_packet = self.local_packet
        self.attention_value = block.attention.outputs.value
        self.shared_value = block.moe.outputs.shared
        with self.preparing():
            if self.split:
                self._exchange(counts)
            else:
                self.attention_widen, self.attention_reduce, self.attention_value = (
                    self._widen()
                )
                self.shared_widen, self.shared_reduce, self.shared_value = self._widen()
                self.widen_status = (
                    self.attention_widen.outputs.status,
                    self.shared_widen.outputs.status,
                )
            self.routed_reduce, routed = self._reduce(block.moe.outputs.routed)
            first = sum(counts[: self.rank]) if self.split else 0
            self.routed_value = self.device.view(
                routed, (block.plan.rows, 5120), first * 5120
            )

    def _flat(self, value):
        return self.device.view(value, (value.numel(),))

    def _prepare(self, kind, tensors, routes=()):
        staging = sum(
            (last - first + 15) // 16 * 16
            for first, last in map(_span, tensors)
            if first != last and (first % 16 or (last - first) % 16)
        )
        if self.allocated_bytes + staging > self.byte_limit:
            raise ValueError("collective staging exceeds the workspace byte limit")
        plan = (
            self.communicator.routing(routes)
            if kind == "routing"
            else self.communicator.prepare(kind, *tensors)
        )
        self.allocated_bytes += staging
        self.prepared.append(plan)
        return plan

    def _reduce(self, value):
        result = self.reserve(tuple(value.shape), "float32")
        plan = self._prepare("all_reduce", (self._flat(value), self._flat(result)))
        return plan, result

    def _widen(self):
        workspace = self.part(
            prepare_transform,
            make_plan(kind="widen", rows=self.block.plan.rows, width=5120),
            self.device,
        )
        plan, result = self._reduce(workspace.outputs.value)
        return workspace, plan, result

    def _exchange(self, counts):
        total = sum(counts)
        self.global_packet = EncodedDispatchInputs(
            *(
                self.reserve((total, field.shape[1]), str(field.dtype))
                for field in self.local_packet
            )
        )
        offsets = [sum(counts[:rank]) for rank in range(4)]
        for source, destination in zip(
            self.local_packet, self.global_packet, strict=True
        ):
            if len(set(counts)) == 1:
                self.exchanges.append(
                    self._prepare(
                        "all_gather", (self._flat(source), self._flat(destination))
                    )
                )
                continue
            width = source.shape[1]
            local = self.device.view(
                destination, tuple(source.shape), offsets[self.rank] * width
            )
            self.copies.append((local, source))
            for step in range(1, 4):
                send, recv = (self.rank + step) % 4, (self.rank - step) % 4
                received = self.device.view(
                    destination, (counts[recv] * width,), offsets[recv] * width
                )
                tensors = (self._flat(source), received)
                self.exchanges.append(
                    self._prepare(
                        "routing",
                        tensors,
                        (
                            Route("send", send, tensors[0]),
                            Route("recv", recv, tensors[1]),
                        ),
                    )
                )

    def _run(self, plan, capture):
        plan.run()
        end = time.monotonic() + 30
        while True:
            status = self.communicator.poll_enqueue()
            if status == "ready":
                break
            if status != "pending":
                raise RuntimeError(f"unexpected collective enqueue status: {status}")
            if time.monotonic() >= end:
                raise TimeoutError("block collective host enqueue deadline exceeded")
            time.sleep(0.001)
        if capture:
            plan.finish_enqueue()
        elif self.communicator.poll() not in ("ready", "pending"):
            raise RuntimeError("block collective completion failed")

    def attention(self, *, capture=False):
        self.check_open()
        if not self.split:
            self.attention_widen.run(
                transform.Inputs(self.block.attention.outputs.value, None)
            )
            self._run(self.attention_reduce, capture)
        return self.attention_value

    def packet(self, *, capture=False):
        self.check_open()
        for destination, source in self.copies:
            self.device.module.copy_bytes(destination, source)
        for plan in self.exchanges:
            self._run(plan, capture)
        return self.global_packet

    def finish(self, *, capture=False):
        self.check_open()
        if not self.split:
            self.shared_widen.run(transform.Inputs(self.block.moe.outputs.shared, None))
            self._run(self.shared_reduce, capture)
        self._run(self.routed_reduce, capture)
        output = self.block.finish(self.routed_value, self.shared_value)
        return Outputs(
            output.residual, output.next_pre, (*output.status, *self.widen_status)
        )

    def prepare_capture(self):
        self.check_open()
        for plan in self.prepared:
            plan.prepare_capture()

    @contextmanager
    def graph_lease(self):
        with self.block.graph_lease(), super().graph_lease():
            yield


def prepare(block, communicator, *, counts, allocate=None, byte_limit=64 << 20):
    return Workspace(block, communicator, counts, allocate, byte_limit)
