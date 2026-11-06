from dataclasses import dataclass
from math import prod

import numpy as np

from ...operators.fa_fp8.copy import copy_prefix as copy_fp8_prefix
from ...operators.gdn_radix.api import Schedule
from .cache_budget import CHECKPOINT_TOKENS, PAGE_TOKENS, cache_layout
from .cache_wire import Endpoint, endpoint, sequence_plan


@dataclass
class Batch:
    scheduled: list
    requests: list
    schedule: Schedule
    tokens: np.ndarray
    metadata: tuple
    writes: tuple
    plans: tuple
    export_copies: tuple

    @property
    def rows(self):
        return len(self.tokens)

    @property
    def max_queries(self):
        return int(max(np.diff(self.schedule.sequence_rows)))

    @property
    def write_mask(self):
        return tuple(bool(value) for value in self.metadata[5])


@dataclass(frozen=True)
class PageState:
    version: int
    index: int
    valid: int
    mtp_valid: int = 0


class Cache:
    @staticmethod
    def layout(config):
        return cache_layout(config)

    def __init__(self, device, config):
        self.device = device
        self.layout = cache_layout(config)
        self.blocks = self.layout.pages
        self.mamba_blocks = self.layout.checkpoints
        self.parallel = self.layout.parallel
        self.private_start = self.layout.private_start
        self.scratch_start = self.layout.scratch_start
        self.slots = self.layout.slots
        self.fa = tuple(
            (
                device.zeros(self.layout.codes_shape, "uint8"),
                device.zeros(self.layout.scales_shape, "float32"),
            )
            for _ in range(self.layout.fa_layers)
        )
        self.state = tuple(
            device.zeros(self.layout.state_shape, "float32") for _ in range(48)
        )
        self.convolution = tuple(
            device.zeros(self.layout.convolution_shape, "bfloat16") for _ in range(48)
        )
        self.hidden = device.zeros(self.layout.hidden_shape, "bfloat16")
        self.head_hidden = (
            device.zeros(self.layout.hidden_shape, "bfloat16")
            if self.layout.fa_layers == 17
            else self.hidden
        )
        self.versions = {0: 0}
        self.checkpoints = {0: Endpoint(0, 0, 0)}
        self.mtp_contexts = {}
        self.pages = {}

    def state_views(self, slot):
        if type(slot) is not int or not 0 < slot < self.slots:
            raise ValueError("invalid GDN checkpoint slot")
        hidden = (
            (self.hidden, self.head_hidden)
            if self.head_hidden is not self.hidden
            else (self.hidden,)
        )
        return tuple(
            self.device.view(value, value.shape[1:], slot * prod(value.shape[1:]))
            for value in (*self.state, *self.convolution, *hidden)
        )

    def copy_endpoint(self, source, destination):
        self.check_endpoint(source)
        self.check_endpoint(destination, written=False)
        if source.end != destination.end:
            raise ValueError("checkpoint copy must preserve the exact token endpoint")
        context = self.mtp_contexts.get(source.slot)
        if source.slot != destination.slot:
            for target, value in zip(
                self.state_views(destination.slot),
                self.state_views(source.slot),
                strict=True,
            ):
                self.device.copy(target, value)
        self.device.synchronize_compute()
        self.record_endpoint(destination)
        if context is not None:
            self.mtp_contexts[destination.slot] = context

    def record_endpoint(self, value):
        self.versions[value.slot] = value.version
        self.checkpoints[value.slot] = value
        self.mtp_contexts.pop(value.slot, None)

    def check_endpoint(self, value, *, written=True):
        version = self.versions.get(value.slot)
        if version is not None and value.version < version:
            raise ValueError("stale checkpoint slot version")
        if written and self.checkpoints.get(value.slot) != value:
            raise ValueError("checkpoint source has no completed matching endpoint")

    def copy_states(self, copies):
        states = {}
        for copy in copies:
            source, destination, length = (
                copy["source_slot"],
                copy["destination_slot"],
                copy["valid_tokens"],
            )
            if (
                type(source) is not int
                or type(destination) is not int
                or type(length) is not int
                or not 0 < source <= self.blocks
                or not 0 < destination <= self.blocks
                or source == destination
                or not 0 < length < PAGE_TOKENS
            ):
                raise ValueError("invalid FP8 KV prefix copy")
            record = self.pages.get(source)
            if (
                record is None
                or record.version != copy["source_version"]
                or record.valid < length
            ):
                raise ValueError("KV copy source has no completed matching prefix")
            previous = self.pages.get(destination)
            if previous is not None and previous.version >= copy["destination_version"]:
                raise ValueError(
                    "KV copy destination version is not a fresh reservation"
                )
            if destination in states:
                raise ValueError("KV copies must have distinct destination slots")
            states[destination] = PageState(
                copy["destination_version"],
                record.index,
                length,
                min(record.mtp_valid, length),
            )
        return states

    def copy_prefix(self, copies):
        states = self.copy_states(copies)
        for copy in copies:
            for codes, scales in self.fa:
                copy_fp8_prefix(
                    codes,
                    scales,
                    copy["source_slot"],
                    copy["destination_slot"],
                    copy["valid_tokens"],
                )
        if copies:
            self.device.synchronize_compute()
        self.pages.update(states)

    def validate_pages(self, plans, copies=()):
        pending = self.copy_states(copies)
        for plan in plans:
            for index, page in enumerate(plan.pages):
                if page is None:
                    continue
                record = pending.get(page["slot"], self.pages.get(page["slot"]))
                prefix = page["valid_tokens"]
                if record is not None and page["version"] < record.version:
                    raise ValueError("stale KV page generation")
                if prefix and (
                    record is None
                    or record.version != page["version"]
                    or record.index != index
                    or record.valid < prefix
                ):
                    raise ValueError("KV page has no completed matching prefix")

    def batch(self, scheduled, requests, verify):
        plans = tuple(sequence_plan(row, self.layout) for row in scheduled)
        starts, reads, writes, tokens, positions, slots, lengths, masks = (
            [0],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        checkpoints, checkpoint_slots, copies = [], [], []
        tables = np.zeros((len(plans), self.layout.table_width), dtype=np.int32)
        for index, (plan, request) in enumerate(zip(plans, requests, strict=True)):
            if plan.read.end != plan.start:
                raise ValueError("forward must start at its actual resume checkpoint")
            self.check_endpoint(plan.read)
            reads.append(plan.read.slot)
            tables[index] = [
                page["slot"] if page is not None else 0 for page in plan.pages
            ]
            first = len(tokens)
            exports = self._row_exports(plan, first, copies)
            for offset, token in enumerate(plan.row["token_ids"]):
                position = plan.start + offset
                tokens.append(token)
                positions.append(request.position(position))
                page = plan.pages[position // PAGE_TOKENS]
                slots.append(page["slot"] * PAGE_TOKENS + position % PAGE_TOKENS)
                lengths.append(position + 1)
                masks.append(plan.write_mask[offset])
                slot = exports.get(position + 1, -1)
                writes.append(slot if verify else -1)
                if not verify and slot >= 0:
                    checkpoints.append(first + offset)
                    checkpoint_slots.append(slot)
            starts.append(len(tokens))
        schedule = Schedule(
            tuple(starts),
            tuple(reads),
            tuple(writes),
            tuple(checkpoints),
            tuple(checkpoint_slots),
            tuple(plan.start for plan in plans),
            self.slots,
            verify,
        )
        actual_writes = schedule.validate()
        metadata = (
            np.asarray(positions, dtype=np.int64).T.copy(),
            np.asarray(slots, dtype=np.int32),
            tables,
            np.asarray(starts, dtype=np.int32),
            np.asarray(lengths, dtype=np.int32),
            np.asarray(masks, dtype=np.int8),
        )
        return Batch(
            scheduled,
            requests,
            schedule,
            np.asarray(tokens, dtype=np.int64),
            metadata,
            actual_writes,
            plans,
            tuple(copies),
        )

    def _row_exports(self, plan, first, copies):
        exports = {}
        refs = {}
        for target in plan.exports:
            if not plan.start < target.end <= plan.end:
                raise ValueError("forward export lies outside the computed rows")
            self.check_endpoint(target, written=False)
            if target.end in exports:
                copies.append((refs[target.end], target))
            else:
                exports[target.end], refs[target.end] = target.slot, target
        required = {
            plan.end,
            *range(
                (plan.start // CHECKPOINT_TOKENS + 1) * CHECKPOINT_TOKENS,
                plan.end + 1,
                CHECKPOINT_TOKENS,
            ),
        }
        if not required.issubset(exports):
            raise ValueError("forward requires periodic and exact tail exports")
        if plan.row["sampling"]["kind"] == "verify" and any(
            end not in exports for end in range(plan.start + 1, plan.end + 1)
        ):
            raise ValueError("verification requires every private endpoint")
        return exports

    def complete_exports(self, batch):
        self.device.synchronize_compute()
        written = set(batch.writes)
        for plan in batch.plans:
            for value in plan.exports:
                if value.slot in written:
                    self.record_endpoint(value)
        for source, destination in batch.export_copies:
            self.copy_endpoint(source, destination)

    def commit(self, request, batch, index, accepted):
        plan = batch.plans[index]
        selected = plan.end if accepted is None else plan.start + accepted + 1
        endpoints = {value.end: value for value in plan.exports}
        if selected not in endpoints:
            raise ValueError("accepted state has no exact export")
        current = endpoints[selected]
        request.endpoints = endpoints
        request.state_slot, request.state_position = current.slot, current.end
        request.computed_end = selected
        request.evaluated_end = plan.end
        request.kv_valid_end = selected
        self.record_pages(plan, selected)

    def record_pages(self, plan, end):
        for index, page in enumerate(plan.pages):
            if page is None or page["access"] != "write":
                continue
            valid = max(0, min(PAGE_TOKENS, end - index * PAGE_TOKENS))
            if plan.row["purpose"] != "finalize":
                valid = max(valid, page["valid_tokens"])
            old = self.pages.get(page["slot"])
            mtp = (
                min(old.mtp_valid, valid)
                if old is not None and old.version == page["version"]
                else 0
            )
            self.pages[page["slot"]] = PageState(page["version"], index, valid, mtp)

    def restore_context(self, row):
        value = row["model_state"]["mtp_endpoint"]
        if value is None:
            if row["forward_start"]:
                raise ValueError("MTP resume has no completed shifted context")
            return 0
        target = Endpoint(value["end"] + 1, value["slot"], value["version"])
        self.check_endpoint(target)
        if self.mtp_contexts.get(target.slot) != value["end"]:
            raise ValueError("MTP resume context has not completed")
        return value["end"]

    def validate_context(self, slots, writes):
        for physical, write in zip(slots, writes, strict=True):
            if write:
                continue
            page = self.pages.get(int(physical) // PAGE_TOKENS)
            if page is None or page.mtp_valid <= int(physical) % PAGE_TOKENS:
                raise ValueError("MTP reuse has no completed matching KV vector")

    def complete_context(self, batch):
        self.device.synchronize_compute()
        for plan, request in zip(batch.plans, batch.requests, strict=True):
            for page in plan.pages:
                if page is None or page["slot"] not in self.pages:
                    continue
                old = self.pages[page["slot"]]
                valid = max(
                    0,
                    min(
                        PAGE_TOKENS, request.mtp_position + 1 - old.index * PAGE_TOKENS
                    ),
                )
                self.pages[page["slot"]] = PageState(
                    old.version, old.index, old.valid, max(old.mtp_valid, valid)
                )
            for value in plan.exports:
                if value.end - 1 <= request.mtp_position:
                    self.check_endpoint(value)
                    self.mtp_contexts[value.slot] = value.end - 1

    def endpoint_source(self, row, request):
        state = row["model_state"]["endpoint_source"]
        if state["kind"] == "saved_hidden":
            source = endpoint(state, self.slots, self.layout.max_context)
        elif state["kind"] == "forward":
            source = request.endpoints.get(row["endpoint"])
            if source is None:
                raise ValueError("exact endpoint is unavailable; scheduler must replay")
        else:
            raise ValueError("invalid checkpoint endpoint source")
        if source.end != row["endpoint"]:
            raise ValueError("saved endpoint does not match the requested end")
        self.check_endpoint(source)
        return source

    def export_endpoint(self, row, request):
        plan = sequence_plan(row, self.layout)
        source = self.endpoint_source(row, request)
        for target in plan.exports:
            self.check_endpoint(target, written=False)
            if target.end != source.end:
                raise ValueError("endpoint export must preserve the accepted token end")
        for target in plan.exports:
            self.copy_endpoint(source, target)
        request.endpoints = {source.end: plan.exports[-1] if plan.exports else source}
        request.state_slot = request.endpoints[source.end].slot
        request.state_position = request.computed_end = request.kv_valid_end = (
            source.end
        )
        request.drafts = []
        self.record_pages(plan, source.end)
        return plan


def page_valid_ends(plan, end):
    result = []
    for index, page in enumerate(plan.pages):
        if page is None:
            continue
        start = index * PAGE_TOKENS
        valid = max(start, min(end, start + PAGE_TOKENS))
        if plan.row["purpose"] != "finalize":
            valid = max(start + page["valid_tokens"], valid)
        result.append(
            {
                "page_index": index,
                "slot": page["slot"],
                "version": page["version"],
                "valid_end": valid,
            }
        )
    return result
