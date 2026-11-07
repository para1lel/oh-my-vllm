from dataclasses import dataclass

import numpy as np
import torch

from ..qwen_fp8_reference import attention_bf16_vectors, quantize_bf16_vectors
from ..storage import decode, equal_bits, state_rows

PAGE_ROWS = 128
LEGACY_ROWS = 784


@dataclass(frozen=True)
class CacheData:
    codes: np.ndarray
    scales: np.ndarray
    tables: np.ndarray
    slots: dict
    written: np.ndarray


def _metadata(fixture, prefixes, tables, width):
    batch = len(tables)
    slots = {}
    for prefix in prefixes:
        source = fixture.inputs
        starts = source[prefix + "sequence_rows"]
        lengths = source[prefix + "lengths"]
        if not np.array_equal(starts, np.arange(batch + 1)):
            raise ValueError("MTP draft fixtures require one query per request")
        positions = lengths.astype(np.int64) - 1
        if np.any(positions < 0) or np.any(positions >= width * PAGE_ROWS):
            raise ValueError("MTP fixture query exceeds the repaged cache")
        original = source[prefix + "tables"]
        old_slots = original[np.arange(batch), positions // LEGACY_ROWS] * LEGACY_ROWS
        old_slots += positions % LEGACY_ROWS
        if not np.array_equal(old_slots, source[prefix + "slots"]):
            raise ValueError("MTP fixture slots differ from logical query positions")
        slots[prefix] = (
            tables[np.arange(batch), positions // PAGE_ROWS] * PAGE_ROWS
            + positions % PAGE_ROWS
        ).astype(np.int32)
    return slots


def repage(fixture, prefixes):
    legacy = fixture.state["kv"]
    if legacy.dtype != np.uint16 or legacy.shape[1:] != (2, LEGACY_ROWS, 4, 256):
        raise ValueError("MTP fixtures require raw BF16 page784 KV storage")
    tables = fixture.inputs[prefixes[0] + "tables"]
    batch = len(tables)
    lengths = np.stack([fixture.inputs[prefix + "lengths"] for prefix in prefixes])
    length = int(lengths.max())
    width = (length + PAGE_ROWS - 1) // PAGE_ROWS
    logical = np.arange(width * PAGE_ROWS)
    if width > 2048 or logical[-1] // LEGACY_ROWS >= tables.shape[1]:
        raise ValueError("MTP fixture context exceeds its stored page table")
    source_pages = tables[:, logical // LEGACY_ROWS]
    required = logical[None, :] < lengths.max(axis=0)[:, None]
    if (
        np.any(source_pages < 0)
        or np.any(source_pages >= len(legacy))
        or np.any((source_pages == 0) & required)
    ):
        raise ValueError("MTP fixture lacks a stored KV page for the repaged context")
    for prefix in prefixes:
        if not np.array_equal(
            fixture.inputs[prefix + "tables"][:, logical // LEGACY_ROWS], source_pages
        ):
            raise ValueError("MTP stages require the same stored context pages")
    bits = np.zeros((batch * width + 2, 2, PAGE_ROWS, 4, 256), np.uint16)
    bits[0] = legacy[0, :, :PAGE_ROWS]
    bits[-1] = legacy[-1, :, :PAGE_ROWS]
    source_slots = source_pages * LEGACY_ROWS + logical % LEGACY_ROWS
    for request in range(batch):
        rows = legacy[source_pages[request], :, logical % LEGACY_ROWS]
        bits[1 + request * width : 1 + (request + 1) * width] = rows.reshape(
            width, PAGE_ROWS, 2, 4, 256
        ).transpose(0, 2, 1, 3, 4)
    new_tables = np.arange(1, batch * width + 1, dtype=np.int32).reshape(batch, width)
    slots = _metadata(fixture, prefixes, new_tables, width)
    written = []
    write = fixture.manifest["writes"]["kv"]
    if write["layout"] != "kv" or write["output"] != "kv":
        raise ValueError("MTP fixture must identify its written KV rows")
    for old_slot in write["indices"]:
        locations = np.flatnonzero(source_slots.reshape(-1) == old_slot)
        if len(locations) != 1:
            raise ValueError("MTP fixture writable KV row must have one cache owner")
        written.append(PAGE_ROWS + int(locations[0]))
    codes, scales = quantize_bf16_vectors(bits)
    return CacheData(codes, scales, new_tables, slots, np.asarray(written, np.int64))


class FixtureCache:
    def __init__(self, prepared, prefixes):
        self.prepared = prepared
        self.initial = repage(prepared.fixture, prefixes)
        self.previous_observe = prepared.observe
        self.state_names = tuple(prepared.state)
        legacy = prepared.state.pop("kv")
        prepared.restores[:] = [
            (target, source)
            for target, source in prepared.restores
            if target is not legacy
        ]
        self.cache = tuple(
            self.upload(name, value, state=True)
            for name, value in (
                ("mtp.kv_codes", self.initial.codes),
                ("mtp.kv_scales", self.initial.scales),
            )
        )
        self.readonly = {}
        tables = self.upload("tables", self.initial.tables)
        self.readonly["tables"] = (tables, self.initial.tables)
        self.metadata = {}
        for prefix in prefixes:
            slots = self.upload(prefix + "slots", self.initial.slots[prefix])
            self.readonly[prefix + "slots"] = (slots, self.initial.slots[prefix])
            self.metadata[prefix] = (
                prepared.inputs[prefix + "positions"],
                slots,
                tables,
                prepared.inputs[prefix + "sequence_rows"],
                prepared.inputs[prefix + "lengths"],
            )
        prepared.observe = self.observe

    def upload(self, name, value, *, state=False):
        prepared = self.prepared
        host = np.require(value, requirements=["C", "W"])
        source = torch.from_numpy(host)
        target = prepared.device.empty(host.shape, str(source.dtype))
        prepared.device.copy(target, source)
        prepared.host.append(host)
        prepared.restores.append((target, source))
        if state:
            prepared.state[name] = target
        return target

    def cache_observation(self):
        codes, scales = (self.prepared.raw(value) for value in self.cache)
        rows = state_rows(codes, "kv")
        scale_rows = scales.transpose(0, 2, 1, 3).reshape(-1, 2, 4)
        selected = self.initial.written
        untouched = np.ones(len(rows), dtype=bool)
        untouched[selected] = False
        guards = equal_bits(
            rows[untouched], state_rows(self.initial.codes, "kv")[untouched]
        ) and equal_bits(
            scale_rows[untouched],
            self.initial.scales.transpose(0, 2, 1, 3).reshape(-1, 2, 4)[untouched],
        )
        bits = attention_bf16_vectors(rows[selected], scale_rows[selected])
        return decode(bits, "bfloat16"), guards

    def observe(self):
        prepared = self.prepared
        prepared.require_open()
        actual = {
            name: prepared.download(value) for name, value in prepared.outputs.items()
        }
        actual.update({name: observe() for name, observe in prepared.observers.items()})
        guards = []
        for name in self.state_names:
            if name == "kv":
                actual[name], guard = self.cache_observation()
            else:
                tensor = prepared.state[name]
                raw = prepared.raw(tensor)
                initial = prepared.fixture.state[name]
                write = prepared.fixture.manifest["writes"].get(name)
                guard = equal_bits(raw, initial)
                if write is not None:
                    rows, baseline = (
                        state_rows(value, write["layout"]) for value in (raw, initial)
                    )
                    selected = np.asarray(write["indices"], dtype=np.int64)
                    actual[write["output"]] = decode(
                        rows[selected], str(tensor.dtype).removeprefix("torch.")
                    )
                    untouched = np.ones(len(rows), dtype=bool)
                    untouched[selected] = False
                    guard = equal_bits(rows[untouched], baseline[untouched])
            guards.append(guard)
        readonly = [
            (value, prepared.fixture.inputs[name])
            for name, value in prepared.inputs.items()
        ]
        readonly.extend(self.readonly.values())
        if any(
            not equal_bits(prepared.raw(value), initial) for value, initial in readonly
        ):
            raise ValueError("MTP operator changed read-only fixture metadata or input")
        if "guards" in prepared.case.outputs:
            actual["guards"] = np.asarray(guards, dtype=bool)
        elif not all(guards):
            raise ValueError("MTP operator changed an unassigned cache row")
        if set(actual) != set(prepared.case.outputs):
            raise ValueError(
                "observations differ from the complete MTP boundary contract"
            )
        return actual

    def close(self):
        self.prepared.observe = self.previous_observe
        self.prepared = self.previous_observe = self.cache = None
        self.metadata.clear()
        self.readonly.clear()
