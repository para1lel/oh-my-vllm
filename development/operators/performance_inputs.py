import hashlib
from dataclasses import dataclass, field

import numpy as np

from .cases import load_cases
from .fixtures import Fixture, load_fixture
from .storage import decode


@dataclass
class Inputs:
    inputs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)

    def add(self, group, name, value, dtype=None):
        value = np.ascontiguousarray(value)
        getattr(self, group)[name] = value
        self.dtypes[group + "." + name] = dtype or str(value.dtype)

    def write(self, name, slots, layout="rows"):
        self.writes[name] = dict(
            indices=np.asarray(slots, dtype=np.int64).tolist(),
            layout=layout,
            output=name,
        )


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def random_hidden(
    data, rng, rows, names=("value", "residual"), width=5120, *, fp32_residual=False
):
    for name in names:
        values = rng.standard_normal((rows, width), dtype=np.float32) * 0.5
        if name == "residual" and fp32_residual:
            data.add("inputs", name, values)
        else:
            data.add("inputs", name, bf16(values), "bfloat16")


def cache(data, case, prototype, weights):
    shape = case.shape
    batch, queries, length = (shape[name] for name in ("batch", "queries", "context"))
    chain = case.boundary == "mtp.chain"
    pages = (length + int(chain) * 3 + 783) // 784
    blocks = batch * pages + 2
    tables = np.zeros((batch, 335), np.int32)
    for request in range(batch):
        tables[request, :pages] = np.arange(
            1 + request * pages, 1 + (request + 1) * pages
        )
    writes = []
    for stage in range(4 if chain else 1):
        prefix = f"stage{stage}." if chain else ""
        position = np.tile(
            np.arange(length - queries + stage, length + stage, dtype=np.int64), batch
        )
        selected = np.concatenate(
            [
                tables[b, position[b * queries : (b + 1) * queries] // 784] * 784
                + position[b * queries : (b + 1) * queries] % 784
                for b in range(batch)
            ]
        ).astype(np.int32)
        writes.append(selected)
        for name, value in dict(
            positions=np.tile(position, (3, 1)),
            slots=selected,
            tables=tables,
            sequence_rows=np.arange(batch + 1, dtype=np.int32) * queries,
            lengths=(position + 1).astype(np.int32),
        ).items():
            data.add("inputs", prefix + name, value)
    assigned = np.stack(writes[1:], axis=1).reshape(-1) if chain else writes[0]
    if case.options.get("variant") == "query_only":
        assigned = []
    raw = prototype.state["kv"][:1]
    kv = np.tile(raw, (blocks, 1, 1, 1, 1))
    layers = case.weights if case.boundary == "fa" else ("mtp",)
    for layer in layers:
        prefix = f"layer{layer}." if len(layers) > 1 else ""
        data.add("state", prefix + "kv", kv, "bfloat16")
        data.write(prefix + "kv", assigned, "kv")


def recurrent(data, case, rng):
    batch, queries, length = (
        case.shape[name] for name in ("batch", "queries", "context")
    )
    rows = case.shape["rows"]
    start = length - queries
    verify = case.options["verify"]
    checkpoints = (
        []
        if verify
        else [
            row
            for row in range(rows)
            if (start + row % queries + 1) % 784 == 0 or row % queries == queries - 1
        ]
    )
    selected = list(range(rows)) if verify else checkpoints
    assigned = np.arange(batch + 1, batch + 1 + len(selected), dtype=np.int32)
    for name, value in dict(
        sequence_rows=np.arange(batch + 1, dtype=np.int32) * queries,
        positions=np.full(batch, start, np.int32),
        read_slots=np.arange(1, batch + 1, dtype=np.int32),
        write_slots=assigned if verify else np.full(rows, -1, np.int32),
        checkpoint_rows=np.asarray(checkpoints, np.int32),
        checkpoint_slots=np.empty(0, np.int32) if verify else assigned,
        tokens=(np.arange(rows, dtype=np.int64) * 17 + 760) % 248320,
    ).items():
        data.add("inputs", name, value)
    for name, dimensions in (
        ("recurrent", (48, 128, 128)),
        ("convolution", (10240, 3)),
    ):
        value = bf16(
            rng.standard_normal(
                (batch + len(selected) + 2, *dimensions), dtype=np.float32
            )
            * 0.02
        )
        if start == 0:
            value[1 : batch + 1] = 0
        dtype = "bfloat16"
        if name == "recurrent":
            value = decode(value, dtype).astype(np.float32)
            dtype = "float32"
        for layer in case.weights:
            prefix = f"layer{layer}." if len(case.weights) > 1 else ""
            data.add("state", prefix + name, value, dtype)
            data.write(prefix + name, assigned)


def make(case, weights):
    numerical = next(
        item
        for item in load_cases()
        if item.id == case.id.replace(".performance.", ".numerics.")
    )
    prototype = load_fixture(numerical)
    loaded = weights.identity["files"]
    if any(
        loaded.get(name) != identity
        for name, identity in prototype.manifest["checkpoint"]["files"].items()
    ):
        raise ValueError("performance prototype uses different checkpoint files")
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(case.id.encode()).digest()[:8])
    )
    data = Inputs()
    boundary, shape = case.boundary, case.shape
    rows = shape.get("rows", 0)
    if boundary in {
        "fa",
        "gdn",
        "ffn",
        "output.hidden",
        "output.distribution",
        "output.sample",
    }:
        random_hidden(data, rng, rows, fp32_residual=boundary != "ffn")
    if boundary == "fa":
        cache(data, case, prototype, weights)
    elif boundary == "gdn":
        recurrent(data, case, rng)
    elif boundary == "output.hidden":
        selected = (
            list(range(rows))
            if rows <= 20
            else [row for row in range(rows) if (row + 1) % 784 == 0 or row == rows - 1]
        )
        slots = np.full(rows, -1, np.int32)
        slots[selected] = np.arange(1, len(selected) + 1)
        data.add("inputs", "slots", slots)
        data.add(
            "state",
            "hidden_cache",
            bf16(
                rng.standard_normal((len(selected) + 2, 5120), dtype=np.float32) * 0.25
            ),
            "bfloat16",
        )
        data.write("hidden_cache", slots[selected])
    elif boundary in {"mtp.context", "mtp.draft", "mtp.chain"}:
        cache(data, case, prototype, weights)
        random_hidden(data, rng, rows, ("hidden",))
        data.add(
            "inputs", "tokens", (np.arange(rows, dtype=np.int64) * 17 + 760) % 248320
        )
        if boundary == "mtp.context":
            indices = np.arange(rows, dtype=np.int32) - 1
            indices[:: shape["queries"]] = -1
            data.add("inputs", "hidden_rows", indices)
            data.add(
                "inputs",
                "boundary_rows",
                np.repeat(np.arange(shape["batch"], dtype=np.int32), shape["queries"]),
            )
            data.add(
                "inputs",
                "boundary_hidden",
                bf16(
                    rng.standard_normal((shape["batch"], 5120), dtype=np.float32) * 0.5
                ),
                "bfloat16",
            )
    if boundary in {
        "output.distribution",
        "output.sample",
        "mtp.draft",
        "mtp.chain",
        "mtp.accept",
    }:
        for group in ("inputs", "state"):
            for name, value in getattr(prototype, group).items():
                if name not in getattr(data, group):
                    data.add(group, name, value.copy(), prototype.dtype(group, name))
        for name, write in prototype.manifest["writes"].items():
            if name not in data.writes:
                data.writes[name] = write
        if boundary == "mtp.chain":
            allowed = np.zeros(248320, np.uint8)
            allowed[
                [
                    3,
                    7,
                    19,
                    31,
                    89,
                    257,
                    761,
                    1601,
                    4099,
                    8191,
                    16703,
                    32771,
                    65537,
                    100003,
                    180001,
                    248319,
                ]
            ] = 1
            data.inputs["mask"][:] = np.packbits(allowed, bitorder="little")
        if (
            boundary in {"output.sample", "output.distribution", "mtp.draft"}
            and not case.options.get("greedy", False)
            and case.options.get("sampling") != "smooth"
        ):
            data.inputs["parameters"][:] = np.array(
                [0.8, 12, 0.92, 0.01, 0.1, 0.15, 1.05], np.float32
            )
        if boundary.startswith("output.") and case.options["greedy"]:
            data.inputs["value"].fill(0)
            data.inputs["residual"].fill(0)
    elif boundary.startswith("vision."):
        for name, value in prototype.inputs.items():
            data.add("inputs", name, value.copy(), prototype.dtype("inputs", name))
    manifest = dict(
        case=case.definition(),
        checkpoint=weights.identity,
        prototype_sha256=numerical.fixture_sha256,
        writes=data.writes,
        tensors={name: dict(tensor_dtype=dtype) for name, dtype in data.dtypes.items()},
    )
    return Fixture(manifest, data.inputs, data.state, {})
