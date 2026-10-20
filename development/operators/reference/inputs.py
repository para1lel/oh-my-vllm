import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Data:
    inputs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)

    def add(self, group, name, value, dtype=None):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        dtype = dtype or str(value.dtype)
        if dtype == "bfloat16":
            value = (
                torch.from_numpy(value).to(torch.bfloat16).view(torch.uint16).numpy()
            )
        getattr(self, group)[name] = np.ascontiguousarray(value)
        self.dtypes[group + "." + name] = dtype

    def write(self, name, indices, layout="rows", output=None):
        self.writes[name] = {
            "indices": [int(index) for index in indices],
            "layout": layout,
            "output": output or name,
        }


def hidden(data, rng, rows, *, fp32_residual=False):
    for name in ("value", "residual"):
        value = rng.normal(0, 0.5, (rows, 5120))
        if name == "residual" and fp32_residual:
            data.add("inputs", name, value.astype(np.float32))
        else:
            data.add("inputs", name, value, "bfloat16")


def attention_metadata(data, case, rng, weights, *, mtp=False, chain=False):
    shape = case.shape
    batch, queries, length = shape["batch"], shape["queries"], shape["context"]
    capacity = length + (3 if chain else 0)
    pages = (capacity + 783) // 784
    blocks = batch * pages + 2
    tables = np.zeros((batch, 335), np.int32)
    for index in range(batch):
        tables[index, :pages] = np.arange(index * pages + 1, (index + 1) * pages + 1)
    all_slots = []
    for stage in range(4 if chain else 1):
        prefix = f"stage{stage}." if chain else ""
        positions = np.tile(
            np.arange(length - queries + stage, length + stage, dtype=np.int64), batch
        )
        slots = np.concatenate(
            [
                tables[index, positions[index * queries : (index + 1) * queries] // 784]
                * 784
                + positions[index * queries : (index + 1) * queries] % 784
                for index in range(batch)
            ]
        ).astype(np.int32)
        all_slots.append(slots)
        for name, value in {
            "positions": np.tile(positions, (3, 1)),
            "slots": slots,
            "tables": tables,
            "sequence_rows": np.arange(batch + 1, dtype=np.int32) * queries,
            "lengths": (positions + 1).astype(np.int32),
        }.items():
            data.add("inputs", prefix + name, value)
    values = torch.from_numpy(rng.normal(0, 0.25, (blocks, 2, 784, 4, 256))).to("cuda")
    values = values.to(torch.bfloat16).to(torch.float64)
    data.add("state", "kv", values, "bfloat16")
    written = all_slots[0]
    if chain:
        written = np.stack(all_slots[1:], axis=1).reshape(-1)
    elif case.options.get("variant") == "query_only":
        written = []
    data.write("kv", written, "kv")


def recurrent_metadata(data, case, rng):
    shape = case.shape
    batch, queries, length = shape["batch"], shape["queries"], shape["context"]
    starts = np.arange(batch + 1, dtype=np.int32) * queries
    positions = np.full(batch, length - queries, np.int32)
    if case.options["verify"]:
        checkpoints = []
        selected = list(range(shape["rows"]))
    else:
        checkpoints = [
            row
            for b in range(batch)
            for row in range(starts[b], starts[b + 1])
            if (positions[b] + row - starts[b] + 1) % 784 == 0
            or row == starts[b + 1] - 1
        ]
        selected = checkpoints
    writes = np.full(shape["rows"], -1, np.int32)
    slots = np.arange(batch + 1, batch + 1 + len(selected), dtype=np.int32)
    if case.options["verify"]:
        writes[:] = slots
    checkpoint_slots = slots if not case.options["verify"] else np.empty(0, np.int32)
    for name, value in {
        "sequence_rows": starts,
        "positions": positions,
        "read_slots": np.arange(1, batch + 1, dtype=np.int32),
        "write_slots": writes,
        "checkpoint_rows": np.asarray(checkpoints, np.int32),
        "checkpoint_slots": checkpoint_slots,
        "tokens": (np.arange(shape["rows"], dtype=np.int64) * 17 + 760) % 248320,
    }.items():
        data.add("inputs", name, value)
    capacity = batch + len(selected) + 2
    for name, dimensions in (
        ("recurrent", (48, 128, 128)),
        ("convolution", (10240, 3)),
    ):
        value = rng.normal(0, 0.02, (capacity, *dimensions))
        for request in range(batch):
            if positions[request] == 0:
                value[request + 1] = 0
        if name == "recurrent":
            value = torch.from_numpy(value).to(torch.bfloat16).float().numpy()
            data.add("state", name, value, "float32")
        else:
            data.add("state", name, value, "bfloat16")
        data.write(name, slots)


def sampling(data, case, rows, *, greedy=False):
    history = np.zeros((rows, 2, 248320), np.int32)
    allowed = np.zeros((rows, 248320), np.uint8)
    ids = np.array(
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
    )
    if case.options.get("sampling") == "fixed_token":
        allowed[np.arange(rows), ids[:rows]] = 1
    else:
        allowed[:, ids] = 1
    parameters = np.tile(
        np.array([0 if greedy else 0.8, -1, 1, 0, 0, 0, 1], np.float32), (rows, 1)
    )
    if not greedy:
        history[:, 0, ids[::3]] = 1
        history[:, 1, ids[1::3]] = 2
        parameters[:] = np.array([0.8, 12, 0.92, 0.01, 0.1, 0.15, 1.05], np.float32)
        if case.options.get("sampling") == "smooth":
            parameters[:, :4] = np.array([1, -1, 1, 0], np.float32)
    data.add("inputs", "history", history)
    data.add("inputs", "mask", np.packbits(allowed, axis=1, bitorder="little"))
    data.add("inputs", "parameters", parameters)
    data.add(
        "state",
        "rng",
        np.array(
            [
                [seed, 8 + row * 4]
                for row, seed in enumerate(
                    case.options.get(
                        "rng_seeds", [17 + row * 31 for row in range(rows)]
                    )
                )
            ],
            np.uint64,
        ),
    )
    if case.boundary != "output.distribution":
        data.write("rng", range(rows))


def make(case, weights):
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(case.id.encode()).digest()[:8])
    )
    data = Data()
    boundary, shape = case.boundary, case.shape
    if boundary in {
        "fa",
        "gdn",
        "ffn",
        "output.hidden",
        "output.distribution",
        "output.sample",
    }:
        hidden(data, rng, shape["rows"], fp32_residual=boundary != "ffn")
    if boundary == "fa":
        attention_metadata(data, case, rng, weights)
    elif boundary == "gdn":
        recurrent_metadata(data, case, rng)
    elif boundary == "output.hidden":
        rows = shape["rows"]
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
            rng.normal(0, 0.25, (len(selected) + 2, 5120)),
            "bfloat16",
        )
        data.write("hidden_cache", slots[selected])
    elif boundary in {"output.sample", "output.distribution"}:
        greedy = case.options["greedy"]
        sampling(data, case, shape["rows"], greedy=greedy)
        if greedy:
            for name in ("value", "residual"):
                data.inputs[name].fill(0)
    elif boundary in {"mtp.context", "mtp.draft", "mtp.chain"}:
        attention_metadata(
            data, case, rng, weights, mtp=True, chain=boundary == "mtp.chain"
        )
        rows = shape["rows"]
        data.add("inputs", "hidden", rng.normal(0, 0.5, (rows, 5120)), "bfloat16")
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
                rng.normal(0, 0.5, (shape["batch"], 5120)),
                "bfloat16",
            )
        else:
            sampling(data, case, shape["batch"], greedy=boundary == "mtp.chain")
    elif boundary == "mtp.accept":
        acceptance(data, case)
    elif boundary.startswith("vision."):
        if boundary == "vision.input":
            data.add(
                "inputs",
                "pixels",
                rng.integers(
                    0,
                    256,
                    (1, shape["frames"], shape["height"] * 16, shape["width"] * 16, 3),
                    dtype=np.uint8,
                ),
            )
        else:
            data.add(
                "inputs",
                "hidden",
                rng.normal(0, 0.5, (shape["rows"], 1152)),
                "bfloat16",
            )
    return data


def acceptance(data, case):
    batch = case.shape["batch"]
    target = np.zeros((batch, 5, 248320), np.float32)
    proposal = np.zeros((batch, 4, 248320), np.float32)
    drafts = np.tile(np.array([3, 7, 19, 31, 0], np.int64), (batch, 1))
    for request in range(batch):
        for stage, token in enumerate(drafts[request, :4]):
            proposal[request, stage, token] = 0.5
            proposal[request, stage, 1601] = 0.5
            target[request, stage, token] = 0.75
            target[request, stage, 1601] = 0.25
        target[request, 4, 89] = 0.25
        target[request, 4, 257] = 0.75
        reject = {
            "all": 4,
            "first-rejection": 0,
            "partial": request + 1,
            "mixed": request,
            "inactive": 4,
        }[case.options["acceptance"]]
        if reject < 4:
            target[request, reject] = 0
            target[request, reject, 257] = 0.25
            target[request, reject, 761] = 0.75
        if case.options["acceptance"] == "inactive":
            active = request
            drafts[request, active:4] = -1
            target[request, active] = 0
            target[request, active, 89] = 1
    if case.options["acceptance"] == "mixed":
        target = target.argmax(axis=-1).astype(np.int64)
        drafts[:, 4] = 1
    for name, value in {
        "target": target,
        "proposal": proposal,
        "drafts": drafts,
        "private_slots": np.arange(1, batch * 5 + 1, dtype=np.int32).reshape(batch, 5),
        "capacity": np.array([batch * 5 + 2], np.int32),
    }.items():
        data.add("inputs", name, value)
    data.add(
        "state",
        "rng",
        np.array([[101 + row, 20 + row * 4] for row in range(batch)], np.uint64),
    )
    data.write("rng", range(batch))
