"""CPU-only canonical communication plans; tensor execution is separate."""

import hashlib
import struct
from dataclasses import dataclass

_DTYPES = {
    "f32": (0, 4),
    "bf16": (1, 2),
    "f16": (2, 2),
    "i32": (3, 4),
    "i64": (4, 8),
    "u8": (5, 1),
}
_OPS = {"all_reduce": 0, "all_gather": 1, "reduce_scatter": 2, "route": 3}


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _object(value, fields):
    _require(type(value) is dict and value.keys() == set(fields), "invalid plan fields")


def _count(value):
    _require(type(value) is int and 0 <= value < 1 << 64, "invalid element count")


@dataclass(frozen=True)
class CollectiveOperation:
    op: str
    dtype: str
    counts: tuple[int, ...]

    def buffer_counts(self, world_size, rank):
        if self.op == "route":
            return (
                sum(self.counts[rank * world_size : (rank + 1) * world_size]),
                sum(self.counts[rank::world_size]),
            )
        count = self.counts[0]
        if self.op == "all_gather":
            return count, count * world_size
        if self.op == "reduce_scatter":
            return count * world_size, count
        return count, count


@dataclass(frozen=True)
class CollectivePlan:
    world_size: int
    operations: tuple[CollectiveOperation, ...]

    @classmethod
    def from_wire(cls, value, *, world_size):
        _object(value, ("version", "world_size", "operations"))
        _require(
            type(value["version"]) is int and value["version"] == 1,
            "unsupported collective plan version",
        )
        _require(
            type(world_size) is int
            and world_size in (1, 4)
            and type(value["world_size"]) is int
            and value["world_size"] == world_size,
            "collective topology differs",
        )
        _require(
            type(value["operations"]) is list and len(value["operations"]) <= 4096,
            "invalid collective operations",
        )
        operations = []
        for operation in value["operations"]:
            _require(type(operation) is dict, "invalid collective operation")
            op, dtype = operation.get("op"), operation.get("dtype")
            _require(type(op) is str and op in _OPS, "invalid collective operation")
            _require(
                type(dtype) is str and dtype in _DTYPES, "invalid collective dtype"
            )
            _object(operation, ("op", "dtype", "counts" if op == "route" else "count"))
            if op == "route":
                matrix = operation["counts"]
                _require(
                    type(matrix) is list
                    and len(matrix) == world_size
                    and all(
                        type(row) is list and len(row) == world_size for row in matrix
                    ),
                    "invalid route count matrix",
                )
                counts = tuple(count for row in matrix for count in row)
            else:
                counts = (operation["count"],)
            for count in counts:
                _count(count)
            item = CollectiveOperation(op, dtype, counts)
            for rank in range(world_size):
                for count in item.buffer_counts(world_size, rank):
                    _require(
                        count * _DTYPES[dtype][1] <= (1 << 63) - 1,
                        "collective buffer exceeds signed byte range",
                    )
            operations.append(item)
        return cls(world_size, tuple(operations))

    @property
    def sha256(self):
        digest = hashlib.sha256(b"oh-my-vllm/collectives/v1\0")
        digest.update(struct.pack("<III", 1, self.world_size, len(self.operations)))
        for operation in self.operations:
            digest.update(bytes((_OPS[operation.op], _DTYPES[operation.dtype][0])))
            for count in operation.counts:
                digest.update(struct.pack("<Q", count))
        return digest.hexdigest()

    def buffer_counts(self, rank):
        _require(type(rank) is int and 0 <= rank < self.world_size, "invalid rank")
        return tuple(op.buffer_counts(self.world_size, rank) for op in self.operations)
