"""Read caller-selected raw tensors from authenticated checkpoint files."""

import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_HEADER_BYTES = 64 << 20
MAX_TILE_BYTES = 64 << 20
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
RAW_DTYPES = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": "u1", "F8_E8M0": "u1", "I8": "i1"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint JSON key")
        result[key] = value
    return result


def _json(data):
    result = json.loads(data, object_pairs_hook=_object)
    if type(result) is not dict:
        raise ValueError("checkpoint JSON must contain an object")
    return result


def _state(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _Source:
    def __init__(self, root, name, identities, resources):
        if type(name) is not str or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ValueError("invalid checkpoint file name")
        expected = identities.get(name)
        if (
            type(expected) is not dict
            or type(expected.get("size")) is not int
            or expected["size"] <= 0
            or type(expected.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        ):
            raise ValueError("checkpoint file requires a verified size and SHA256")
        self.expected = dict(expected)
        self.path = root / name
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            source = os.fdopen(descriptor, "rb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        self.source = resources.enter_context(source)
        info = os.fstat(descriptor)
        self.initial = _state(info)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected["size"]:
            raise ValueError("checkpoint file type or size differs")
        self.unchanged()

    def unchanged(self):
        if (
            _state(os.fstat(self.source.fileno())) != self.initial
            or _state(self.path.stat(follow_symlinks=False)) != self.initial
        ):
            raise ValueError("checkpoint file changed during loading")

    def verify(self, digest):
        if digest != self.expected["sha256"]:
            raise ValueError("checkpoint SHA256 differs")
        self.unchanged()

    def control(self):
        if self.initial[2] > MAX_HEADER_BYTES:
            raise ValueError("checkpoint JSON exceeds its byte limit")
        data = self.source.read(self.initial[2] + 1)
        self.verify(hashlib.sha256(data).hexdigest())
        return _json(data)


def _header(source, index):
    prefix = source.source.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated checkpoint header")
    length = struct.unpack("<Q", prefix)[0]
    if not 0 < length <= min(MAX_HEADER_BYTES, source.initial[2] - 8) or length % 8:
        raise ValueError("invalid checkpoint header length")
    data = source.source.read(length)
    if len(data) != length:
        raise ValueError("truncated checkpoint header")
    header = _json(data)
    metadata = header.pop("__metadata__", {})
    if type(metadata) is not dict or any(
        type(value) is not str for value in metadata.values()
    ):
        raise ValueError("invalid checkpoint metadata")
    if set(header) != {
        name for name, shard in index.items() if shard == source.path.name
    }:
        raise ValueError("checkpoint header differs from index")
    spans = []
    for name, entry in header.items():
        if type(entry) is not dict or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("invalid checkpoint tensor record")
        kind, shape, offsets = (
            entry[key] for key in ("dtype", "shape", "data_offsets")
        )
        if (
            type(kind) is not str
            or kind not in DTYPE_BYTES
            or type(shape) is not list
            or any(type(size) is not int or size < 0 for size in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[kind]
        ):
            raise ValueError("invalid checkpoint tensor storage")
        spans.append((*offsets, name))
    cursor = 0
    for begin, end, _ in sorted(spans):
        if begin != cursor:
            raise ValueError("checkpoint tensors overlap or leave a gap")
        cursor = end
    if length + 8 + cursor != source.initial[2]:
        raise ValueError("checkpoint payload length differs")
    source.unchanged()
    return header, length + 8


@dataclass(frozen=True)
class TensorSlice:
    """Coordinates and shape use stored elements, including packed FP4 bytes."""

    dtype: str
    shape: tuple
    start: tuple
    stop: tuple

    def __post_init__(self):
        if type(self.dtype) is not str or self.dtype not in RAW_DTYPES:
            raise ValueError("unsupported raw checkpoint dtype")
        for field in ("shape", "start", "stop"):
            value = getattr(self, field)
            if not isinstance(value, (tuple, list)) or any(
                type(part) is not int for part in value
            ):
                raise ValueError("tensor slice coordinates must be integer sequences")
            object.__setattr__(self, field, tuple(value))
        if (
            len(self.shape) not in (1, 2)
            or len(self.start) != len(self.shape)
            or len(self.stop) != len(self.shape)
            or any(
                not 0 <= first < last <= size
                for size, first, last in zip(
                    self.shape, self.start, self.stop, strict=True
                )
            )
        ):
            raise ValueError("tensor slice must select a nonempty vector or matrix")


class EncodedReader:
    """One caller; raw destinations stay caller-owned and must not be in flight.

    Whole touched shards are hashed before data is returned. Call finish before
    publishing weights. No architecture, placement, decoding, or promotion is
    inferred here. Source buffer and destination each have a 64 MiB cap.
    """

    def __init__(self, root, verified_files, selections, *, read_bytes=64 << 20):
        if type(read_bytes) is not int or not 1 <= read_bytes <= MAX_TILE_BYTES:
            raise ValueError("checkpoint stream buffer exceeds its byte limit")
        if type(verified_files) is not dict or type(selections) is not dict:
            raise ValueError(
                "checkpoint identities and selections must be file/tensor maps"
            )
        if any(
            type(name) is not str or not name or type(spec) is not TensorSlice
            for name, spec in selections.items()
        ):
            raise ValueError(
                "checkpoint selections require names and TensorSlice values"
            )
        self.root = Path(root).resolve(strict=True)
        self.identities, self.selections = (
            copy.deepcopy(verified_files),
            dict(selections),
        )
        self.read_bytes, self.buffer = read_bytes, None
        self.sources, self.headers = {}, {}
        self.resources, self.closed = ExitStack(), False
        try:
            self.config = self._source("config.json").control()
            self.index = (
                self._source("model.safetensors.index.json").control().get("weight_map")
            )
            if type(self.index) is not dict or any(
                type(name) is not str
                or not name
                or type(shard) is not str
                or re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", shard) is None
                for name, shard in self.index.items()
            ):
                raise ValueError("invalid checkpoint weight index")
            if not self.selections.keys() <= self.index.keys():
                raise ValueError("checkpoint index omits a selected tensor")
            self._unchanged()
        except BaseException:
            self.close()
            raise

    def _source(self, name):
        if name not in self.sources:
            self.sources[name] = _Source(
                self.root, name, self.identities, self.resources
            )
        return self.sources[name]

    def _unchanged(self):
        if self.closed:
            raise RuntimeError("encoded checkpoint reader is closed")
        for source in self.sources.values():
            source.unchanged()

    def _prepare(self, filename):
        source = self._source(filename)
        if filename not in self.headers:
            source.source.seek(0)
            header, base = _header(source, self.index)
            for name, spec in self.selections.items():
                if self.index[name] == filename and (
                    header[name]["dtype"] != spec.dtype
                    or tuple(header[name]["shape"]) != spec.shape
                ):
                    raise ValueError("checkpoint tensor differs from caller selection")
            if self.buffer is None:
                self.buffer = bytearray(self.read_bytes)
            source.source.seek(0)
            digest, position = hashlib.sha256(), 0
            while position < source.initial[2]:
                view = memoryview(self.buffer)[
                    : min(self.read_bytes, source.initial[2] - position)
                ]
                count = source.source.readinto(view)
                if not count:
                    raise ValueError("truncated checkpoint payload")
                digest.update(view[:count])
                position += count
            source.verify(digest.hexdigest())
            self.headers[filename] = header, base
        return source, *self.headers[filename]

    def _read(self, source, offset, destination):
        source.source.seek(offset)
        position = 0
        while position < len(destination):
            count = source.source.readinto(
                destination[position : position + self.read_bytes]
            )
            if not count:
                raise ValueError("truncated checkpoint tensor")
            position += count

    def _matrix(self, source, offset, spec, rows, destination):
        item = DTYPE_BYTES[spec.dtype]
        stride, first, last = spec.shape[1] * item, spec.start[1], spec.stop[1]
        if first == 0 and last == spec.shape[1]:
            self._read(source, offset, memoryview(destination[:rows]).cast("B"))
        elif stride > self.read_bytes:
            for row in range(rows):
                self._read(
                    source,
                    offset + row * stride + first * item,
                    memoryview(destination[row]).cast("B"),
                )
        else:
            step = self.read_bytes // stride
            for row in range(0, rows, step):
                count = min(step, rows - row)
                view = memoryview(self.buffer)[: count * stride]
                self._read(source, offset + row * stride, view)
                values = np.frombuffer(view, dtype=RAW_DTYPES[spec.dtype]).reshape(
                    count, spec.shape[1]
                )
                destination[row : row + count] = values[:, first:last]

    def read_into(self, name, row_start, destination):
        """Fill a local row prefix; return its length and leave the tail unchanged."""
        self._unchanged()
        if type(name) is not str or name not in self.selections:
            raise ValueError("unknown or nonowned checkpoint tensor")
        spec = self.selections[name]
        if (
            type(row_start) is not int
            or not 0 <= row_start <= spec.stop[0] - spec.start[0]
            or not isinstance(destination, np.ndarray)
            or destination.dtype != np.dtype(RAW_DTYPES[spec.dtype])
            or destination.ndim != len(spec.shape)
            or not destination.flags.c_contiguous
            or not destination.flags.writeable
            or destination.nbytes > MAX_TILE_BYTES
            or (
                destination.ndim == 2
                and destination.shape[1] != spec.stop[1] - spec.start[1]
            )
        ):
            raise ValueError("invalid checkpoint row or raw destination")
        source, header, base = self._prepare(self.index[name])
        self._unchanged()
        rows = min(destination.shape[0], spec.stop[0] - spec.start[0] - row_start)
        stride = math.prod(spec.shape[1:]) * DTYPE_BYTES[spec.dtype]
        offset = (
            base
            + header[name]["data_offsets"][0]
            + (spec.start[0] + row_start) * stride
        )
        if rows and len(spec.shape) == 1:
            self._read(source, offset, memoryview(destination[:rows]).cast("B"))
        elif rows:
            self._matrix(source, offset, spec, rows, destination)
        self._unchanged()
        return rows

    def finish(self):
        self._unchanged()
        if (
            self.sources.keys()
            - self.headers.keys()
            - {
                "config.json",
                "model.safetensors.index.json",
            }
        ):
            raise ValueError("checkpoint shard verification did not complete")
        return {
            name: source.expected["sha256"] for name, source in self.sources.items()
        }

    def close(self):
        self.closed = True
        self.resources.close()
        self.buffer = None

    def __enter__(self):
        self._unchanged()
        return self

    def __exit__(self, *args):
        self.close()
