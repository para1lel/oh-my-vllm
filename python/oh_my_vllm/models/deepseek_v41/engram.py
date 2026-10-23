"""Stream verified encoded Engram tensors into one sealed shared store."""

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

from ...operators.engram.shared import SharedWeightFile

MAX_HEADER_BYTES = 64 << 20
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}


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
            source = os.fdopen(descriptor, "rb")
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


@dataclass(frozen=True)
class Table:
    rows: int
    table_offset: int
    scale_offset: int


@dataclass
class EngramWeights:
    shared: SharedWeightFile
    tables: dict
    identity: dict

    def close(self):
        self.shared.close()

    def __enter__(self):
        self.shared.fileno()
        return self

    def __exit__(self, *args):
        self.close()


def _layout(config):
    text = config.get("text_config", {})
    if (
        config.get("architectures") != ["DeepseekV41ForCausalLM"]
        or config.get("model_type") != "deepseek_v41"
        or type(text) is not dict
        or text.get("engram_head_dim") != 256
    ):
        raise ValueError("unsupported Engram configuration")
    layers, rows = text.get("engram_layer_ids"), text.get("engram_num_embeddings")
    if (
        type(layers) is not list
        or type(rows) is not list
        or not 1 <= len(layers) <= 40
        or len(layers) != len(rows)
        or any(type(layer) is not int or not 0 <= layer < 40 for layer in layers)
        or len(set(layers)) != len(layers)
        or any(type(count) is not int or not 1 <= count < 1 << 31 for count in rows)
    ):
        raise ValueError("invalid Engram layers or row counts")
    return {
        f"layers.{layer}.engram.embed.{suffix}": (layer, count, dtype, width)
        for layer, count in zip(layers, rows, strict=True)
        for suffix, dtype, width in (
            ("weight", "F8_E4M3", 256),
            ("scale", "F8_E8M0", 8),
        )
    }


def _spans(source, index, layout):
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
        if name in layout:
            _, count, dtype, width = layout[name]
            if kind != dtype or shape != [count, width]:
                raise ValueError("Engram tensor layout differs from configuration")
        spans.append((*offsets, name))
    cursor, selected = 0, []
    for begin, end, name in sorted(spans):
        if begin != cursor:
            raise ValueError("checkpoint tensors overlap or leave a gap")
        cursor = end
        if name in layout:
            selected.append((length + 8 + begin, length + 8 + end, name))
    if length + 8 + cursor != source.initial[2]:
        raise ValueError("checkpoint payload length differs")
    source.unchanged()
    return selected


def _copy(source, spans, shared, offsets, buffer):
    source.source.seek(0)
    digest, position = hashlib.sha256(), 0
    buffer = memoryview(buffer)
    while position < source.initial[2]:
        remaining = min(len(buffer), source.initial[2] - position)
        count = source.source.readinto(buffer[:remaining])
        if not count:
            raise ValueError("truncated checkpoint payload")
        view = buffer[:count]
        digest.update(view)
        for begin, end, name in spans:
            left, right = max(begin, position), min(end, position + count)
            if left < right:
                offsets.setdefault(name, shared.written)
                shared.append(view[left - position : right - position])
        position += count
    source.verify(digest.hexdigest())


def load_engram(root, identities, *, limit, read_bytes=64 << 20):
    """The caller supplies trusted file identities and one shared-host budget.

    No CUDA allocation occurs. Registration follows successful sealing. Closing
    this owner releases its fd; existing rank mappings retain the sealed pages.
    """
    if (
        type(read_bytes) is not int
        or not 1 <= read_bytes <= SharedWeightFile.max_write_bytes
    ):
        raise ValueError("Engram stream buffer exceeds its byte limit")
    if type(identities) is not dict:
        raise ValueError("checkpoint identities must contain a file map")
    root = Path(root).resolve(strict=True)
    with ExitStack() as resources:
        sources = {}

        def source(name):
            if name not in sources:
                sources[name] = _Source(root, name, identities, resources)
            return sources[name]

        layout = _layout(source("config.json").control())
        index = source("model.safetensors.index.json").control().get("weight_map")
        if type(index) is not dict or {
            name for name in index if ".engram.embed." in name
        } != set(layout):
            raise ValueError("Engram checkpoint index differs from configuration")
        filenames = [index[name] for name in layout]
        if any(
            type(name) is not str or not name.endswith(".safetensors")
            for name in filenames
        ):
            raise ValueError("invalid Engram checkpoint shard")
        shards = [
            (source(name), _spans(source(name), index, layout))
            for name in sorted(set(filenames))
        ]
        size = sum(count * width for _, count, _, width in layout.values())
        shared = SharedWeightFile(size, limit=limit)
        try:
            offsets, buffer = {}, bytearray(read_bytes)
            for opened, spans in shards:
                _copy(opened, spans, shared, offsets, buffer)
            for opened in sources.values():
                opened.unchanged()
            shared.seal()
            tables = {
                layer: Table(
                    count,
                    offsets[f"layers.{layer}.engram.embed.weight"],
                    offsets[f"layers.{layer}.engram.embed.scale"],
                )
                for layer, count, _, _ in layout.values()
            }
            identity = {
                name: opened.expected["sha256"] for name, opened in sources.items()
            }
            return EngramWeights(shared, tables, identity)
        except BaseException:
            shared.close()
            raise
