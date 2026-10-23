"""Stream verified encoded Engram tensors into one sealed shared store."""

import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ...operators.engram.shared import SharedWeightFile
from .checkpoint import _header, _Source


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
    header, base = _header(source, index)
    selected = []
    for name, entry in header.items():
        if name in layout:
            _, count, dtype, width = layout[name]
            if entry["dtype"] != dtype or entry["shape"] != [count, width]:
                raise ValueError("Engram tensor layout differs from configuration")
            begin, end = entry["data_offsets"]
            selected.append((base + begin, base + end, name))
    return sorted(selected)


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
