"""Data-only CPU continuation checkpoints for the long-context reference."""

import hashlib
import json
import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .identity import file_state, open_file, parse_json, unchanged

STATE_FIELDS = (
    "window_kv",
    "pending_values",
    "pending_gates",
    "compress_kv",
    "index_k",
)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value)).hexdigest()


def completed_lengths(metadata, identity):
    lengths = identity["checkpoints"]
    cursor, endpoint, phase = (
        metadata[name] for name in ("cursor", "next_endpoint", "phase")
    )
    if (
        type(cursor) is not int
        or type(endpoint) is not int
        or not 0 < cursor <= len(identity["tokens"])
        or not 0 <= endpoint <= len(lengths)
    ):
        raise ValueError("invalid long-context checkpoint cursor")
    if phase == "complete":
        valid = endpoint == len(lengths) and cursor == lengths[-1]
    elif endpoint < len(lengths) and phase in {"prefill", "probe"}:
        lower = lengths[endpoint - 1] if endpoint else 0
        valid = (
            cursor == lengths[endpoint]
            if phase == "probe"
            else lower <= cursor < lengths[endpoint]
            and (cursor - lower) % identity["execution"]["prefill_chunk_size"] == 0
        )
    else:
        valid = False
    if not valid:
        raise ValueError("invalid long-context checkpoint phase")
    completed = lengths[:endpoint]
    expected_endpoints = {
        f"{kind}_{length}" for length in completed for kind in ("prefill", "decode")
    }
    if set(metadata["endpoints"]) != expected_endpoints or set(
        metadata["decode_tokens"]
    ) != {str(length) for length in completed}:
        raise ValueError("incomplete long-context checkpoint endpoints")
    return completed


def state_shapes(metadata, text):
    cursor = metadata["cursor"]
    history = metadata["engram_history"]
    history_size = (
        min(cursor, text["engram_max_ngram_size"] - 1)
        if text["engram_layer_ids"]
        else 0
    )
    if len(history) != history_size or any(
        type(token) is not int or not 0 <= token < text["engram_compressed_vocab_size"]
        for token in history
    ):
        raise ValueError("invalid long-context checkpoint Engram history")
    shapes = {}
    if len(metadata["layers"]) != text["num_hidden_layers"]:
        raise ValueError("invalid long-context checkpoint layer count")
    for index, layer in enumerate(metadata["layers"]):
        if (
            type(layer["layer"]) is not int
            or type(layer["length"]) is not int
            or layer["layer"] != index
            or layer["length"] != cursor
        ):
            raise ValueError("invalid long-context checkpoint layer cursor")
        ratio = text["compress_ratios"][index]
        owner = index in text.get("kv_source_layer_ids", [])
        expected = {
            "window_kv": (min(cursor, text["sliding_window"]), text["head_dim"]),
            "pending_values": (cursor % ratio, text["head_dim"]) if owner else None,
            "pending_gates": (cursor % ratio, text["head_dim"]) if owner else None,
            "compress_kv": (cursor // ratio, text["head_dim"]) if owner else None,
            "index_k": (cursor // ratio, text["index_head_dim"]) if owner else None,
        }
        for field, shape in expected.items():
            key = f"layer_{index}_{field}"
            if layer[field] != (key if shape is not None else None):
                raise ValueError("invalid long-context checkpoint state field")
            if shape is not None:
                shapes[key] = shape
    return shapes


def validate_point(metadata, arrays, identity):
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("invalid long-context checkpoint schema")
    if metadata["identity"] != identity:
        raise ValueError("long-context checkpoint identity differs")
    elapsed = metadata["generation_seconds"]
    if type(elapsed) not in (int, float) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid long-context checkpoint generation time")
    completed = completed_lengths(metadata, identity)
    text = identity["config"]["text_config"]
    shapes = state_shapes(metadata, text)
    shapes.update(
        (name, (text["vocab_size"],))
        for name in ("last_logits", *metadata["endpoints"])
    )
    if set(arrays) != set(shapes) or set(metadata["arrays"]) != set(shapes):
        raise ValueError("invalid long-context checkpoint array set")
    for name, shape in shapes.items():
        value, record = arrays[name], metadata["arrays"][name]
        if (
            value.dtype != np.float64
            or value.shape != shape
            or not np.isfinite(value).all()
            or record != {"shape": list(shape), "sha256": digest(value)}
        ):
            raise ValueError(f"invalid long-context checkpoint array: {name}")
    for length in completed:
        token = metadata["decode_tokens"][str(length)]
        if type(token) is not int or token != int(arrays[f"prefill_{length}"].argmax()):
            raise ValueError("invalid long-context checkpoint decode token")


def pack(point, identity):
    arrays = {"last_logits": np.asarray(point["last_logits"]), **point["arrays"]}
    layers = []
    for layer in point["state"].layers:
        record = {"layer": layer.layer, "length": layer.length}
        for field in STATE_FIELDS:
            value = getattr(layer, field)
            key = f"layer_{layer.layer}_{field}"
            record[field] = None if value is None else key
            if value is not None:
                arrays[key] = np.asarray(value)
        layers.append(record)
    metadata = {
        "schema_version": 1,
        "identity": identity,
        "cursor": point["cursor"],
        "next_endpoint": point["next_endpoint"],
        "phase": point["phase"],
        "generation_seconds": point["generation_seconds"],
        "engram_history": list(point["state"].engram_history),
        "layers": layers,
        "decode_tokens": point["decode_tokens"],
        "endpoints": sorted(point["arrays"]),
        "arrays": {
            name: {"shape": list(value.shape), "sha256": digest(value)}
            for name, value in arrays.items()
        },
    }
    if point["state"].length != point["cursor"]:
        raise ValueError("long-context checkpoint state differs from cursor")
    validate_point(metadata, arrays, identity)
    return metadata, arrays


def unpack(metadata, arrays):
    import torch

    from .reference_attention import AttentionState
    from .reference_model import ModelState

    layers = tuple(
        AttentionState(
            layer["layer"],
            layer["length"],
            **{
                field: None
                if layer[field] is None
                else torch.from_numpy(arrays[layer[field]])
                for field in STATE_FIELDS
            },
        )
        for layer in metadata["layers"]
    )
    return {
        "cursor": metadata["cursor"],
        "next_endpoint": metadata["next_endpoint"],
        "phase": metadata["phase"],
        "generation_seconds": metadata["generation_seconds"],
        "state": ModelState(
            metadata["cursor"], layers, tuple(metadata["engram_history"])
        ),
        "last_logits": arrays["last_logits"],
        "arrays": {name: arrays[name] for name in metadata["endpoints"]},
        "decode_tokens": metadata["decode_tokens"],
    }


class Checkpoint:
    def __init__(
        self,
        path,
        identity,
        *,
        checkpoint_root=None,
        final_output=None,
        validate_inputs=lambda: None,
    ):
        path = Path(path)
        self.path = path.parent.resolve(strict=True) / path.name
        self.previous = self.path.with_name(self.path.stem + ".previous.npz")
        self.identity = parse_json(json.dumps(identity, allow_nan=False))
        self.validate_inputs = validate_inputs
        self.generation_seconds = 0.0
        self._loaded = False
        self._snapshot = None
        repository = Path(__file__).resolve().parents[3]
        if self.path.is_relative_to(repository) or (
            checkpoint_root is not None
            and self.path.is_relative_to(Path(checkpoint_root).resolve())
        ):
            raise ValueError(
                "long-context checkpoint must be external to source and weights"
            )
        if final_output is not None and Path(final_output) in (
            self.path,
            self.previous,
        ):
            raise ValueError("long-context checkpoint conflicts with final output")
        self._check_paths()

    def _check_paths(self):
        for path in (self.path, self.previous):
            if (
                path.suffix != ".npz"
                or path.is_symlink()
                or (path.exists() and not path.is_file())
            ):
                raise ValueError("long-context checkpoint must be a regular NPZ file")

    def load(self):
        self._check_paths()
        if not self.path.exists():
            if self.previous.exists():
                raise ValueError(
                    "long-context checkpoint missing; select previous explicitly"
                )
            self._loaded = True
            return None
        try:
            with (
                open_file(self.path) as source,
                np.load(source, allow_pickle=False) as stored,
            ):
                snapshot = file_state(os.fstat(source.fileno()))
                if len(stored.files) != len(set(stored.files)):
                    raise ValueError("duplicate checkpoint arrays")
                encoded_value = stored["metadata"]
                if encoded_value.shape != () or encoded_value.dtype.kind != "U":
                    raise ValueError("checkpoint metadata must be a JSON string")
                encoded = encoded_value.item()
                if (
                    hashlib.sha256(encoded.encode()).hexdigest()
                    != stored["metadata_sha256"].item()
                ):
                    raise ValueError("checkpoint metadata digest differs")
                metadata = parse_json(encoded)
                if metadata["identity"] != self.identity:
                    raise ValueError("checkpoint identity differs")
                arrays = {
                    name: stored[name]
                    for name in stored.files
                    if name not in {"metadata", "metadata_sha256"}
                }
                unchanged(self.path, source, snapshot)
            validate_point(metadata, arrays, self.identity)
            self.generation_seconds = metadata["generation_seconds"]
            self._snapshot = snapshot
            self._loaded = True
            return unpack(metadata, arrays)
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            raise ValueError(f"invalid long-context checkpoint: {error}") from error

    def save(self, point):
        self._check_paths()
        if not self._loaded:
            self.load()
        metadata, arrays = pack(point, self.identity)
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with NamedTemporaryFile(dir=self.path.parent, delete=False) as temporary:
            name = Path(temporary.name)
            backup = name.with_suffix(".previous")
            try:
                np.savez(
                    temporary,
                    metadata=np.asarray(encoded),
                    metadata_sha256=np.asarray(
                        hashlib.sha256(encoded.encode()).hexdigest()
                    ),
                    **arrays,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
                self.validate_inputs()
                self._check_paths()
                snapshot = file_state(self.path.stat()) if self.path.exists() else None
                if snapshot != self._snapshot:
                    raise ValueError(
                        "long-context checkpoint changed before replacement"
                    )
                if self.path.exists():
                    os.link(self.path, backup)
                    os.replace(backup, self.previous)
                    self._sync_directory()
                os.replace(name, self.path)
                self._sync_directory()
                self._snapshot = file_state(self.path.stat())
                self.generation_seconds = metadata["generation_seconds"]
            finally:
                name.unlink(missing_ok=True)
                backup.unlink(missing_ok=True)

    def _sync_directory(self):
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
