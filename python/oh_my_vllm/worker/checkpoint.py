import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from ..operators.specification import MODEL


def file_state(path):
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve()
        config = (self.path / "config.json").read_bytes()
        if hashlib.sha256(config).hexdigest() != MODEL["config_sha256"]:
            raise ValueError("checkpoint configuration differs from the model contract")
        self.identity = {"config": MODEL["config_sha256"]}
        index = (self.path / "model.safetensors.index.json").read_bytes()
        self.identity["index"] = hashlib.sha256(index).hexdigest()
        self.index = json.loads(index)["weight_map"]
        self.headers = {}
        self._ready_identity = None
        self._ready_states = {}
        self._ready_unread = set()

    def stream_identity(self):
        loaded = dict(self.identity)
        names = {"config": "config.json", "index": "model.safetensors.index.json"}
        observed = {}
        for name in loaded:
            with (self.path / names.get(name, name)).open("rb") as source:
                observed[name] = hashlib.file_digest(source, "sha256").hexdigest()
        if self.identity != loaded:
            raise ValueError("loaded checkpoint set changed during verification")
        return observed

    def verify_identity(self, expected=None):
        expected = dict(self.identity) if expected is None else expected
        if self.identity != expected:
            raise ValueError("loaded checkpoint set or identity changed")
        observed = self.stream_identity()
        if observed != expected:
            raise ValueError("checkpoint file bytes changed after loading")
        return observed

    def complete_identity(self):
        names = set(self.index.values())
        for name in names:
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".safetensors")
            ):
                raise ValueError("invalid checkpoint shard file name")
        snapshot = {name: file_state(self.path / name) for name in names}
        for name in sorted(names):
            if name not in self.identity:
                with (self.path / name).open("rb") as source:
                    self.identity[name] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()
        identity = self.verify_identity()
        if snapshot != {name: file_state(self.path / name) for name in names}:
            raise ValueError("checkpoint file bytes changed during readiness")
        self._ready_identity = dict(identity)
        self._ready_states = snapshot
        self._ready_unread = set(names)
        return identity

    def _check_ready_shard(self, filename):
        if self._ready_identity is not None and (
            filename not in self._ready_states
            or file_state(self.path / filename) != self._ready_states[filename]
        ):
            raise ValueError("checkpoint file bytes changed after readiness")

    def header(self, name):
        if name in self.headers:
            return self.headers[name]
        path = self.path / name
        with path.open("rb") as file:
            size = struct.unpack("<Q", file.read(8))[0]
            if size > 1 << 28:
                raise ValueError("checkpoint header is too large")
            result = json.loads(file.read(size)), 8 + size
        self.headers[name] = result
        return result

    def raw_tensor(self, name):
        filename = self.index[name]
        self._check_ready_shard(filename)
        if filename not in self.identity or filename in self._ready_unread:
            with (self.path / filename).open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if self._ready_identity is not None and digest != self._ready_identity.get(
                filename
            ):
                raise ValueError("checkpoint file bytes changed after readiness")
            self._check_ready_shard(filename)
            self.identity[filename] = digest
        header, base = self.header(filename)
        tensor = header[name]
        dtype = {"BF16": "<u2", "F8_E4M3": "u1", "F32": "<f4"}[tensor["dtype"]]
        begin, end = tensor["data_offsets"]
        array = np.memmap(
            self.path / filename,
            mode="r",
            dtype=dtype,
            offset=base + begin,
            shape=tuple(tensor["shape"]),
        )
        if array.nbytes != end - begin:
            raise ValueError("checkpoint tensor length differs from shape")
        self._check_ready_shard(filename)
        self._ready_unread.discard(filename)
        return array, tensor["dtype"]

    def tensor(self, name):
        array, kind = self.raw_tensor(name)
        if kind == "BF16":
            return (array.astype(np.uint32) << 16).view(np.float32)
        return array
