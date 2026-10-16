import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from ..kernels.specification import MODEL


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        config = (self.path / "config.json").read_bytes()
        if hashlib.sha256(config).hexdigest() != MODEL["config_sha256"]:
            raise ValueError("checkpoint configuration differs from the model contract")
        self.identity = {"config": MODEL["config_sha256"]}
        self.identity["index"] = hashlib.sha256(
            (self.path / "model.safetensors.index.json").read_bytes()
        ).hexdigest()
        self.index = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers = {}

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
        if filename not in self.identity:
            with (self.path / filename).open("rb") as file:
                self.identity[filename] = hashlib.file_digest(
                    file, "sha256"
                ).hexdigest()
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
        return array, tensor["dtype"]

    def tensor(self, name):
        array, kind = self.raw_tensor(name)
        if kind == "BF16":
            return (array.astype(np.uint32) << 16).view(np.float32)
        return array
