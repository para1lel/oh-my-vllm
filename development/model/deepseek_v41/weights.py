import copy
from pathlib import Path

import numpy as np

from .identity import (
    CONTROL_FILES,
    VERSION,
    control_file,
    file_state,
    inspect_shard,
    layout_identity,
    local_file,
    open_file,
    read_header,
    shard_layouts,
    unchanged,
)

RAW_DTYPES = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": "u1", "F8_E8M0": "u1", "I8": "i1"}


def select_rows(rows, count):
    if not isinstance(rows, (list, tuple, np.ndarray)):
        raise ValueError("rows must be a one-dimensional sequence of integer indices")
    if isinstance(rows, np.ndarray) and rows.ndim != 1:
        raise ValueError("rows must be one-dimensional")
    if any(
        not isinstance(row, (int, np.integer))
        or isinstance(row, (bool, np.bool_))
        or not 0 <= row < count
        for row in rows
    ):
        raise ValueError("rows must contain valid integer indices")
    return np.asarray(rows, dtype=np.int64)


class Weights:
    def __init__(self, root, manifest, expected_layout):
        if manifest.get("version") != VERSION or manifest.get("verification") != "full":
            raise ValueError("reference weights require a full checkpoint identity")
        if manifest.get("layout_sha256") != layout_identity(expected_layout):
            raise ValueError("reference tensor layout differs from checkpoint identity")
        self.root = Path(root).resolve(strict=True)
        self.manifest = copy.deepcopy(manifest)
        self.layout = copy.deepcopy(expected_layout)
        self.states = {}
        self.headers = {}
        index = self._control("model.safetensors.index.json", decode=True)
        self.groups = shard_layouts(index, self.layout)
        self.index = index["weight_map"]
        for filename in CONTROL_FILES:
            if filename != "model.safetensors.index.json":
                self._control(filename)
        for filename, record in self.manifest["files"].items():
            if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
                raise ValueError("reference weights require full file digests")
            if filename not in self.groups and filename not in CONTROL_FILES:
                self._control(filename)
        if not set(self.groups).issubset(self.manifest["files"]):
            raise ValueError("checkpoint identity omits a weight shard")

    def _control(self, filename, *, decode=False):
        parsed, record, initial = control_file(self.root, filename, decode=decode)
        if record != self.manifest["files"].get(filename):
            raise ValueError(f"reference control file identity differs: {filename}")
        self.states[filename] = initial
        return parsed

    def _verify_states(self):
        for filename, initial in self.states.items():
            path = local_file(self.root, filename)
            if file_state(path.stat(follow_symlinks=False)) != initial:
                raise ValueError(
                    f"reference checkpoint changed after verification: {filename}"
                )

    def _prepare(self, filename):
        if filename in self.headers:
            return self.headers[filename]
        _, record, initial, _ = inspect_shard(
            self.root, filename, self.groups[filename], True
        )
        if record != self.manifest["files"][filename]:
            raise ValueError(
                f"reference weight digest differs from checkpoint identity: {filename}"
            )
        self.states[filename] = initial
        path = local_file(self.root, filename)
        with open_file(path) as source:
            header, base = read_header(source)
            unchanged(path, source, initial)
        self.headers[filename] = header, base
        return header, base

    def _raw(self, name, rows=None):
        filename = self.index[name]
        header, base = self._prepare(filename)
        entry = header[name]
        path = local_file(self.root, filename)
        with open_file(path) as source:
            unchanged(path, source, self.states[filename])
            mapped = np.memmap(
                source,
                mode="r",
                dtype=RAW_DTYPES[entry["dtype"]],
                offset=base + entry["data_offsets"][0],
                shape=tuple(entry["shape"]),
            )
            try:
                result = np.array(mapped if rows is None else mapped[rows], copy=True)
            finally:
                mapped._mmap.close()
            unchanged(path, source, self.states[filename])
        return result

    def tensor(self, name, *, rows=None):
        from .quantization import decode_tensor

        if name not in self.layout:
            raise ValueError(f"unknown reference tensor: {name}")
        description = self.layout[name]
        if ".engram.embed." in name and rows is None:
            raise ValueError("Engram reference lookup requires explicit rows")
        if rows is not None:
            rows = select_rows(rows, description["shape"][0])
        self._verify_states()
        raw = self._raw(name, rows)
        dtype = description["dtype"]
        if dtype not in {"F8_E4M3", "I8"}:
            result = decode_tensor(raw, dtype)
        else:
            scale_name = name.removesuffix(".weight") + ".scale"
            if scale_name not in self.layout:
                raise ValueError(f"reference weight has no scale tensor: {name}")
            row_scales = dtype == "I8" or ".engram.embed." in name
            scale_rows = rows if row_scales or rows is None else rows // 32
            scales = self._raw(scale_name, scale_rows)
            scale_layout = "row32" if row_scales or rows is not None else "dense32"
            result = decode_tensor(raw, dtype, scales=scales, scale_layout=scale_layout)
        self._verify_states()
        return result
