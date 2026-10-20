import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cases import digest
from .storage import state_rows, storage_dtype

FIXTURE_ROOT = Path(__file__).with_name("fixtures")
MAX_SHARD_BYTES = 32 * 1024 * 1024
GROUPS = {"inputs", "state", "expected"}


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class Fixture:
    manifest: dict
    inputs: dict
    state: dict
    expected: dict

    def dtype(self, group, name):
        return self.manifest["tensors"][f"{group}.{name}"]["tensor_dtype"]


def shard_path(root, name):
    path = root / name
    if Path(name).name != name or path.is_symlink() or not path.is_file():
        raise ValueError("fixture shard is missing or is not a regular local file")
    if path.stat().st_size > MAX_SHARD_BYTES:
        raise ValueError("fixture shard exceeds 32 MiB")
    return path


def load_tensor(root, record, files):
    if set(record) != {"shape", "dtype", "tensor_dtype", "parts"}:
        raise ValueError("incomplete fixture tensor description")
    shape = record["shape"]
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("invalid fixture shape")
    dtype = np.dtype(record["dtype"])
    if dtype.kind not in "biuf" or dtype.hasobject:
        raise ValueError("unsupported fixture dtype")
    if np.dtype(storage_dtype(record["tensor_dtype"])) != dtype:
        raise ValueError("fixture storage differs from its logical tensor dtype")
    count = int(np.prod(shape, dtype=object))
    output = np.empty(count, dtype=dtype)
    offset = 0
    for part in record["parts"]:
        if set(part) != {"file", "key", "offset", "count"} or part["file"] not in files:
            raise ValueError("unbound fixture tensor part")
        if (
            part["offset"] != offset
            or type(part["count"]) is not int
            or part["count"] < 0
        ):
            raise ValueError("fixture tensor parts are incomplete or overlap")
        end = offset + part["count"]
        if end > count or part["count"] * dtype.itemsize > MAX_SHARD_BYTES:
            raise ValueError("fixture tensor part exceeds its bounds")
        with np.load(shard_path(root, part["file"]), allow_pickle=False) as data:
            if part["key"] not in data:
                raise ValueError("missing fixture array")
            value = data[part["key"]]
            if value.dtype != dtype or value.size != part["count"]:
                raise ValueError(
                    "fixture array differs from its declared dtype or shape"
                )
            output[offset:end] = value.reshape(-1)
        offset = end
    if offset != count:
        raise ValueError("incomplete fixture tensor")
    output = output.reshape(shape)
    output.setflags(write=False)
    return output


def load_fixture(case, root=None):
    root = Path(root) if root is not None else FIXTURE_ROOT / case.id
    path = root / "manifest.json"
    if root.is_symlink() or path.is_symlink() or not case.fixture_sha256:
        raise ValueError("fixture has no approved regular-file identity")
    if file_digest(path) != case.fixture_sha256:
        raise ValueError("fixture manifest differs from its approved identity")
    manifest = json.loads(path.read_text())
    if set(manifest) != {
        "version",
        "case",
        "checkpoint",
        "reference",
        "files",
        "tensors",
        "writes",
    }:
        raise ValueError("incomplete fixture manifest")
    if (
        manifest["version"] != "operator-fixture-v1"
        or digest(manifest["case"]) != case.identity
    ):
        raise ValueError("fixture differs from the fixed case")
    if (
        not manifest["checkpoint"]
        or manifest["reference"].get("precision") != "torch.float64"
    ):
        raise ValueError("fixture lacks checkpoint or PyTorch reference identity")
    for key in ("source", "torch"):
        if not manifest["reference"].get(key):
            raise ValueError("fixture lacks reference provenance")
    for name, expected in manifest["files"].items():
        if file_digest(shard_path(root, name)) != expected:
            raise ValueError("fixture shard differs from its approved identity")
    groups = {group: {} for group in GROUPS}
    for name, record in manifest["tensors"].items():
        group, separator, field = name.partition(".")
        if not separator or group not in GROUPS or not field:
            raise ValueError("invalid fixture tensor name")
        groups[group][field] = load_tensor(root, record, manifest["files"])
    if set(groups["expected"]) != set(case.outputs):
        raise ValueError("fixture outputs differ from the complete boundary contract")
    if not groups["inputs"] and not groups["state"]:
        raise ValueError("fixture has no stored inputs")
    validate_writes(manifest["writes"], groups)
    return Fixture(manifest, **groups)


def validate_writes(writes, groups):
    outputs = set()
    for name, write in writes.items():
        if name not in groups["state"] or set(write) != {"indices", "layout", "output"}:
            raise ValueError("fixture write does not identify a stored state")
        if write["output"] not in groups["expected"] or write["output"] in outputs:
            raise ValueError("fixture write has no unique expected output")
        outputs.add(write["output"])
        rows = state_rows(groups["state"][name], write["layout"])
        indices = write["indices"]
        if (
            not isinstance(indices, list)
            or any(
                type(index) is not int or not 0 <= index < len(rows)
                for index in indices
            )
            or len(set(indices)) != len(indices)
            or groups["expected"][write["output"]].shape
            != (len(indices), *rows.shape[1:])
        ):
            raise ValueError("fixture writes differ from assigned state rows")
