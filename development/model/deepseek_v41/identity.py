import argparse
import hashlib
import json
import math
import os
import stat
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "deepseek-v41-checkpoint-v1"
MAX_HEADER_BYTES = 64 << 20
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
CONTROL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SOURCE_FILES = (
    "chat_template.jinja",
    "encoding/encoding.py",
    "inference/model.py",
    "inference/kernel.py",
    "inference/convert.py",
    "inference/engram.py",
    "inference/vision.py",
    "inference/image_processor.py",
)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    try:
        result = json.loads(data, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid checkpoint JSON") from error
    if not isinstance(result, dict):
        raise ValueError("checkpoint JSON must contain an object")
    return result


def file_state(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def local_file(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(
            f"checkpoint file is a link or leaves the model directory: {name}"
        )
    if not path.is_file():
        raise ValueError(f"checkpoint file is missing or is not a regular file: {name}")
    return path


def open_file(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"checkpoint file is not regular: {path.name}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def unchanged(path, source, initial):
    if file_state(os.fstat(source.fileno())) != initial:
        raise ValueError(f"checkpoint file changed during reading: {path.name}")
    if file_state(path.stat(follow_symlinks=False)) != initial:
        raise ValueError(f"checkpoint file changed during reading: {path.name}")


def control_file(root, name, *, decode=False):
    path = local_file(root, name)
    with open_file(path) as source:
        initial = file_state(os.fstat(source.fileno()))
        if decode:
            if initial[2] > MAX_HEADER_BYTES:
                raise ValueError(f"checkpoint JSON is too large: {name}")
            data = source.read()
            parsed = parse_json(data)
            digest = hashlib.sha256(data).hexdigest()
        else:
            parsed = None
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        unchanged(path, source, initial)
    return parsed, {"size": initial[2], "sha256": digest}, initial


def tensor_record(name, entry):
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise ValueError(f"invalid tensor header: {name}")
    dtype, shape, offsets = (
        entry[field] for field in ("dtype", "shape", "data_offsets")
    )
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported tensor dtype: {name}")
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError(f"invalid tensor shape: {name}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(offset) is not int or offset < 0 for offset in offsets)
    ):
        raise ValueError(f"invalid tensor offsets: {name}")
    begin, end = offsets
    size = math.prod(shape) * DTYPE_BYTES[dtype]
    if end - begin != size:
        raise ValueError(f"tensor byte length differs from its shape: {name}")
    return {"dtype": dtype, "shape": shape}, begin, end


def read_header(source):
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated checkpoint header length")
    length = struct.unpack("<Q", prefix)[0]
    if length == 0 or length > MAX_HEADER_BYTES or length % 8:
        raise ValueError("invalid checkpoint header length")
    encoded = source.read(length)
    if len(encoded) != length:
        raise ValueError("truncated checkpoint header")
    header = parse_json(encoded)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(value, str) for value in metadata.values()
    ):
        raise ValueError("invalid safetensors metadata")
    return header, length + 8


def inspect_shard(root, name, expected, hash_weights):
    path = local_file(root, name)
    with open_file(path) as source:
        initial = file_state(os.fstat(source.fileno()))
        header, base = read_header(source)
        if set(header) != set(expected):
            raise ValueError(f"shard tensors differ from the index: {name}")
        spans = []
        for tensor, entry in header.items():
            description, begin, end = tensor_record(tensor, entry)
            if description != expected[tensor]:
                raise ValueError(
                    f"tensor layout differs from the model contract: {tensor}"
                )
            spans.append((begin, end, tensor))
        cursor = 0
        components = Counter()
        for begin, end, tensor in sorted(spans):
            if begin != cursor:
                raise ValueError(
                    f"tensor offsets overlap or leave a payload gap: {tensor}"
                )
            cursor = end
            components[component(tensor)] += end - begin
        if base + cursor != initial[2]:
            raise ValueError(
                f"shard payload length differs from its tensor offsets: {name}"
            )
        digest = None
        if hash_weights:
            source.seek(0)
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        unchanged(path, source, initial)
    return name, {"size": initial[2], "sha256": digest}, initial, components


def component(name):
    if ".engram." in name:
        return "engram"
    if name.startswith("mtp."):
        return "dspark"
    if name.startswith(("vision.", "visual.", "aligner.")):
        return "vision"
    return "target"


def shard_layouts(index, expected_layout):
    mapping = index.get("weight_map")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or set(mapping) != set(expected_layout)
    ):
        raise ValueError("checkpoint index tensors differ from the model contract")
    groups = defaultdict(dict)
    for tensor, filename in mapping.items():
        if (
            not isinstance(filename, str)
            or "/" in filename
            or "\\" in filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(
                "checkpoint shard filename must be a local safetensors basename"
            )
        groups[filename][tensor] = expected_layout[tensor]
    return groups


def layout_identity(expected_layout):
    encoded = json.dumps(
        expected_layout, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def scan_checkpoint(root, expected_layout, *, hash_weights=True, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("checkpoint hash worker count must be between 1 and 8")
    root = Path(root).resolve(strict=True)
    files, states, controls = {}, {}, {}
    for name in CONTROL_FILES:
        controls[name], files[name], states[name] = control_file(
            root, name, decode=name in CONTROL_FILES[:2]
        )
    index = controls["model.safetensors.index.json"]
    groups = shard_layouts(index, expected_layout)
    for name in groups:
        local_file(root, name)
    components = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(inspect_shard, root, name, expected, hash_weights)
            for name, expected in sorted(groups.items())
        ]
        for future in futures:
            name, files[name], states[name], counts = future.result()
            components.update(counts)
    total = sum(components.values())
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("invalid checkpoint index metadata")
    declared = metadata.get("total_size")
    if declared is not None and (type(declared) is not int or declared != total):
        raise ValueError("checkpoint index total_size differs from tensor bytes")
    for name in SOURCE_FILES:
        if (root / name).exists() or (root / name).is_symlink():
            _, files[name], states[name] = control_file(root, name)
    for name, initial in states.items():
        if file_state((root / name).stat(follow_symlinks=False)) != initial:
            raise ValueError(f"checkpoint file changed during verification: {name}")
    return {
        "version": VERSION,
        "verification": "full" if hash_weights else "metadata-only",
        "files": files,
        "layout_sha256": layout_identity(expected_layout),
        "tensor_count": len(expected_layout),
        "shard_count": len(groups),
        "tensor_bytes": total,
        "component_bytes": dict(sorted(components.items())),
    }


def inspect_checkpoint(root, *, hash_weights=True, workers=1):
    from .layout import tensor_layout

    root = Path(root).resolve(strict=True)
    config, identity, _ = control_file(root, "config.json", decode=True)
    result = scan_checkpoint(
        root, tensor_layout(config), hash_weights=hash_weights, workers=workers
    )
    if identity != result["files"]["config.json"]:
        raise ValueError("checkpoint configuration changed during layout construction")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate DeepSeek V4.1 checkpoint bytes and tensor layout"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if output.is_relative_to(repository) or output.is_relative_to(args.model.resolve()):
        parser.error(
            "write the manifest outside the repository and checkpoint directory"
        )
    if output.exists():
        parser.error("the output manifest already exists")
    result = inspect_checkpoint(
        args.model, hash_weights=not args.metadata_only, workers=args.workers
    )
    with output.open("x") as target:
        json.dump(result, target, sort_keys=True, indent=2)
        target.write("\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "files"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
