import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

from oh_my_vllm.operators.runtime.native import cache_directory, file_sha256

from .device import device_identity

DEEPSEEK_NATIVE_MODULES = (
    "device",
    "paged",
    "activation",
    "hyperconnection",
    "projection",
    "routing",
    "sparse_attention",
    "transform",
)


def _checkpoint_shard_manifest(root):
    index_path = Path(root) / "model.safetensors.index.json"
    index = index_path.read_bytes()
    names = set(json.loads(index)["weight_map"].values())
    if not names or any(
        re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) is None
        for name in names
    ):
        raise ValueError("checkpoint index must name local safetensors shards")
    return hashlib.sha256(index).hexdigest(), sorted(names)


def _checkpoint_shard_assignments(root, names, world_size):
    loads = [0] * world_size
    assignments = {}
    sizes = {name: (Path(root) / name).stat().st_size for name in names}
    for name in sorted(names, key=lambda value: (-sizes[value], value)):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[name] = rank
        loads[rank] += sizes[name]
    return assignments


def checkpoint_shard_identity(root, *, rank, world_size):
    if type(rank) is not int or type(world_size) is not int:
        raise ValueError("checkpoint identity rank and world size must be integers")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("checkpoint identity rank is outside its world")
    root = Path(root)
    _, names = _checkpoint_shard_manifest(root)
    assignments = _checkpoint_shard_assignments(root, names, world_size)
    return {
        name: file_sha256(root / name) for name in names if assignments[name] == rank
    }


def checkpoint_identity(model):
    model = Path(model)
    index_digest, names = _checkpoint_shard_manifest(model)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(
            pool.map(
                lambda rank: checkpoint_shard_identity(
                    model,
                    rank=rank,
                    world_size=4,
                ),
                range(4),
            )
        )
    result = {
        "config": file_sha256(model / "config.json"),
        "index": index_digest,
    }
    for part in parts:
        if set(result) & set(part):
            raise ValueError("checkpoint identity partitions overlap")
        result.update(part)
    if set(result) != {"config", "index", *names}:
        raise ValueError("checkpoint identity partitions are incomplete")
    return result


def native_identity():
    directory = cache_directory()
    result = {}
    for name in DEEPSEEK_NATIVE_MODULES:
        metadata = json.loads((directory / f"{name}.json").read_text())
        if file_sha256(metadata["library"]) != metadata["binary_id"]:
            raise RuntimeError(
                f"native binary identity differs from its manifest: {name}"
            )
        result[name] = metadata
    return result


def topology_identity(uuids):
    return [device_identity(uuid) for uuid in uuids]


def numeric_runtime_identity():
    import torch

    return {
        "python": list(sys.version_info[:3]),
        "torch": str(torch.__version__),
        "torch_source": torch.version.git_version,
        "cuda": torch.version.cuda,
        "nccl": (
            list(torch.cuda.nccl.version())
            if torch.distributed.is_nccl_available()
            else None
        ),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cpu_threads": torch.get_num_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NCCL_ALGO",
                "NCCL_PROTO",
                "NCCL_COLLNET_ENABLE",
                "NCCL_NVLS_ENABLE",
                "NCCL_MIN_NCHANNELS",
                "NCCL_MAX_NCHANNELS",
                "NCCL_CUMEM_ENABLE",
                "NCCL_CUMEM_HOST_ENABLE",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            )
        },
    }


def checkpoint_descriptor_identity(model):
    model = Path(model)
    return {
        "config": file_sha256(model / "config.json"),
        "index": file_sha256(model / "model.safetensors.index.json"),
    }
