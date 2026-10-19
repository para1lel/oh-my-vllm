import functools
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from oh_my_vllm.kernels.native import (
    build_digest,
    cache_directory,
    source_digest,
    source_files,
)

from .inputs import Prepared


@functools.cache
def build_core(fingerprint):
    import tvm_ffi
    from tvm_ffi.cpp import build_inline

    if os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a") != "10.0a":
        raise ValueError("GDN recurrent core requires architecture 10.0a")
    source = Path(__file__).with_suffix(".cu")
    cuda = Path(os.environ["CUDA_HOME"])
    compiler = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"]).decode()
    flags = (
        "-O3",
        "--generate-code=arch=compute_100a,code=sm_100a",
        "-lineinfo",
        "-static-global-template-stub=false",
        "--ptxas-options=-v",
    )
    functions = ("gdn_recurrent_core",)
    identity = build_digest(
        fingerprint,
        compiler,
        flags,
        importlib.metadata.version("apache-tvm-ffi"),
        None,
        functions,
    )
    directory = cache_directory() / "baselines" / "gdn_recurrent" / identity
    directory.mkdir(parents=True, exist_ok=True)
    library = build_inline(
        f"oh_my_vllm_gdn_recurrent_{identity[:16]}",
        cuda_sources=f'#include "{source}"',
        functions=functions,
        extra_cuda_cflags=flags,
        build_directory=str(directory),
        backend="cuda",
    )
    metadata = {
        "source_id": identity,
        "compiler": compiler,
        "cuda_flags": flags,
        "source_files": [str(path) for path in (*source_files(source), Path(__file__))],
        "binaries": [str(Path(library).resolve())],
    }
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return tvm_ffi.load_module(library), metadata


def prepare_with_tile(case, inputs, value_tile):
    import torch
    import tvm_ffi

    if case.operator != "gdn_attention" or not case.verify:
        raise ValueError("GDN recurrent core supplies only recurrent GDN")
    if max(case.queries) > 5:
        raise ValueError("GDN recurrent core supports at most five rows per sequence")
    offsets = inputs.tensors["sequence_offsets"].cpu().tolist()
    expected_offsets = [0]
    for count in case.queries:
        expected_offsets.append(expected_offsets[-1] + count)
    reads = inputs.read_slots.cpu().tolist()
    if (
        offsets != expected_offsets
        or offsets != list(inputs.sequence_offsets)
        or len(reads) != case.batch
        or any(slot < 0 or slot >= inputs.state_pool.shape[0] for slot in reads)
    ):
        raise ValueError("fixed recurrent sequence metadata is invalid")
    if (
        inputs.checkpoint_rows != tuple(range(case.rows))
        or len(inputs.checkpoint_slots) != case.rows
        or len(set(inputs.checkpoint_slots)) != case.rows
        or any(
            slot < 0 or slot >= inputs.state_pool.shape[0]
            for slot in inputs.checkpoint_slots
        )
        or set(inputs.checkpoint_slots).intersection(reads)
    ):
        raise ValueError("recurrent exports must use a distinct private slot per row")
    source = Path(__file__).with_suffix(".cu")
    module, build = build_core(source_digest(source))
    pool = inputs.state_pool.clone()
    output = torch.empty_like(inputs.v)
    writes = torch.tensor(
        inputs.checkpoint_slots, device=inputs.q.device, dtype=torch.int32
    )
    tensors = (
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        inputs.A_log,
        inputs.dt_bias,
        inputs.tensors["sequence_offsets"],
        inputs.read_slots,
        writes,
        pool,
        output,
    )
    views = tuple(tvm_ffi.from_dlpack(tensor) for tensor in tensors)

    def run():
        module.gdn_recurrent_core(*views, max(case.queries), value_tile)

    return Prepared(
        run,
        lambda: pool.copy_(inputs.state_pool),
        {"output": output, "state_pool": pool},
        {
            **build,
            "backend": "repository CUDA C++ fused recurrent core",
            "adaptation": "none; separate BF16 Q/K/V and gate input tensors",
            "launches": 1,
            "value_tile": value_tile,
            "threads": 128,
            "precision": "FP32 normalized Q/K and live carry; BF16 output and exports",
            "history": "complete; nonzero incoming state; no history truncation",
            "metadata_validation": (
                "fixed sequence offsets, read slots, and private destinations"
                " checked on the host during preparation; input hashes checked"
                " before and after timing; tensor contracts checked before launch"
            ),
            "gate_math": (
                "FP32 exp intrinsic within [-80,80], full-range expf otherwise;"
                " sixth-order log1p for inputs below 1/16 preserves tiny"
                " positive gates; subnormal-safe sigmoid; no decay cutoff"
            ),
            "checkpoint_exports": "each private row; FP32 carry is never reloaded",
            "read_slots_unchanged": True,
        },
        (module, tensors, views),
    )


def prepare(case, inputs):
    value_tile = 32 if case.batch >= 4 else 16
    return prepare_with_tile(case, inputs, value_tile)


def prepare_tile16(case, inputs):
    return prepare_with_tile(case, inputs, 16)


def prepare_tile32(case, inputs):
    return prepare_with_tile(case, inputs, 32)
