import functools
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from oh_my_vllm.kernels.gdn import chunk_workspace
from oh_my_vllm.kernels.native import (
    build_digest,
    cache_directory,
    source_digest,
    source_files,
)

from .inputs import Prepared


@functools.cache
def build_core():
    import tvm_ffi
    from tvm_ffi.cpp import build_inline

    source = Path(__file__).with_suffix(".cu")
    cuda = Path(os.environ["CUDA_HOME"])
    compiler = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"]).decode()
    flags = (
        "-O3",
        "--generate-code=arch=compute_100a,code=sm_100a",
        "-lineinfo",
        "-static-global-template-stub=false",
    )
    functions = ("gdn_core",)
    cutlass = Path(os.environ["OH_MY_VLLM_CUTLASS"])
    commit = (
        subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    if commit != "098de2a652cf8f00fd70b2df54051c7eccbb855a":
        raise ValueError("CUTLASS must use the pinned v4.8.0 commit")
    if subprocess.check_output(["git", "-C", str(cutlass), "status", "--porcelain"]):
        raise ValueError("CUTLASS checkout must be clean")
    identity = build_digest(
        source_digest(source),
        compiler,
        flags,
        importlib.metadata.version("apache-tvm-ffi"),
        commit,
        functions,
    )
    directory = cache_directory() / "baselines" / "gdn" / identity
    directory.mkdir(parents=True, exist_ok=True)
    library = build_inline(
        "oh_my_vllm_gdn_core",
        cuda_sources=f'#include "{source}"',
        functions=functions,
        extra_cuda_cflags=flags,
        extra_include_paths=[str(cutlass / "include")],
        build_directory=str(directory),
        backend="cuda",
    )
    metadata = {
        "source_id": identity,
        "compiler": compiler,
        "cuda_flags": flags,
        "cutlass": commit,
        "source_files": [str(path) for path in (*source_files(source), Path(__file__))],
        "binaries": [str(Path(library).resolve())],
    }
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return tvm_ffi.load_module(library), metadata


def prepare(case, inputs):
    import torch
    import tvm_ffi

    if case.operator != "gdn_attention":
        raise ValueError("GDN native core cannot supply FA")
    if case.verify:
        from .gdn_recurrent import prepare as prepare_recurrent

        prepared = prepare_recurrent(case, inputs)
        prepared.metadata["source_files"].append(str(Path(__file__)))
        return prepared
    module, build = build_core()
    pool = inputs.state_pool.clone()
    output = torch.empty(
        (case.rows, 48, 128), device=inputs.q.device, dtype=torch.bfloat16
    )
    starts = inputs.tensors["sequence_offsets"]
    writes = torch.full((case.rows,), -1, device=inputs.q.device, dtype=torch.int32)
    if (
        len(inputs.checkpoint_rows) != len(inputs.checkpoint_slots)
        or len(set(inputs.checkpoint_slots)) != len(inputs.checkpoint_slots)
        or set(inputs.checkpoint_slots).intersection(range(case.batch))
        or any(
            slot < case.batch or slot >= pool.shape[0]
            for slot in inputs.checkpoint_slots
        )
    ):
        raise ValueError("GDN core checkpoints must use distinct private slots")
    writes[list(inputs.checkpoint_rows)] = torch.tensor(
        inputs.checkpoint_slots, device=inputs.q.device, dtype=torch.int32
    )
    workspace = chunk_workspace(case.rows, case.batch, max(case.queries), case.verify)
    scratch = tuple(
        torch.empty(shape, device=inputs.q.device, dtype=dtype)
        for shape, dtype in (
            ((case.rows, 10240), torch.bfloat16),
            ((case.rows, 96), torch.bfloat16),
            ((0 if case.verify else case.rows, 96), torch.float32),
        )
    )
    chunks = tuple(
        torch.empty(
            shape,
            device=inputs.q.device,
            dtype={
                "int32": torch.int32,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[dtype],
        )
        for shape, dtype in workspace
    )
    tensors = (
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        inputs.A_log,
        inputs.dt_bias,
        pool,
        starts,
        inputs.read_slots,
        writes,
        *scratch,
        output,
        *chunks,
    )
    views = tuple(tvm_ffi.from_dlpack(value) for value in tensors)

    def run():
        module.gdn_core(*views, case.verify, max(case.queries))

    return Prepared(
        run,
        lambda: pool.copy_(inputs.state_pool),
        {"output": output, "state_pool": pool},
        {
            **build,
            "backend": "repository CUDA C++",
            "adaptation": (
                "timed Q/K/V packing, prefill Q/K normalization, and FP32 gates"
            ),
            "precision": (
                "FP32 live carry and recurrent Q/K; BF16 chunk matrix operands, "
                "prefill Q/K and checkpoints; FP32 accumulation"
            ),
            "history": "complete; nonzero incoming state; no gate-decay truncation",
            "checkpoint_exports": (
                "all required private rows; export does not reload live carry"
            ),
            "read_slots_unchanged": True,
        },
        (module, tensors, views),
    )
