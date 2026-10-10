"""Fail-closed inventory of compiled and explicitly planned GPU operations."""

import hashlib
import importlib.metadata
import math
from pathlib import Path


def runtime_contract():
    """Bind eager paths and external providers to the reviewed equation set.

    FX inventory covers compiled units. Source identities additionally cover
    eager sampling, rollback, metadata, and CUDA Graph replay paths. A source
    change requires semantic review before updating the checked-in contract.
    """
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in ("worker", "models", "ir", "kernels"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix in (".py", ".cu", ".cpp", ".cuh", ".h"):
                sources[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    providers = {}
    distribution = importlib.metadata.distribution("flashinfer-python")
    for name in (
        "flashinfer/gemm/gemm_base.py",
        "flashinfer/prefill.py",
        "flashinfer/decode.py",
        "flashinfer/jit/attention/modules.py",
        "flashinfer/jit/gemm/core.py",
        "flashinfer/gdn_kernels/blackwell/gdn_prefill.py",
        "flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py",
        "flashinfer/gdn_prefill.py",
        "flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py",
    ):
        path = distribution.locate_file(name)
        if not path.is_file():
            raise ValueError(f"missing reviewed GPU provider: {name}")
        providers[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    from oh_my_vllm.kernels.cuda_backend import _project_headers as operator_headers
    from oh_my_vllm.kernels.cuda_backend.groupwise import (
        _FLAGS,
        _headers,
    )
    from oh_my_vllm.kernels.cuda_backend.groupwise import (
        _project_headers as gemm_headers,
    )

    data = distribution.locate_file("flashinfer/data")
    roots = (
        data / "cccl/cub",
        data / "cccl/libcudacxx/include",
        data / "cccl/thrust",
        data / "cutlass/include",
        data / "cutlass/tools/util/include",
    )
    import json

    gemm_inputs = {
        "headers_sha256": hashlib.sha256(
            json.dumps(_headers(data, roots), sort_keys=True).encode()
        ).hexdigest(),
        "flags": list(_FLAGS),
        "project_headers": gemm_headers(),
    }
    return {
        "sources": sources,
        "providers": providers,
        "gemm_inputs": gemm_inputs,
        "kernel_inputs": {"project_headers": operator_headers()},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "flashinfer-python", "triton")
        },
    }


IR_OPERATIONS = {
    "add_norm_fp8_linear",
    "gated_norm_fp8_linear",
    "add_rms_norm",
    "causal_conv",
    "delta_gates",
    "fp8_linear",
    "fp8_quantize",
    "gdn_prefill",
    "gdn_prepare",
    "gdn_recurrent",
    "logits_gemm",
    "planned_attention",
    "prepare_attention",
    "prepare_context",
    "prepare_query",
    "rms_norm",
    "silu_mul",
    "dspark_rms_norm",
    "dspark_norm_rope",
    "dspark_append",
    "dspark_attention",
}
FUNCTIONS = {
    "_operator.add",
    "_operator.getitem",
    "_operator.mul",
    "_operator.sub",
    "torch._C._nn.linear",
    "torch.nn.functional.embedding",
    "torch.arange",
    "torch.cat",
    "torch.stack",
    "torch.isfinite",
    "torch.nn.functional.silu",
}
METHODS = {
    "flatten",
    "float",
    "index_copy_",
    "index_select",
    "reshape",
    "sigmoid",
    "split",
    "to",
    "view",
    "squeeze",
    "argmax",
    "max",
    "new_zeros",
}
STEP_OPERATIONS = {
    "target",
    "commit",
    "mtp_forward",
    "mtp_logits",
    "mtp_proposal_graph",
    "dspark_inject",
    "dspark_backbone",
    "execution_route",
}


def metadata_covers(unit, target, descriptions):
    """Recognize every typed occurrence, without a global arithmetic allowlist.

    Proposal positions and tables are host-known in the reviewed source unit.
    Their address calculation can move to the host in the ideal implementation.
    Tensor size queries return shape values, with no GPU computation/readback.
    """
    rows = [
        row
        for row in descriptions
        if (row.get("unit"), row.get("target")) == (unit, target)
    ]
    if not rows:
        return False
    for row in rows:
        if row.get("receiver_tensor") is not True:
            return False
        if target == "call_method:size":
            if row.get("category") != "shape_query" or row.get("result_kind") not in (
                "dimension",
                "shape",
            ):
                return False
            continue
        consumers = row.get("consumers")
        expected = (
            "_operator.getitem" if target == "_operator.floordiv" else "_operator.add"
        )
        if (
            unit != "proposal_graph"
            or target not in ("_operator.floordiv", "_operator.mod")
            or row.get("category") != "host_known_page_address"
            or type(row.get("divisor")) is not int
            or row["divisor"] != 784
            or any(
                row.get(name) != "torch.int64"
                for name in ("receiver_dtype", "result_dtype")
            )
            or any(
                row.get(name) != "cuda" for name in ("receiver_device", "result_device")
            )
            or any(
                type(row.get(name)) is not int or row[name] != 1
                for name in ("receiver_rank", "result_rank")
            )
            or row.get("same_shape") is not True
            or not isinstance(consumers, list)
            or not consumers
            or any(consumer != expected for consumer in consumers)
        ):
            return False
    return True


def validate_weights(weights):
    """Derive bytes from geometry and validate loaded target and draft tensors."""
    sizes = {"torch.float8_e4m3fn": 1, "torch.bfloat16": 2, "torch.float32": 4}
    if not isinstance(weights, dict) or not weights:
        raise ValueError("missing loaded parameter geometry")
    for name, row in weights.items():
        shape, dtype = row.get("shape"), row.get("dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(d) is not int or d <= 0 for d in shape)
            or dtype not in sizes
            or type(row.get("bytes")) is not int
            or row["bytes"] != math.prod(shape) * sizes[dtype]
        ):
            raise ValueError(f"invalid loaded parameter byte geometry: {name}")

    expected = {}

    def tensor(name, shape, dtype):
        expected[name] = (shape, "torch." + dtype)

    def linear(name, n, k, *, fp8=True):
        tensor(name + ".weight", [n, k], "float8_e4m3fn" if fp8 else "bfloat16")
        if fp8:
            tensor(name + ".scale", [n // 128, k // 128], "float32")

    tensor("target.embedding", [248320, 5120], "bfloat16")
    tensor("target.head", [248320, 5120], "bfloat16")
    tensor("target.norm", [5120], "float32")
    for layer in range(64):
        prefix = f"target.layers.{layer}"
        for name in ("input_norm", "post_norm"):
            tensor(prefix + "." + name, [5120], "float32")
        linear(prefix + ".gate_up", 34816, 5120)
        linear(prefix + ".down", 5120, 17408)
        linear(prefix + ".out", 5120, 6144)
        if layer % 4 == 3:
            linear(prefix + ".qkv", 14336, 5120)
            for name in ("q_norm", "k_norm"):
                tensor(prefix + "." + name, [256], "float32")
        else:
            linear(prefix + ".qkvz", 16384, 5120)
            linear(prefix + ".ba", 96, 5120, fp8=False)
            tensor(prefix + ".conv", [10240, 4], "bfloat16")
            for name, width in (("a_log", 48), ("dt_bias", 48), ("gate_norm", 128)):
                tensor(prefix + "." + name, [width], "float32")
    if any(name.startswith("target.mtp") for name in weights):
        for name in ("mtp_embedding_norm", "mtp_hidden_norm", "mtp_norm"):
            tensor("target." + name, [5120], "float32")
        linear("target.mtp_fc", 5120, 10240, fp8=False)
        prefix = "target.mtp"
        for name in ("input_norm", "post_norm"):
            tensor(prefix + "." + name, [5120], "float32")
        linear(prefix + ".gate_up", 34816, 5120)
        linear(prefix + ".down", 5120, 17408)
        linear(prefix + ".out", 5120, 6144)
        linear(prefix + ".qkv", 14336, 5120)
        for name in ("q_norm", "k_norm"):
            tensor(prefix + "." + name, [256], "float32")
    if any(name.startswith("dspark.") for name in weights):
        for name in ("hidden_norm", "norm"):
            tensor("dspark." + name, [5120], "bfloat16")
        tensor("dspark.fc", [5120, 25600], "bfloat16")
        for name in ("markov_w1", "markov_w2"):
            tensor("dspark." + name, [248320, 256], "bfloat16")
        tensor("dspark.confidence_weight", [1, 5376], "bfloat16")
        tensor("dspark.confidence_bias", [1], "bfloat16")
        tensor("dspark.inv_freq", [64], "float32")
        for layer in range(5):
            prefix = f"dspark.layers.{layer}"
            for name in ("input_norm", "post_norm"):
                tensor(prefix + "." + name, [5120], "bfloat16")
            for name in ("q_norm", "k_norm"):
                tensor(prefix + "." + name, [128], "float32")
            for name, n, k in (
                ("q", 4096, 5120),
                ("k", 1024, 5120),
                ("v", 1024, 5120),
                ("o", 5120, 4096),
                ("gate", 17408, 5120),
                ("up", 17408, 5120),
                ("down", 5120, 17408),
            ):
                tensor(prefix + "." + name, [n, k], "bfloat16")
    for name, (shape, dtype) in expected.items():
        row = weights.get(name, {})
        if row.get("shape") != shape or row.get("dtype") != dtype:
            raise ValueError(f"loaded parameter differs from model equations: {name}")


def validate_inventory(trace):
    if trace.get("incomplete"):
        raise ValueError("incomplete semantic trace")
    validate_weights(trace.get("weights"))
    contract = Path(__file__).with_name("contract.json")
    import json

    reviewed = json.loads(contract.read_text())
    if trace.get("runtime_contract") != reviewed:
        raise ValueError("GPU source/provider contract requires semantic review")
    loaded_cuda = trace.get("cuda_provenance") or {}
    if loaded_cuda.get("source_sha256") != reviewed["sources"].get(
        "kernels/cuda_backend/kernels.cu"
    ):
        raise ValueError("loaded CUDA source differs from reviewed source")
    if any(
        loaded_cuda.get(key) != value
        for key, value in reviewed["kernel_inputs"].items()
    ):
        raise ValueError("loaded CUDA headers differ from reviewed source")
    if any(
        not isinstance(loaded_cuda.get(key), str)
        or len(loaded_cuda[key]) != 64
        or set(loaded_cuda[key]) - set("0123456789abcdef")
        for key in ("so_sha256", "build_input_sha256")
    ):
        raise ValueError("loaded CUDA binary identity is incomplete")
    loaded_gemm = trace.get("cuda_gemm_provenance") or {}
    if loaded_gemm.get("source_sha256") != reviewed["sources"].get(
        "kernels/cuda_backend/groupwise_fp8.cu"
    ):
        raise ValueError("loaded GEMM source differs from reviewed source")
    if any(
        not isinstance(loaded_gemm.get(key), str)
        or len(loaded_gemm[key]) != 64
        or set(loaded_gemm[key]) - set("0123456789abcdef")
        for key in ("so_sha256", "build_input_sha256")
    ):
        raise ValueError("loaded GEMM binary identity is incomplete")
    if any(
        loaded_gemm.get(key) != value for key, value in reviewed["gemm_inputs"].items()
    ):
        raise ValueError("owned GEMM templates or flags require semantic review")
    if not trace.get("graph_operations"):
        raise ValueError("missing compiled GPU operation inventory")
    unknown = []
    for unit, target in trace["graph_operations"]:
        target = target.removeprefix("torch._ops.")
        if target in FUNCTIONS:
            continue
        if target in (
            "_operator.floordiv",
            "_operator.mod",
            "call_method:size",
        ) and metadata_covers(unit, target, trace.get("graph_metadata", [])):
            continue
        if (
            target.startswith("call_method:")
            and target.removeprefix("call_method:") in METHODS
        ):
            continue
        if target.startswith("oh_my_vllm_ir.") and target.endswith(".default"):
            operation = target.removeprefix("oh_my_vllm_ir.").removesuffix(".default")
            if operation in IR_OPERATIONS:
                continue
        unknown.append((unit, target))
    for step in trace["steps"]:
        for operation in step["operations"]:
            if operation["kind"] not in STEP_OPERATIONS:
                unknown.append(("step", operation["kind"]))
            if operation["kind"] == "target" and any(
                r.get("plain_greedy") is not True for r in operation["requests"]
            ):
                unknown.append(("sampling", "non-plain-greedy"))
            if operation["kind"] == "dspark_backbone" and not operation["greedy"]:
                unknown.append(("sampling", "non-greedy-dspark"))
    if unknown:
        raise ValueError(f"unrecognized GPU operations: {unknown}")
