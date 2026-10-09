"""Auditable production call sites for owned and FlashInfer model operators.

The manifest covers the model's computation entry points. Kernel wrappers and
FlashInfer calls inside an implementation are deliberately below the IR boundary.
Ordinary PyTorch operations, host planning, and sampling are outside its scope.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Site:
    file: str
    function: str
    call: str
    operation: str


SITES = (
    Site("models/qwen.py", "Linear.__call__", "fp8.linear", "fp8_linear"),
    Site("models/qwen.py", "Linear.silu", "fp8.linear", "fp8_linear"),
    Site("models/qwen.py", "Linear.silu", "silu_mul", "silu_mul"),
    Site("models/qwen.py", "Batch.delta", "gdn_prefill", "gdn_prefill"),
    Site("models/qwen.py", "Batch.delta", "gdn_recurrent", "gdn_recurrent"),
    Site(
        "models/qwen.py",
        "Layer.full_attention",
        "prepare_attention",
        "prepare_attention",
    ),
    Site("models/qwen.py", "Layer.delta_attention", "gdn_prepare", "gdn_prepare"),
    Site("ir/gdn_prepare.py", "_kernel_prepare", "causal_conv", "causal_conv"),
    Site("ir/gdn_prepare.py", "_kernel_prepare", "delta_gates", "delta_gates"),
    Site("ir/gdn_prepare.py", "_kernel_prepare", "linear", "fp8_linear"),
    Site("models/qwen.py", "Layer.delta_attention", "rms_norm", "rms_norm"),
    Site(
        "models/qwen.py",
        "Layer.delta_attention",
        "gated_norm_fp8_linear",
        "gated_norm_fp8_linear",
    ),
    Site("models/qwen.py", "Layer.forward_residual", "rms_norm", "rms_norm"),
    Site("models/qwen.py", "Layer.forward_residual", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Layer.forward_selected", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Layer.forward_selected", "rms_norm", "rms_norm"),
    Site(
        "models/qwen.py", "Layer.forward_selected", "prepare_context", "prepare_context"
    ),
    Site("models/qwen.py", "Layer.forward_selected", "prepare_query", "prepare_query"),
    Site("models/qwen.py", "Layer.mlp_residual", "add_rms_norm", "add_rms_norm"),
    Site(
        "models/qwen.py",
        "Layer.mlp_residual",
        "add_norm_fp8_linear",
        "add_norm_fp8_linear",
    ),
    Site("models/qwen.py", "Qwen.forward", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Qwen._selected_final", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Qwen.forward_features", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Qwen.draft", "rms_norm", "rms_norm"),
    Site("models/qwen.py", "Qwen.draft", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Qwen.draft_context", "rms_norm", "rms_norm"),
    Site("models/qwen.py", "Qwen.draft_context", "add_rms_norm", "add_rms_norm"),
    Site("models/qwen.py", "Qwen.logits", "logits_gemm", "logits_gemm"),
    Site("models/dspark.py", "rms_norm", "dspark_rms_norm", "dspark_rms_norm"),
    Site("models/dspark.py", "DSparkModel.inject", "rms_norm", "dspark_rms_norm"),
    Site("models/dspark.py", "DSparkModel.inject", "prepare_qk", "dspark_norm_rope"),
    Site("models/dspark.py", "DSparkModel.inject", "append", "dspark_append"),
    Site("models/dspark.py", "DSparkModel.forward", "rms_norm", "dspark_rms_norm"),
    Site("models/dspark.py", "DSparkModel.forward", "prepare_qk", "dspark_norm_rope"),
    Site("models/dspark.py", "DSparkModel.forward", "attention", "dspark_attention"),
    Site(
        "kernels/attention.py",
        "PagedAttention.__call__",
        "planned_attention",
        "planned_attention",
    ),
    Site(
        "kernels/mtp_attention.py",
        "MTPAttention.__call__",
        "planned_attention",
        "planned_attention",
    ),
    Site(
        "worker/decode_graph.py",
        "DecodeAttention.__call__",
        "planned_attention",
        "planned_attention",
    ),
)

# These imports are planning/type references or provider bodies. A new direct
# backend import in model execution needs an explicit review and reason here.
DIRECT_IMPORT_EXCEPTIONS = {
    (
        "models/dspark.py",
        "oh_my_vllm.kernels.dspark_attention",
        name,
    ): "Public DSpark wrapper delegates to a registered semantic IR operation"
    for name in ("rms_norm", "prepare_qk", "append", "attention")
} | {
    (
        "models/qwen.py",
        "oh_my_vllm.kernels",
        "attention",
    ): "AttentionBatch type and planned attention object",
    (
        "models/qwen.py",
        "oh_my_vllm.kernels.mtp_attention",
        "MTPAttention",
    ): "AttentionBatch type only",
    (
        "worker/model_runner.py",
        "oh_my_vllm.kernels.attention",
        "PagedAttention",
    ): "host batch planning outside the compiled unit",
    (
        "worker/mtp.py",
        "oh_my_vllm.kernels.mtp_attention",
        "MTPAttention",
    ): "host MTP planning outside the compiled unit",
    (
        "worker/prefill_graph.py",
        "oh_my_vllm.kernels.attention",
        "PagedAttention",
    ): "host prefill planning before graph capture",
    (
        "worker/decode_graph.py",
        "oh_my_vllm.kernels.decode_attention",
        "decode",
    ): "owned attention provider body",
    (
        "worker/decode_graph.py",
        "flashinfer.decode",
        "trtllm_batch_decode_with_kv_cache",
    ): "FlashInfer attention provider body",
}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _calls(path: Path) -> Counter[tuple[str, str]]:
    found: Counter[tuple[str, str]] = Counter()

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                for nested in ast.walk(node):
                    if isinstance(nested, ast.Call):
                        found[(name, _call_name(nested.func))] += 1

    visit(ast.parse(path.read_text()).body)
    return found


def check_coverage(package_root: Path, registered: set[str]) -> None:
    """Fail when a required site disappears or its semantic op is unregistered."""
    observed = {file: _calls(package_root / file) for file in {s.file for s in SITES}}
    missing = [
        site
        for site in SITES
        if observed[site.file][(site.function, site.call)] == 0
        or site.operation not in registered
    ]
    if missing:
        raise AssertionError(f"uncovered model operator sites: {missing}")
    high_level_files = {
        "models/qwen.py",
        "models/dspark.py",
        "worker/model_runner.py",
        "worker/mtp.py",
        "worker/decode_graph.py",
        "worker/dspark.py",
        "worker/dspark_graph.py",
        "worker/mtp_context_graph.py",
        "worker/prefill_graph.py",
    }
    used_exceptions = set()
    for file in high_level_files:
        module = ast.parse((package_root / file).read_text())
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith(("oh_my_vllm.kernels", "flashinfer")):
                    for name in node.names:
                        key = (file, node.module, name.name)
                        if key not in DIRECT_IMPORT_EXCEPTIONS:
                            raise AssertionError(
                                f"unlisted direct backend import: {key}"
                            )
                        used_exceptions.add(key)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith(("oh_my_vllm.kernels", "flashinfer")):
                        raise AssertionError(
                            f"unlisted direct backend import: {(file, name.name)}"
                        )
            elif isinstance(node, ast.Call) and _call_name(node.func).startswith(
                "torch.ops."
            ):
                raise AssertionError(f"{file} bypasses IR through torch.ops")
            elif (
                file == "models/qwen.py"
                and isinstance(node, ast.Call)
                and _call_name(node.func).startswith("attention.")
            ):
                raise AssertionError(f"{file} calls a backend through attention")
    if used_exceptions != DIRECT_IMPORT_EXCEPTIONS.keys():
        raise AssertionError(
            f"stale direct-import exceptions: "
            f"{DIRECT_IMPORT_EXCEPTIONS.keys() - used_exceptions}"
        )
