"""Buffer semantic execution metadata; write it after measured requests finish."""

import json
import os
from pathlib import Path

_CURRENT = None
_GRAPH_OPERATIONS = set()


def tracing():
    """Check before constructing optional shape dictionaries or token sets."""
    return _CURRENT is not None


def record(kind, **fields):
    """Record host-known effective sizes without GPU readback or per-step I/O."""
    if _CURRENT is not None:
        operation = {"kind": kind, **fields}
        _CURRENT["operations"].append(operation)
        return operation


def certify_feedback():
    """Mark work completed by an existing device-to-host token readback.

    This adds no synchronization. Work queued after the final readback can
    continue after Rust receives the final token and cannot raise its bound.
    """
    if _CURRENT is not None:
        _CURRENT["completed_operations"] = len(_CURRENT["operations"])


def _target_name(target):
    """Use stable names; repr of Python functions includes process addresses."""
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    return f"{module}.{name}" if module and name else str(target)


def observe_graph(unit, graph):
    """Keep the pre-provider GPU operation inventory for fail-closed validation."""
    for node in graph.nodes:
        if node.op == "call_function":
            _GRAPH_OPERATIONS.add((unit, _target_name(node.target)))
        elif node.op not in ("placeholder", "output", "get_attr"):
            _GRAPH_OPERATIONS.add((unit, f"{node.op}:{node.target}"))


class ExecutionTrace:
    """Opt-in, process-local trace with GPU event times used only for diagnosis."""

    def __init__(self, worker=None):
        self.path = os.environ.get("OH_MY_VLLM_SEMANTIC_TRACE")
        self.rows = []
        self.events = []
        self.device = None
        self.weights = {}
        self.saved = False
        self.contract = None
        if self.path:
            import torch

            if worker is not None:
                self.weights = model_weights(worker)
            from .coverage import runtime_contract

            self.contract = runtime_contract()
            props = torch.cuda.get_device_properties(0)
            self.device = {
                "uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                "name": props.name,
                "sm_count": props.multi_processor_count,
                "l2_bytes": props.L2_cache_size,
                "register_bytes_per_sm": props.regs_per_multiprocessor * 4,
                "shared_bytes_per_sm": props.shared_memory_per_multiprocessor,
                "compute_capability": [props.major, props.minor],
            }
            # nvidia-smi reports the hardware clock ceiling, not a measured rate.
            import subprocess

            self.device["max_clock_hz"] = 1e6 * float(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        os.environ["CUDA_VISIBLE_DEVICES"],
                        "--query-gpu=clocks.max.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )

    def begin(self, step_id, mode):
        global _CURRENT
        if not self.path:
            return
        import torch

        _CURRENT = {"step_id": step_id, "mode": mode, "operations": []}
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        self.events.append((start, end))
        self.rows.append(_CURRENT)

    def end(self):
        global _CURRENT
        if self.path:
            self.events[-1][1].record()
        _CURRENT = None

    def save(self, *, incomplete=False):
        """Serialize after request timing, including incomplete diagnostic traces."""
        if not self.path or self.saved:
            return
        import torch

        if not incomplete:
            torch.cuda.synchronize()
            for row, (start, end) in zip(self.rows, self.events, strict=True):
                row["gpu_elapsed_s"] = start.elapsed_time(end) / 1000
        payload = {
            "schema": 1,
            "device": self.device,
            "weights": self.weights,
            "runtime_contract": self.contract,
            "incomplete": incomplete,
            "graph_operations": sorted(_GRAPH_OPERATIONS),
            "steps": self.rows,
        }
        from dataclasses import asdict

        from oh_my_vllm.ir.execution import execution_policy
        from oh_my_vllm.kernels.cuda_backend import provenance
        from oh_my_vllm.kernels.cuda_backend.groupwise import (
            provenance as gemm_provenance,
        )

        payload["execution_policy"] = asdict(execution_policy())
        payload["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
        payload["cuda_provenance"] = provenance()
        payload["cuda_gemm_provenance"] = gemm_provenance()
        path = Path(self.path)
        with path.open("x") as out:
            json.dump(payload, out)
            out.write("\n")
        self.saved = True


def model_weights(worker):
    """Describe loaded runtime parameter tensors, including converted scales."""
    import torch

    from oh_my_vllm.models.qwen import Linear

    result = {}

    def collect(prefix, obj):
        if isinstance(obj, torch.Tensor):
            result[prefix] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "bytes": obj.numel() * obj.element_size(),
            }
        elif isinstance(obj, Linear):
            collect(prefix + ".weight", obj.weight)
            if obj.scale is not None:
                collect(prefix + ".scale", obj.scale)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                collect(f"{prefix}.{index}", value)
        elif obj is not None and hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if key not in ("target", "device", "config", "provenance"):
                    collect(f"{prefix}.{key}", value)

    collect("target", worker.model)
    if getattr(worker, "dspark", None) is not None:
        collect("dspark", worker.dspark.model)
    return result
