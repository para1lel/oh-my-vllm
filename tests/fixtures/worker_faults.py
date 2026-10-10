"""Explicit fault injection for external serving tests.

Production imports no code from this module. A test-owned worker launcher can
call install_fault() before the real bridge starts. The injector fails once per
marker path, so a reconstructed worker can serve the recovery probe.
"""

import os
from pathlib import Path

import torch


def install_fault(stage: str, kind: str, marker: str | Path) -> None:
    if stage not in ("after_commit", "graph_capture", "sampling", "proposal"):
        raise ValueError("unknown worker fault stage")
    if kind not in ("oom", "device", "internal", "real_oom"):
        raise ValueError("unknown worker fault kind")
    marker = Path(marker)

    def inject() -> None:
        # O_EXCL also coordinates fresh worker processes. The marker is a
        # test artifact; it never belongs in a production model directory.
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return
        os.close(descriptor)
        if kind == "oom":
            raise torch.cuda.OutOfMemoryError(f"test-owned OOM at {stage}")
        if kind == "device":
            raise torch.AcceleratorError(f"test-owned CUDA device fault at {stage}")
        if kind == "internal":
            raise RuntimeError(f"test-owned internal fault at {stage}")
        # Exceed physical free memory and cached reservations. This must enter
        # the real CUDA allocator OOM path, without consuming another GPU.
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        torch.empty(free + cached + (1 << 30), dtype=torch.uint8, device="cuda")
        raise AssertionError(
            "real OOM injector unexpectedly allocated its oversized buffer"
        )

    def after(owner, attribute):
        original = getattr(owner, attribute)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            inject()
            return result

        setattr(owner, attribute, wrapped)

    if stage == "after_commit":
        from oh_my_vllm.worker.model_runner import OhMyVllmWorker

        after(OhMyVllmWorker, "_commit_plans")
    elif stage == "sampling":
        from oh_my_vllm.worker import model_runner
        from oh_my_vllm.worker.sampler import RequestSampler

        after(model_runner, "greedy_rows")
        after(RequestSampler, "draw_rows")
    elif stage == "proposal":
        from oh_my_vllm.worker.dspark import DSpark
        from oh_my_vllm.worker.mtp import MTP

        after(DSpark, "propose")
        after(MTP, "propose")
    else:
        from oh_my_vllm.worker.graph_cache import GraphCache

        original = GraphCache.get_or_create

        def get_or_create(self, family, key, create):
            def capture():
                result = create()
                inject()
                return result

            return original(self, family, key, capture)

        GraphCache.get_or_create = get_or_create
