"""Keep cuBLAS workspaces in the memory pool that owns captured execution."""

import re
from contextlib import contextmanager

import torch


def validate_graph_allocator() -> None:
    """Reject allocator settings that cannot retain a live graph's workspace.

    Read effective settings, including changes through PyTorch's runtime API.
    Do not replace a caller's explicit allocator configuration.
    """
    backend = torch.cuda.get_allocator_backend()
    settings = torch._C._accelerator_getAllocatorSettings()
    enabled = torch._C._cuda_cudaCachingAllocator_is_enabled()
    if (
        backend != "native"
        or not enabled
        or re.search(r"(?:^|,)\s*max_split_size_mb\s*:", settings)
    ):
        raise ValueError(
            "CUDA graphs require an enabled native allocator "
            "without max_split_size_mb; "
            "use OH_MY_VLLM_ENFORCE_EAGER=1 for other allocator configurations"
        )


@contextmanager
def capture_graph(graph, *, pool=None):
    """Allocate each capture's cuBLAS workspace in its graph's private pool.

    The pinned PyTorch caches workspaces by handle and stream. A graph reset
    clears that cache even if another graph embeds the same raw address. Clear
    before capture so its first GEMM allocates inside the owning graph pool.
    Synchronization protects previous GPU work before releasing cached owners.
    No workspace budget or GEMM algorithm setting is changed.

    Captures and replays are serial on the worker host thread. Shared-pool
    outputs must be consumed before another capture or replay in that family.
    """
    validate_graph_allocator()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    with torch.cuda.graph(graph, pool=pool):
        yield
