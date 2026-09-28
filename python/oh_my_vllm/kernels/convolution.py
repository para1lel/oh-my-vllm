"""Width-four depthwise causal convolution with immutable input snapshots."""

import torch

from .backend import NAME, kernel

_conv = kernel("convolution", "_conv")


def causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    """Snapshot source rows before parallel tokens write their candidate states.

    Only the caller-selected destination slots are updated. Negative write slots
    suppress snapshots, allowing long prefills to save only their final state.
    All metadata is GPU-resident; source state is never implicitly mutated.
    """
    if x.ndim != 2 or x.numel() == 0 or weight.shape != (x.shape[1], 4):
        raise ValueError("convolution requires [tokens,C] inputs and [C,4] weights")
    if pool.ndim != 3 or pool.shape[1:] != (x.shape[1], 3):
        raise ValueError("convolution pool must have [slots,C,3] layout")
    if any(t.dtype != torch.bfloat16 for t in (x, weight, pool)):
        raise ValueError("convolution inputs, weights and state must be BF16")
    if sequence_ids.shape != (len(x),) or write_slots.shape != (len(x),):
        raise ValueError("convolution requires per-token sequence and write slots")
    if starts.numel() != read_slots.numel() + 1:
        raise ValueError("convolution offsets do not match read slots")
    metadata = (sequence_ids, starts, read_slots, write_slots)
    if any(t.ndim != 1 or t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise ValueError("convolution metadata must be integer vectors")
    if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
        raise ValueError(
            "convolution rows must be non-overlapping with unit channel stride"
        )
    tensors = (weight, pool, *metadata)
    if any(
        not t.is_cuda or t.device != x.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("convolution tensors must be contiguous on one CUDA device")
    if not x.is_cuda or x.device != pool.device:
        raise ValueError("convolution inputs must share the pool CUDA device")
    sources = pool.index_select(0, read_slots)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if NAME == "cuda":
        _conv()(x, weight, pool, sequence_ids, starts, sources, write_slots, out)
    else:
        # The frozen TileLang path still selects one row for decode and eight
        # for long prefills; CUDA derives its launch shape from the tensor.
        rows = 8 if len(x) >= 128 else 1
        types = tuple(
            str(t.dtype).removeprefix("torch.")
            for t in (sequence_ids, starts, write_slots)
        )
        _conv(x.shape[1], x.stride(0), len(pool), len(read_slots), types, rows)(
            x, weight, pool, sequence_ids, starts, sources, write_slots, out
        )
    return out
