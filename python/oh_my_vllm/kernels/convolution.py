"""Width-four depthwise causal convolution with immutable input snapshots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(
    X,
    Weight,
    Pool,
    SeqIds,
    Starts,
    Sources,
    Writes,
    Out,
    C: tl.constexpr,
    XStride: tl.constexpr,
    BC: tl.constexpr,
    Tokens: tl.constexpr,
    BT: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)[:, None]
    channels = tl.program_id(1) * BC + tl.arange(0, BC)[None, :]
    valid = token < Tokens
    seq = tl.load(SeqIds + token, valid, 0)
    first = tl.load(Starts + seq)
    target = tl.load(Writes + token, valid, -1)
    total = tl.full((BT, BC), 0, tl.float32)
    for tap in tl.static_range(4):
        pos = token + tap - 3
        x = tl.load(
            X + pos * XStride + channels, valid & (pos >= first) & (channels < C), 0
        ).to(tl.float32)
        old = tl.load(
            Sources + (seq * C + channels) * 3 + pos - first + 3,
            valid & (pos < first) & (channels < C),
            0,
        ).to(tl.float32)
        value = tl.where(pos >= first, x, old)
        weight = tl.load(Weight + channels * 4 + tap, channels < C, 0).to(tl.float32)
        total += value * weight
        if tap > 0:
            tl.store(
                Pool + (target * C + channels) * 3 + tap - 1,
                value,
                (target >= 0) & (channels < C),
            )
    total = total * tl.sigmoid(total)
    tl.store(Out + token * C + channels, total, valid & (channels < C))


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
    # Tile long-prefill rows to amortize CTA scheduling; decode retains one row.
    rows = 8 if len(x) >= 128 else 1
    _conv[(triton.cdiv(len(x), rows), triton.cdiv(x.shape[1], 128))](
        x,
        weight,
        pool,
        sequence_ids,
        starts,
        sources,
        write_slots,
        out,
        x.shape[1],
        x.stride(0),
        128,
        len(x),
        rows,
    )
    return out
