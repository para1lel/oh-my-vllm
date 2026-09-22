"""Width-four depthwise causal convolution with immutable input snapshots."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _conv(c: int, stride: int, slots: int, sequences: int, index_types: tuple, bt: int):
    n = T.dynamic("n")
    seq_dtype, start_dtype, write_dtype = index_types

    @T.prim_func
    def kernel(
        x: T.StridedTensor((n, c), (stride, 1), "bfloat16"),
        weight: T.Tensor((c, 4), "bfloat16"),
        pool: T.Tensor((slots, c, 3), "bfloat16"),
        seq_ids: T.Tensor((n,), seq_dtype),
        starts: T.Tensor((sequences + 1,), start_dtype),
        sources: T.Tensor((sequences, c, 3), "bfloat16"),
        writes: T.Tensor((n,), write_dtype),
        out: T.Tensor((n, c), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(n, bt), T.ceildiv(c, 128), threads=128) as (br, bc):
            total = T.alloc_fragment((bt, 128), "float32")
            T.clear(total)
            for tap in T.unroll(4):
                for i, j in T.Parallel(bt, 128):
                    token, channel = br * bt + i, bc * 128 + j
                    if token < n and channel < c:
                        seq = seq_ids[token]
                        first = starts[seq]
                        pos = token + tap - 3
                        value = T.if_then_else(
                            pos >= first,
                            x[pos, channel],
                            sources[seq, channel, pos - first + 3],
                        ).astype("float32")
                        total[i, j] += value * weight[channel, tap].astype("float32")
                        if tap > 0:
                            target = writes[token].astype("int64")
                            if target >= 0:
                                pool[target, channel, tap - 1] = value
            for i, j in T.Parallel(bt, 128):
                if br * bt + i < n and bc * 128 + j < c:
                    out[br * bt + i, bc * 128 + j] = total[i, j] / (
                        1 + T.exp(-total[i, j])
                    )

    return kernel


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
    types = tuple(
        str(t.dtype).removeprefix("torch.") for t in (sequence_ids, starts, write_slots)
    )
    _conv(x.shape[1], x.stride(0), len(pool), len(read_slots), types, rows)(
        x, weight, pool, sequence_ids, starts, sources, write_slots, out
    )
    return out
