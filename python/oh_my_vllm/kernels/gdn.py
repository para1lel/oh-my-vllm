"""Gated delta recurrence with explicit read and per-token write slots.

State layout is [slot, value_head, value_dim, key_dim]. A verification call
reads its committed state and writes separate candidate states. The scheduler
chooses the accepted slot later; rejected candidates never mutate a prefix.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _normalize_qk(h: int, qs: int, ks: int):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        q: T.StridedTensor((n, h, 128), (qs, 128, 1), "bfloat16"),
        k: T.StridedTensor((n, h, 128), (ks, 128, 1), "bfloat16"),
        oq: T.Tensor((n, h, 128), "bfloat16"),
        ok: T.Tensor((n, h, 128), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(n * h, 8), threads=128) as bx:
            qv = T.alloc_fragment((8, 128), "float32")
            kv = T.alloc_fragment((8, 128), "float32")
            qq = T.alloc_fragment((8, 128), "float32")
            kk = T.alloc_fragment((8, 128), "float32")
            qsum = T.alloc_fragment((8,), "float32")
            ksum = T.alloc_fragment((8,), "float32")
            for i, j in T.Parallel(8, 128):
                row = bx * 8 + i
                qv[i, j] = T.if_then_else(
                    row < n * h, q[row // h, row % h, j].astype("float32"), 0
                )
                kv[i, j] = T.if_then_else(
                    row < n * h, k[row // h, row % h, j].astype("float32"), 0
                )
                qq[i, j] = qv[i, j] * qv[i, j]
                kk[i, j] = kv[i, j] * kv[i, j]
            T.reduce_sum(qq, qsum, dim=1)
            T.reduce_sum(kk, ksum, dim=1)
            for i, j in T.Parallel(8, 128):
                row = bx * 8 + i
                if row < n * h:
                    oq[row // h, row % h, j] = qv[i, j] * T.rsqrt(qsum[i] + 1e-6)
                    ok[row // h, row % h, j] = kv[i, j] * T.rsqrt(ksum[i] + 1e-6)

    return kernel


def normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize validated packed GDN inputs without FP32 intermediate tensors."""
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    _normalize_qk(q.shape[1], q.stride(0), k.stride(0))(q, k, oq, ok)
    return oq, ok


@tilelang.jit
def _recurrent(
    sequences: int,
    slots: int,
    hq: int,
    hv: int,
    bv: int,
    strides: tuple,
    state_dtype: str,
    index_types: tuple,
):
    n = T.dynamic("n")
    qs, ks, vs = strides
    start_dtype, read_dtype, write_dtype = index_types

    @T.prim_func
    def kernel(
        q: T.StridedTensor((n, hq, 128), (qs, 128, 1), "bfloat16"),
        k: T.StridedTensor((n, hq, 128), (ks, 128, 1), "bfloat16"),
        v: T.StridedTensor((n, hv, 128), (vs, 128, 1), "bfloat16"),
        decay: T.Tensor((n, hv), "float32"),
        beta: T.Tensor((n, hv), "float32"),
        pool: T.Tensor((slots, hv, 128, 128), state_dtype),
        starts: T.Tensor((sequences + 1,), start_dtype),
        reads: T.Tensor((sequences,), read_dtype),
        writes: T.Tensor((n,), write_dtype),
        out: T.Tensor((n, hv, 128), "bfloat16"),
    ):
        with T.Kernel(sequences, hv, 128 // bv, threads=128) as (seq, head, block):
            state = T.alloc_fragment((bv, 128), "float32")
            work = T.alloc_fragment((bv, 128), "float32")
            recalled = T.alloc_fragment((bv,), "float32")
            result = T.alloc_fragment((bv,), "float32")
            qv = T.alloc_fragment((128,), "float32")
            kv = T.alloc_fragment((128,), "float32")
            qq = T.alloc_fragment((128,), "float32")
            kk = T.alloc_fragment((128,), "float32")
            qsum = T.alloc_fragment((1,), "float32")
            ksum = T.alloc_fragment((1,), "float32")
            # One sequence benefits from warp-local reductions; larger batches
            # retain the compiler layout for higher state throughput.
            if sequences == 1:
                T.annotate_layout(
                    {
                        state: tilelang.layout.Fragment(
                            (bv, 128),
                            forward_thread_fn=lambda i, j: (i % 4) * 32 + j % 32,
                            forward_index_fn=lambda i, j: (i // 4) * 4 + j // 32,
                        ),
                        work: tilelang.layout.Fragment(
                            (bv, 128),
                            forward_thread_fn=lambda i, j: (i % 4) * 32 + j % 32,
                            forward_index_fn=lambda i, j: (i // 4) * 4 + j // 32,
                        ),
                        qv: tilelang.layout.Fragment(
                            (128,),
                            replicate=4,
                            forward_thread_fn=lambda j, r: r * 32 + j % 32,
                            forward_index_fn=lambda j: j // 32,
                        ),
                        kv: tilelang.layout.Fragment(
                            (128,),
                            replicate=4,
                            forward_thread_fn=lambda j, r: r * 32 + j % 32,
                            forward_index_fn=lambda j: j // 32,
                        ),
                    }
                )
            source = reads[seq].astype("int64")
            qhead = head // (hv // hq)
            for i, j in T.Parallel(bv, 128):
                state[i, j] = pool[source, head, block * bv + i, j].astype("float32")
            # Keep FP32 state across candidates; only snapshots round to pool dtype.
            for token in T.serial(starts[seq], starts[seq + 1]):
                for j in T.Parallel(128):
                    qv[j] = q[token, qhead, j].astype("float32")
                    kv[j] = k[token, qhead, j].astype("float32")
                    qq[j] = qv[j] * qv[j]
                    kk[j] = kv[j] * kv[j]
                T.reduce_sum(qq, qsum, dim=0)
                T.reduce_sum(kk, ksum, dim=0)
                for j in T.Parallel(128):
                    qv[j] *= T.rsqrt(qsum[0] + 1e-6) * (128**-0.5)
                    kv[j] *= T.rsqrt(ksum[0] + 1e-6)
                for i, j in T.Parallel(bv, 128):
                    state[i, j] *= T.exp(decay[token, head])
                    work[i, j] = state[i, j] * kv[j]
                T.reduce_sum(work, recalled, dim=1)
                for i, j in T.Parallel(bv, 128):
                    residual = (
                        v[token, head, block * bv + i].astype("float32") - recalled[i]
                    ) * beta[token, head]
                    state[i, j] += residual * kv[j]
                    work[i, j] = state[i, j] * qv[j]
                T.reduce_sum(work, result, dim=1)
                for i in T.Parallel(bv):
                    out[token, head, block * bv + i] = result[i]
                target = writes[token].astype("int64")
                if target >= 0:
                    for i, j in T.Parallel(bv, 128):
                        pool[target, head, block * bv + i, j] = state[i, j]

    return kernel


def _validate_rows(tensor: torch.Tensor) -> None:
    if (
        not tensor.is_cuda
        or tensor.stride(2) != 1
        or tensor.stride(1) != tensor.shape[2]
        or tensor.stride(0) < tensor.shape[1] * tensor.shape[2]
    ):
        raise ValueError(
            "GDN requires dense head rows with non-overlapping token strides"
        )


def recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    """Decode/verify ragged sequences, storing each token's recurrent state.

    Slot allocation is the caller's responsibility: destinations must be unique,
    must not overwrite another sequence's source, and must not be shared prefixes.
    Index tensors stay on GPU; this function performs no device-to-host reads.
    """
    if q.shape != k.shape or q.ndim != 3 or q.shape[-1] != 128:
        raise ValueError("GDN requires matching [tokens, heads, 128] q/k")
    if v.ndim != 3 or v.shape[0] != q.shape[0] or v.shape[-1] != 128:
        raise ValueError("GDN value shape does not match q/k")
    if (
        q.shape[0] == 0
        or q.shape[1] == 0
        or v.shape[1] == 0
        or v.shape[1] % q.shape[1]
        or pool.shape[1:] != (v.shape[1], 128, 128)
    ):
        raise ValueError("GDN head ratio or state layout is invalid")
    if log_decay.shape != v.shape[:2] or beta.shape != log_decay.shape:
        raise ValueError("GDN gate shape does not match value heads")
    if starts.numel() != read_slots.numel() + 1:
        raise ValueError("GDN sequence offsets do not match read slots")
    if write_slots.numel() != q.shape[0]:
        raise ValueError("GDN requires one candidate state slot per token")
    tensors = (log_decay, beta, pool, starts, read_slots, write_slots)
    for tensor in (q, k, v):
        _validate_rows(tensor)
        if tensor.device != q.device:
            raise ValueError("GDN tensors must share one CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("GDN q/k/v must be BF16")
    if any(t.dtype != torch.float32 for t in (log_decay, beta)):
        raise ValueError("GDN gates must be FP32")
    if pool.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("GDN states must be FP32 or BF16")
    if any(
        t.ndim != 1 or t.dtype not in (torch.int32, torch.int64)
        for t in (starts, read_slots, write_slots)
    ):
        raise ValueError("GDN offsets and slots must be integer vectors")
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != q.device for t in tensors
    ):
        raise ValueError("GDN tensors must be contiguous on the same CUDA device")
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    tile = 32 if read_slots.numel() >= 4 else 16
    types = tuple(
        str(t.dtype).removeprefix("torch.") for t in (starts, read_slots, write_slots)
    )
    _recurrent(
        len(read_slots),
        len(pool),
        q.shape[1],
        v.shape[1],
        tile,
        (q.stride(0), k.stride(0), v.stride(0)),
        str(pool.dtype).removeprefix("torch."),
        types,
    )(q, k, v, log_decay, beta, pool, starts, read_slots, write_slots, output)
    return output


def prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FlashInfer prefill; gates use log-decay at our boundary."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    if q.ndim != 3 or q.shape != k.shape or q.shape[-1] != 128:
        raise ValueError("GDN prefill requires matching [tokens,heads,128] q/k")
    if (
        v.ndim != 3
        or v.shape[0] != q.shape[0]
        or v.shape[-1] != 128
        or q.numel() == 0
        or v.shape[1] == 0
        or v.shape[1] % q.shape[1]
    ):
        raise ValueError("GDN prefill value layout or head ratio is invalid")
    if log_decay.shape != v.shape[:2] or beta.shape != log_decay.shape:
        raise ValueError("GDN prefill gate shape does not match value heads")
    if (
        starts.ndim != 1
        or starts.dtype not in (torch.int32, torch.int64)
        or initial_state.shape != (starts.numel() - 1, v.shape[1], 128, 128)
    ):
        raise ValueError("GDN prefill initial states do not match sequence offsets")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or any(
        t.dtype != torch.float32 for t in (log_decay, beta, initial_state)
    ):
        raise ValueError("GDN prefill requires BF16 q/k/v and FP32 gates/state")
    if any(
        not t.is_cuda or t.device != q.device or not t.is_contiguous()
        for t in (log_decay, beta, initial_state, starts)
    ):
        raise ValueError("GDN prefill tensors must be contiguous on one CUDA device")

    for tensor in (q, k, v):
        _validate_rows(tensor)
        if tensor.device != q.device:
            raise ValueError("GDN tensors must share one CUDA device")

    # FlashInfer 0.6.18 exposes use_qk_l2norm_in_kernel but its prefill body
    # does not implement it. Normalize explicitly, including the model epsilon.
    q, k = normalize_qk(q, k)
    # FlashInfer uses multiplicative decay rather than log-decay.
    return chunk_gated_delta_rule(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=log_decay.float().exp().contiguous(),
        beta=beta.float().contiguous(),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=starts,
        use_qk_l2norm_in_kernel=False,
    )
