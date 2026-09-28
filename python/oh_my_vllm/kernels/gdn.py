"""Gated delta recurrence with explicit read and per-token write slots.

State layout is [slot, value_head, value_dim, key_dim]. A verification call
reads its committed state and writes separate candidate states. The scheduler
chooses the accepted slot later; rejected candidates never mutate a prefix.
"""

import torch

from .backend import kernel

_normalize_qk = kernel("gdn", "_normalize_qk")


def normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed GDN inputs without FP32 intermediate tensors.

    Requires nonempty BF16 CUDA [tokens, heads, 128] tensors with matching
    shapes and dense, non-overlapping head rows on the same device.
    """
    if q.dtype != torch.bfloat16:
        raise ValueError(f"normalize_qk requires bfloat16, got {q.dtype}")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"normalize_qk requires bfloat16, got {k.dtype}")
    if q.shape != k.shape:
        raise ValueError(f"q and k shapes must match: {q.shape} vs {k.shape}")
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("normalize_qk requires nonempty [tokens, heads, 128] tensors")
    head_dim = q.shape[-1]
    if head_dim != 128:
        raise ValueError(f"normalize_qk requires head_dim=128, got {head_dim}")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("normalize_qk requires unit stride in last dimension")
    if q.stride(-2) != 128 or k.stride(-2) != 128:
        raise ValueError("normalize_qk requires head_stride=128")
    if q.stride(0) < q.shape[1] * 128 or k.stride(0) < k.shape[1] * 128:
        raise ValueError("normalize_qk requires non-overlapping token rows")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("normalize_qk requires tensors on the same CUDA device")
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    _normalize_qk(q.shape[1], q.stride(0), k.stride(0))(q, k, oq, ok)
    return oq, ok


_recurrent = kernel("gdn", "_recurrent")


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
