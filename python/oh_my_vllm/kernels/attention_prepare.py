"""Qwen full-attention preparation, including all KV cache side effects."""

import torch

from .backend import NAME


def tilelang_prepare(packed, q_weight, k_weight, positions, cache, slots):
    """The original complete chain, explicitly bound to immutable kernels.

    Preserve contiguous() rather than forcing a copy: single-token V can already
    be contiguous. This is also the formal frozen-chain comparison.
    """
    from .tilelang_reference.attention import append
    from .tilelang_reference.normalization import rms_rotary

    qg, k, v = packed.split([12288, 1024, 1024], -1)
    q = qg.reshape(-1, 24, 512)[..., :256]
    q = rms_rotary(q, q_weight, positions)
    k = rms_rotary(k.reshape(-1, 4, 256), k_weight, positions)
    v = v.reshape(-1, 4, 256).contiguous()
    append(cache, k, v, slots)
    return q


def prepare_attention(packed, q_weight, k_weight, positions, cache, slots):
    """Return normalized/rotated Q and write normalized K plus unchanged V.

    The projection and physical cache are distinct allocations in the model.
    Metadata slots are unique when nonnegative, as for the existing append path.
    """
    if packed.ndim != 2 or packed.shape[1] != 14336 or not packed.shape[0]:
        raise ValueError("attention preparation requires nonempty packed14336 rows")
    if cache.ndim != 5 or cache.shape[1:] != (2, 784, 4, 256) or not cache.shape[0]:
        raise ValueError("attention preparation requires [pages,2,784,4,256] cache")
    if packed.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("attention preparation requires BF16 projection/cache")
    for weight in (q_weight, k_weight):
        if weight.shape != (256,) or weight.dtype != torch.float32:
            raise ValueError("attention preparation requires FP32 norm weights256")
    for index in (positions, slots):
        if index.shape != (len(packed),) or index.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("attention preparation needs integer metadata per token")
    tensors = (packed, q_weight, k_weight, positions, cache, slots)
    if any(
        not t.is_cuda or t.device != packed.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("attention preparation tensors must be contiguous on one GPU")
    first = cache.data_ptr()
    last = first + cache.numel() * cache.element_size()
    if any(
        t.data_ptr() < last and first < t.data_ptr() + t.numel() * t.element_size()
        for t in (packed, q_weight, k_weight, positions, slots)
    ):
        raise ValueError("attention cache must not overlap projection or metadata")
    if NAME == "tilelang":
        return tilelang_prepare(*tensors)
    from .cuda_backend import compiled

    out = torch.empty((len(packed), 24, 256), device=packed.device, dtype=packed.dtype)
    compiled().prepare_attention(*tensors, out)
    return out
