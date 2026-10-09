"""Owned context/query preparation and complete frozen comparison chains."""

import torch

from .backend import NAME


def tilelang_context(packed, weight, positions, cache, slots):
    from .tilelang_reference.attention import append
    from .tilelang_reference.normalization import rms_rotary

    keys = rms_rotary(packed[:, :1024].reshape(-1, 4, 256), weight, positions)
    values = packed[:, 1024:].reshape(-1, 4, 256).contiguous()
    append(cache, keys, values, slots)


def tilelang_query(packed, weight, positions):
    from .tilelang_reference.normalization import rms_rotary

    return rms_rotary(packed.reshape(-1, 24, 512)[..., :256], weight, positions)


def prepare_context(packed, weight, positions, cache, slots):
    """Write normalized/rotated K and unchanged V, without unused query work."""
    if NAME == "tilelang":
        return tilelang_context(packed, weight, positions, cache, slots)
    from .cuda_backend import compiled

    compiled().prepare_context(packed, weight, positions, cache, slots)


def prepare_query(packed, weight, positions):
    """Return normalized/rotated Q from the selected packed Q/gate rows."""
    if NAME == "tilelang":
        return tilelang_query(packed, weight, positions)
    from .cuda_backend import compiled

    out = torch.empty((len(packed), 24, 256), device=packed.device, dtype=packed.dtype)
    compiled().prepare_query(packed, weight, positions, out)
    return out
