"""Owned context/query preparation and complete frozen comparison chains."""

import torch


def prepare_context(packed, weight, positions, cache, slots):
    """Write normalized/rotated K and unchanged V, without unused query work."""
    from .cuda_backend import compiled

    compiled().prepare_context(packed, weight, positions, cache, slots)


def prepare_query(packed, weight, positions):
    """Return normalized/rotated Q from the selected packed Q/gate rows."""
    from .cuda_backend import compiled

    out = torch.empty((len(packed), 24, 256), device=packed.device, dtype=packed.dtype)
    compiled().prepare_query(packed, weight, positions, out)
    return out
