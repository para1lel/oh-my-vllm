from __future__ import annotations

import math
from functools import lru_cache

import torch

from .kernel import retain_cached_tensor


@retain_cached_tensor
@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """
    Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 this applies YaRN: dimensions whose wavelength already
    fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`,
    and the
    `beta_fast`..`beta_slow` band in between is faded across with a linear ramp.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # the dim whose wavelength completes `rotations` turns over the training context
        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """
    Rotate `x` in place, taking adjacent element pairs as complex numbers. Accepts [b,
    s, d] and
    [b, s, h, d]; `inverse` conjugates the rotation, which is how the attention output
    gets the
    query's rotation removed again so the cache can stay in one shared rotated form.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    batch = x.size(0) if freqs_cis.ndim == 3 else 1
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(batch, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(batch, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@retain_cached_tensor
@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """
    Which sliding-window cache slots each query attends to; -1 marks a slot holding
    nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each
    seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed
    oldest first.
    Keep slot order and masked positions for block-wise probability rounding.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def position_rows(
    table: torch.Tensor, position: torch.Tensor, length: int, *, stride=1
):
    """Read shared or request-local positions without a device-to-host read."""
    rows = (
        position
        + torch.arange(length, device=position.device, dtype=torch.int64) * stride
    )
    return table.index_select(0, rows.flatten()).reshape(*rows.shape, *table.shape[1:])


def position_window_topk_idxs(
    window_size: int, bsz: int, position: torch.Tensor
) -> torch.Tensor:
    """`get_window_topk_idxs` for one decode query at a device position.

    The ring offset and the not-yet-filled mask both come from `position`.
    """
    offsets = torch.arange(window_size, device=position.device, dtype=torch.int64)
    oldest = position.remainder(window_size) + 1
    idxs = (offsets + oldest).remainder(window_size)
    idxs = torch.where(idxs > position, -1, idxs)  # ring still filling
    if position.ndim:
        return idxs.int().unsqueeze(1).contiguous()
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def context_token_bucket(end_pos: int) -> int:
    """
    Power-of-two token extent that holds `end_pos`.

    Every end position in `(extent / 2, extent]` maps to the same extent, so a decode
    graph
    captured for one bucket keeps the same shapes for every replay in that bucket.
    """
    if end_pos <= 1:
        return max(end_pos, 0)
    return 1 << (end_pos - 1).bit_length()


def compressed_context_width(end_pos: int, ratio: int, capacity: int) -> int:
    """Compressed rows a decode graph reserves for `end_pos`.

    `capacity` is the cache row count, so the last bucket never exceeds the allocation.
    """
    if end_pos <= 0:
        return 0
    return min(context_token_bucket(end_pos) // ratio, capacity)


@retain_cached_tensor
@lru_cache(1)
def get_prefill_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
):
    """Build causal indices for a chunk after the first prefill chunk.

    The key rows contain the previous ring window followed by the current chunk.
    Leading empty ring slots keep the one-token decode block boundaries.
    """
    previous = min(window_size, start_pos)
    end = (previous + torch.arange(seqlen)).unsqueeze(1)
    indices = end - window_size + 1 + torch.arange(window_size)
    indices = torch.where(indices >= 0, indices, -1)
    return indices.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


@retain_cached_tensor
@lru_cache(1)
def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):
    assert start_pos > 0
    window = torch.arange(min(window_size, start_pos + 1))
    drafts = window_size + torch.arange(block_size)
    # Proposal rows share the full draft block, unlike causal target rows.
    indices = torch.cat((window, drafts))
    return indices.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
