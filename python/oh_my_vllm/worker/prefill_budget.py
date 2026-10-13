"""Conservative capture storage admission for the supported Qwen target."""

import math
from dataclasses import fields

import torch


def prefill_capture_headroom(model, caches, tokens, batch, write_slots) -> int:
    """Count capture storage from planned shapes without reading GPU tensors.

    Warm execution uses the default allocator; capture uses the private pool.
    Count both phase-local activation/output buffers without pool reuse credit.
    Saved mutation rows and cloned metadata coexist with either phase. Keep a
    4 GiB fragmentation/compiler allowance and two 128 MiB vendor workspaces.
    Large-query ragged attention also keeps its measured 32 GiB safety floor.
    This admission estimate does not guarantee future allocator/provider peaks.
    """
    rows = tokens.numel()
    slots = len({slot for slot in write_slots if slot >= 0})
    rollback = sum(
        slots
        * sum(math.prod(state.shape[1:]) * state.element_size() for state in cache)
        if isinstance(cache, tuple)
        else rows * math.prod((cache.shape[1], *cache.shape[3:])) * cache.element_size()
        for cache in caches
    )
    metadata = tokens.numel() * tokens.element_size() + sum(
        value.numel() * value.element_size()
        for field in fields(batch)
        if isinstance(value := getattr(batch, field.name), torch.Tensor)
    )
    for owner, names in (
        (
            batch.attention,
            (
                "query_starts",
                "kv_starts",
                "kv_lengths",
                "subpages",
                "key_slots",
                "value_slots",
            ),
        ),
        (batch.output_attention, ("tables", "lengths")),
    ):
        metadata += sum(
            value.numel() * value.element_size()
            for name in names
            if isinstance(value := getattr(owner, name, None), torch.Tensor)
        )
    # Qwen validates H=5120/I=17408, FA24x256 and GDN16/48x128.
    # Include packed+gated MLP, residual/norm/branch hidden, packed FA QKV,
    # normalized query/attended/gated FA, packed GDN projection+mixed/z,
    # normalized q/k, contiguous v, output/concatenation/gated norm.
    # FA and GDN are mutually exclusive layer kinds. Their branches join
    # before the MLP; layer outputs join before the next layer. Persistent
    # feature taps are counted separately below.
    bf16_columns = 4 * 5120 + max(
        3 * 17408,
        14336 + 3 * 6144,
        16384 + 10240 + 5 * 6144 + 4096,
    )
    # FP8 hidden/intermediate plus FP32 block scales, decay/beta/exp copies.
    layer_scratch = rows * (
        2 * bf16_columns + 5120 + 17408 + 4 * ((5120 + 17408) // 128 + 4 * 48)
    )
    sequences = batch.state_reads.numel()
    state_scratch = sequences * 48 * 128 * 128 * (4 + 4 + 4)
    chunks = (rows + 63) // 64 + sequences - 1
    chunk_scratch = (
        chunks * 48 * (64 * (64 * 2 + 4 + (128 + 64 + 128) * 2 + 128 * 4) + 4)
    )
    # All five individual feature taps coexist with their concatenated output.
    feature_bytes = rows * 5120 * 2 * len(model.feature_layer_ids)
    outputs = rows * 5120 * 2 + 2 * feature_bytes
    required = (
        (4 << 30)
        + rollback
        + metadata
        + 2 * (layer_scratch + state_scratch + chunk_scratch + outputs)
        + (256 << 20)
    )
    if getattr(batch.attention, "key_slots", None) is not None:
        # Compact BF16 K and V [context_tokens,4,256], in both allocators.
        required += 2 * 2 * batch.attention.key_slots.numel() * 4 * 256 * 2
        required = max(32 << 30, required)
    return required
