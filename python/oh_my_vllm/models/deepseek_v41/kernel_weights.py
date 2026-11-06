from typing import NamedTuple

import torch

from .kernel_graph import _upload_pointer_array, retain_cached_tensor
from .kernel_native import _projection_module

_PACKED_WEIGHTS = {}


class _PackedWeightCacheEntry(NamedTuple):
    packed: tuple
    raw_weight: object
    raw_scales: object
    versions: tuple


def _round128(value):
    return (value + 127) // 128 * 128


def _scale_bytes(rows, width):
    return _round128(rows) * (_round128(width) // 32)


def invalidate_packed_weights(weight, scales):
    storages = {
        (value.device, value.untyped_storage().data_ptr())
        for value in (weight, scales)
        if value is not None
    }
    for key, entry in tuple(_PACKED_WEIGHTS.items()):
        if any(
            (value.device, value.untyped_storage().data_ptr()) in storages
            for value in (entry.raw_weight, entry.raw_scales)
        ):
            del _PACKED_WEIGHTS[key]


@retain_cached_tensor
def _packed_weight(weight, scales, bits):
    # Inference tensors have no version counter. Change them through Linear's
    # state-loading entry point, which invalidates entries for shared storage.
    versions = tuple(
        None if value.is_inference() else value._version for value in (weight, scales)
    )
    key = (
        weight.data_ptr(),
        scales.data_ptr(),
        bits,
        weight.device.index,
        (weight.dtype, weight.shape, weight.stride()),
        (scales.dtype, scales.shape, scales.stride()),
    )
    cached = _PACKED_WEIGHTS.get(key)
    if cached is not None and cached.versions == versions:
        return cached.packed
    module = _projection_module()
    raw_weight = weight.view(torch.uint8)
    raw_scales = scales.view(torch.uint8)
    outputs, stored_width = raw_weight.shape
    width = stored_width * (8 // bits)
    padded = _round128(width) if bits == 4 else width
    if (
        padded == width
        and raw_weight.is_contiguous()
        and raw_weight.data_ptr() % (64 if bits == 4 else 16) == 0
        and (
            raw_weight.data_ptr() + raw_weight.numel() <= raw_scales.data_ptr()
            or raw_scales.data_ptr() + raw_scales.numel() <= raw_weight.data_ptr()
        )
    ):
        packed_weight = raw_weight
    else:
        packed_weight = torch.empty(
            (outputs, padded * bits // 8), dtype=torch.uint8, device=weight.device
        )
    packed_scales = torch.empty(
        (_scale_bytes(outputs, width),), dtype=torch.uint8, device=weight.device
    )
    module.pack_weights(raw_weight, raw_scales, packed_weight, packed_scales, bits)
    packed = (packed_weight, packed_scales, width)
    _PACKED_WEIGHTS[key] = _PackedWeightCacheEntry(packed, weight, scales, versions)
    return packed


def _grouped_pointer_arrays(packed, device):
    weight_ptrs = _upload_pointer_array(
        [weight.data_ptr() for weight, _, _ in packed], "weight", device
    )
    scale_ptrs = _upload_pointer_array(
        [scales.data_ptr() for _, scales, _ in packed], "scale", device
    )
    return weight_ptrs, scale_ptrs


def _static_pointer_arrays(packed, device):
    """Device pointer arrays over every expert; prepared weights never move.

    The pinned host tensors stay with the result set, so the asynchronous copy
    never reads released storage.
    """
    packed = [_packed_weight(weight, scale, 4) for weight, scale in packed]
    weight_host = torch.tensor(
        [weight.data_ptr() for weight, _, _ in packed],
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    scale_host = torch.tensor(
        [scales.data_ptr() for _, scales, _ in packed],
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    weight_ptrs = weight_host.to(device, non_blocking=True)
    scale_ptrs = scale_host.to(device, non_blocking=True)
    weight_ptrs._packed_owners = scale_ptrs._packed_owners = tuple(packed)
    return weight_ptrs, scale_ptrs, weight_host, scale_host
