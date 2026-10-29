import torch


def sequence_cache(args, shape, *, dtype=None, ratio=1):
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.get_default_device()
    if getattr(args, "lazy_cache", False) and device.type == "cuda" and all(shape):
        from ...operators.runtime.paged import allocate_paged

        value = allocate_paged(shape, dtype=dtype, device=device)
        value._context_ratio = ratio
        return value
    return torch.zeros(shape, dtype=dtype, device=device)


def release_sequence_slot(value, slot):
    owner = getattr(value, "_paged_storage", None)
    if owner is None:
        value[slot].zero_()
    else:
        owner.release(slot)


def reserve_sequence_slot(value, slot, stop):
    owner = getattr(value, "_paged_storage", None)
    if owner is not None:
        owner.reserve(slot, stop)
