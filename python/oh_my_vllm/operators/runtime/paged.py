import operator

import torch

from .native import load
from .torch_adapter import native_shape, native_tensor, torch_tensor, use_stream

FUNCTIONS = ("allocate", "commit_rows", "release_slot", "mapped_bytes")


def _module():
    return load("paged", FUNCTIONS)[0]


def mapped_bytes():
    """Return physical bytes, including pending cleanup, without CUDA calls."""
    return int(_module().mapped_bytes())


class _PagedStorage:
    """Keep the native allocation alive without retaining a Python tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor):
        self._tensor = tensor

    def reserve(self, slot, stop):
        device = self._tensor.device.index
        stream = torch.cuda.current_stream(device)
        with use_stream(f"cuda:{device}", stream):
            _module().commit_rows(
                native_tensor(self._tensor, require_contiguous=False),
                operator.index(slot),
                operator.index(stop),
            )

    def release(self, slot):
        device = self._tensor.device.index
        stream = torch.cuda.current_stream(device)
        with use_stream(f"cuda:{device}", stream):
            _module().release_slot(
                native_tensor(self._tensor, require_contiguous=False),
                operator.index(slot),
            )


def allocate_paged(shape, *, dtype, device):
    """Reserve CUDA addresses without mapping physical pages.

    The tensor and its views retain the native allocation. After its last owner
    is released, cleanup waits for the next Eager allocation, reserve, or release
    on the same device. The byte getter does not collect pending storage.
    """
    shape = tuple(operator.index(extent) for extent in shape)
    device = (
        torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    if device.type != "cuda":
        raise ValueError("paged storage requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    if not isinstance(dtype, torch.dtype):
        raise TypeError("paged storage requires a torch dtype")
    stream = torch.cuda.current_stream(index)
    with use_stream(f"cuda:{index}", stream):
        native = _module().allocate(
            native_shape(shape), str(dtype).removeprefix("torch."), index
        )
        result = torch_tensor(native)
    result._paged_storage = _PagedStorage(native)
    return result
