import operator

from .native import load

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
        import torch
        import tvm_ffi

        device = self._tensor.device.index
        stream = torch.cuda.current_stream(device)
        with tvm_ffi.use_raw_stream(
            tvm_ffi.device(f"cuda:{device}"), stream.cuda_stream
        ):
            _module().commit_rows(
                self._tensor, operator.index(slot), operator.index(stop)
            )

    def release(self, slot):
        import torch
        import tvm_ffi

        device = self._tensor.device.index
        stream = torch.cuda.current_stream(device)
        with tvm_ffi.use_raw_stream(
            tvm_ffi.device(f"cuda:{device}"), stream.cuda_stream
        ):
            _module().release_slot(self._tensor, operator.index(slot))


def allocate_paged(shape, *, dtype, device):
    """Reserve CUDA addresses without mapping physical pages.

    The tensor and its views retain the native allocation. After its last owner
    is released, cleanup waits for the next Eager allocation, reserve, or release
    on the same device. The byte getter does not collect pending storage.
    """
    import torch
    import tvm_ffi

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
    with tvm_ffi.use_raw_stream(tvm_ffi.device(f"cuda:{index}"), stream.cuda_stream):
        native = _module().allocate(
            tvm_ffi.Shape(shape), str(dtype).removeprefix("torch."), index
        )
        result = torch.from_dlpack(native)
    result._paged_storage = _PagedStorage(native)
    return result
