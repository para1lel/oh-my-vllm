import numpy as np
import tvm_ffi


class Metadata:
    def __init__(self, device, values):
        values = tuple(values)
        self.device = device
        sizes, offsets = {}, []
        for value in values:
            if not isinstance(value, np.ndarray) or value.dtype.itemsize not in (
                1,
                2,
                4,
                8,
                16,
            ):
                raise ValueError(
                    "metadata requires NumPy arrays with aligned element sizes"
                )
            alignment = 16 // value.dtype.itemsize
            size = sizes.get(value.dtype, 0)
            offset = (size + alignment - 1) // alignment * alignment
            offsets.append(offset)
            sizes[value.dtype] = offset + value.size
        self.host_arenas = {
            dtype: np.zeros(size, dtype=dtype) for dtype, size in sizes.items()
        }
        self.device_arenas = {
            dtype: device.empty(value.shape, str(dtype))
            for dtype, value in self.host_arenas.items()
        }
        self.host_tensors = {
            dtype: tvm_ffi.from_dlpack(value)
            for dtype, value in self.host_arenas.items()
        }
        self.host_views = tuple(
            self.host_arenas[value.dtype][offset : offset + value.size].reshape(
                value.shape
            )
            for value, offset in zip(values, offsets, strict=True)
        )
        self.arrays = tuple(
            device.view(self.device_arenas[value.dtype], value.shape, offset)
            for value, offset in zip(values, offsets, strict=True)
        )
        self.update(values)

    def update(self, values):
        values = tuple(values)
        if len(values) != len(self.host_views):
            raise ValueError("metadata array count changed")
        for destination, source in zip(self.host_views, values, strict=True):
            if (
                not isinstance(source, np.ndarray)
                or source.shape != destination.shape
                or source.dtype != destination.dtype
            ):
                raise ValueError("metadata shape or dtype changed")
        for destination, source in zip(self.host_views, values, strict=True):
            np.copyto(destination, source, casting="no")
        for dtype, destination in self.device_arenas.items():
            self.device.module.copy_bytes(destination, self.host_tensors[dtype])
