import torch


def _cpu_tensor(value):
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if result.device.type != "cpu" or not result.is_contiguous():
        raise ValueError("metadata requires contiguous CPU Torch tensors")
    if result.dtype.itemsize not in (1, 2, 4, 8, 16):
        raise ValueError("metadata requires aligned element sizes")
    return result


class Metadata:
    def __init__(self, device, values):
        values = tuple(_cpu_tensor(value) for value in values)
        self.device = device
        self.uploads = device.uploads()
        sizes, offsets = {}, []
        for value in values:
            alignment = 16 // value.dtype.itemsize
            size = sizes.get(value.dtype, 0)
            offset = (size + alignment - 1) // alignment * alignment
            offsets.append(offset)
            sizes[value.dtype] = offset + value.numel()
        self.host_arenas = {
            dtype: torch.zeros(size, dtype=dtype) for dtype, size in sizes.items()
        }
        self.device_arenas = {
            dtype: device.scratch(value.shape, dtype)
            for dtype, value in self.host_arenas.items()
        }
        self.host_views = tuple(
            self.host_arenas[value.dtype]
            .narrow(0, offset, value.numel())
            .view(value.shape)
            for value, offset in zip(values, offsets, strict=True)
        )
        self.arrays = tuple(
            device.view(self.device_arenas[value.dtype], value.shape, offset)
            for value, offset in zip(values, offsets, strict=True)
        )
        self.update(values)

    def close(self):
        self.uploads.close()

    def update(self, values):
        values = tuple(_cpu_tensor(value) for value in values)
        if len(values) != len(self.host_views):
            raise ValueError("metadata array count changed")
        for destination, source in zip(self.host_views, values, strict=True):
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError("metadata shape or dtype changed")
        for destination, source in zip(self.host_views, values, strict=True):
            destination.copy_(source)
        for dtype, destination in self.device_arenas.items():
            self.uploads.copy(destination, self.host_arenas[dtype])
