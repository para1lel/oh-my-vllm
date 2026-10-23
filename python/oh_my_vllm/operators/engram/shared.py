"""Bounded immutable shared bytes; CUDA registration is a separate owner."""

import fcntl
import os
from contextlib import suppress


class SharedWeightFile:
    """Write and seal once at startup. All methods require a single caller."""

    max_write_bytes = 64 << 20

    def __init__(self, size, *, limit):
        if type(size) is not int or type(limit) is not int or not 0 < size <= limit:
            raise ValueError("Shared weight size must fit the supplied byte limit")
        page = os.sysconf("SC_PAGE_SIZE")
        mapped = (size + page - 1) // page * page
        if mapped > limit:
            raise ValueError("Page rounding exceeds the shared weight limit")
        self.size, self.mapped_bytes, self.written = size, mapped, 0
        self.sealed = False
        self.fd = os.memfd_create(
            "oh-my-vllm-weights", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        try:
            os.ftruncate(self.fd, mapped)
        except BaseException:
            self.close()
            raise

    def fileno(self):
        if self.fd is None:
            raise RuntimeError("Shared weight file is closed")
        return self.fd

    def append(self, data):
        descriptor = self.fileno()
        if self.sealed:
            raise RuntimeError("Shared weight file is sealed")
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("Shared weight input must be contiguous")
        view = view.cast("B")
        size = view.nbytes
        if size > self.max_write_bytes or size > self.size - self.written:
            raise ValueError("Shared weight write exceeds its byte limit")
        offset = self.written
        try:
            while view:
                count = os.pwrite(descriptor, view, self.written)
                if count <= 0:
                    raise OSError("Shared weight write made no progress")
                self.written += count
                view = view[count:]
        except BaseException:
            self.close()
            raise
        return offset

    def seal(self):
        descriptor = self.fileno()
        if self.sealed:
            return
        if self.written != self.size:
            raise ValueError("Shared weight payload is incomplete")
        flags = (
            fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SEAL
        )
        try:
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, flags)
        except BaseException:
            self.close()
            raise
        self.sealed = True

    def close(self):
        descriptor, self.fd = self.fd, None
        if descriptor is not None:
            os.close(descriptor)

    def __enter__(self):
        self.fileno()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        with suppress(OSError):
            if getattr(self, "fd", None) is not None:
                self.close()
