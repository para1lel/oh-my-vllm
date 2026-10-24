"""Shared text RoPE tables; model frequency selection stays in the adapter.

Construction completes on the caller's stream before returning. Gather positions
must be valid indices. Retain tables and gather inputs through Eager completion
or Graph release. All layers share one window and one compressed table.
"""

import ctypes
import math
from contextlib import contextmanager

import numpy as np
import tvm_ffi

from ...operators.runtime.native import load
from ...operators.transform.api import FUNCTIONS


def _frequencies(compressed):
    theta = 160000 if compressed else 10000
    index = np.arange(32, dtype=np.float64)
    frequencies = theta ** (-2 * index / 64)
    if compressed:
        low = max(
            math.floor(
                64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(theta))
            ),
            0,
        )
        high = min(
            math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(theta))), 63
        )
        ramp = np.clip((index - low) / max(high - low, 1e-3), 0, 1)
        frequencies *= 1 - ramp + ramp / 16
    return frequencies


class RotaryTables:
    def __init__(self, device, length):
        if type(length) is not int or not 1 <= length <= 1048576:
            raise ValueError("rotary table length must be between 1 and 1048576")
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        capture = ctypes.c_int()
        device.checked(
            device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture))
        )
        if capture.value:
            raise RuntimeError("prepare rotary tables outside capture")
        self.module, self.source_id, self.library = load("transform", FUNCTIONS)
        self.device, self.length, self.closed = device, length, False
        self.window = device.empty((length, 32, 2), "float32")
        self.compressed = device.empty((length, 32, 2), "float32")
        frequencies = []
        try:
            for compressed, table in ((False, self.window), (True, self.compressed)):
                frequencies.append(device.upload(_frequencies(compressed)))
                self.module.rope_table(frequencies[-1], table)
        finally:
            device.synchronize_compute()

    def check_open(self):
        if self.closed:
            raise RuntimeError("rotary tables are closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.window, self.compressed, self.module
        try:
            yield
        finally:
            del owners

    def gather(self, positions, output, *, compressed):
        self.check_open()
        if type(compressed) is not bool:
            raise ValueError("compressed rotary selection must be boolean")
        if str(output.dtype) != "float32":
            raise ValueError("rotary coefficients require FP32 storage")
        first, end = output.data_ptr(), output.data_ptr() + output.numel() * 4
        for table in (self.window, self.compressed):
            start = table.data_ptr()
            if max(first, start) < min(end, start + table.numel() * 4):
                raise ValueError("coefficient output overlaps a shared rotary table")
        table = self.compressed if compressed else self.window
        self.device.module.gather_rows(output, table, positions)
        return output

    def close(self):
        self.closed = True


def prepare(device, *, length=1048576):
    return RotaryTables(device, length)
