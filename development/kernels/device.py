import ctypes

from oh_my_vllm.worker.device import FUNCTIONS
from oh_my_vllm.worker.device import Device as RuntimeDevice

from .build import build


class Device(RuntimeDevice):
    def __init__(self):
        build("device", FUNCTIONS)
        super().__init__()
        self.eviction_buffers = None
        self.l2_bytes = self.module.l2_bytes()
        self.eviction_bytes = 1 << (2 * self.l2_bytes).bit_length()

    def evict(self):
        if self.eviction_buffers is None:
            self.eviction_buffers = tuple(
                self.empty((self.eviction_bytes // 4,), "uint32") for _ in range(2)
            )
        self.module.evict(*self.eviction_buffers)

    def profile(self, operation):
        self.evict()
        self.synchronize()
        self.checked(self.runtime.cudaProfilerStart())
        try:
            operation()
            self.synchronize()
        finally:
            self.checked(self.runtime.cudaProfilerStop())

    def time(self, operation, reset, *, warmup=2, repeats=5):
        events = [ctypes.c_void_p(), ctypes.c_void_p()]
        created = []
        try:
            for event in events:
                self.checked(self.runtime.cudaEventCreate(ctypes.byref(event)))
                created.append(event)
            measurements = []
            for index in range(warmup + repeats):
                reset()
                self.evict()
                self.checked(self.runtime.cudaEventRecord(events[0], None))
                operation()
                self.checked(self.runtime.cudaEventRecord(events[1], None))
                self.checked(self.runtime.cudaEventSynchronize(events[1]))
                duration = ctypes.c_float()
                self.checked(
                    self.runtime.cudaEventElapsedTime(ctypes.byref(duration), *events)
                )
                if index >= warmup:
                    measurements.append(duration.value * 1e-3)
            return measurements
        finally:
            for event in created:
                self.checked(self.runtime.cudaEventDestroy(event))
