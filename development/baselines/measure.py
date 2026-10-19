import ctypes
import os
import statistics
import uuid
from contextlib import suppress
from pathlib import Path


def require_gpu_lock():
    import torch

    selected = os.environ.get("OH_MY_VLLM_GPU_UUID")
    if not selected or os.environ.get("CUDA_VISIBLE_DEVICES") != selected:
        raise ValueError("select one GPU UUID with scripts/with-gpu.sh")
    target = f"/tmp/oh-my-vllm-{selected}.lock"
    inherited = False
    for descriptor in Path("/proc/self/fd").iterdir():
        with suppress(FileNotFoundError):
            inherited |= str(descriptor.resolve(strict=True)) == target
    if not inherited:
        raise ValueError("GPU wrapper lock descriptor is missing")
    properties = torch.cuda.get_device_properties(0)
    actual = "GPU-" + str(uuid.UUID(str(properties.uuid).removeprefix("GPU-")))
    if actual != selected:
        raise ValueError("CUDA runtime UUID differs from the selected GPU UUID")
    return selected


class Measurement:
    def __init__(self, mode):
        import torch

        self.mode = mode
        self.stream = (
            torch.cuda.Stream() if mode == "graph" else torch.cuda.current_stream()
        )
        runtime = ctypes.CDLL(str(Path(os.environ["CUDA_HOME"]) / "lib64/libcudart.so"))
        l2 = ctypes.c_int()
        status = runtime.cudaDeviceGetAttribute(ctypes.byref(l2), 38, 0)
        if status or l2.value <= 0:
            raise ValueError("cannot read the selected GPU L2 capacity")
        self.l2_bytes = l2.value
        self.eviction_bytes = 1 << (2 * l2.value).bit_length()
        self.eviction = torch.zeros(
            self.eviction_bytes, device="cuda", dtype=torch.uint8
        )

    def invoke(self, operation):
        import torch
        import tvm_ffi

        if self.stream != torch.cuda.current_stream():
            self.stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(self.stream),
            tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), self.stream.cuda_stream),
        ):
            return operation()

    def executable(self, prepared):
        import torch

        for _ in range(2):
            self.invoke(prepared.reset)
            self.invoke(prepared.run)
        torch.cuda.synchronize()
        if self.mode == "eager":
            return lambda: self.invoke(prepared.run), None
        self.invoke(prepared.reset)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.invoke(prepared.run)
        torch.cuda.synchronize()
        return graph.replay, graph

    def time(self, prepared, executable):
        import torch

        if self.stream != torch.cuda.current_stream():
            self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            return self.time_on_stream(prepared, executable)

    def time_on_stream(self, prepared, executable):
        import torch

        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        samples = []
        for index in range(7):
            self.invoke(prepared.reset)
            self.eviction.add_(1)
            start.record()
            executable()
            end.record()
            end.synchronize()
            if index >= 2:
                samples.append(start.elapsed_time(end) * 1e-3)
        return {
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "spread_seconds": max(samples) - min(samples),
        }

    def policy(self):
        return {
            "warmup": 2,
            "repeats": 5,
            "statistic": "median",
            "clock": "CUDA events on the caller stream",
            "reset": "before eviction; outside timed interval",
            "cache_policy": "cold; uint8 read-modify-write across the eviction buffer",
            "l2_bytes": self.l2_bytes,
            "eviction_bytes": self.eviction_bytes,
            "eviction_timed": False,
            "adaptation_timed": True,
            "mode": self.mode,
        }
