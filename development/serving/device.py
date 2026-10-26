import ctypes
import os
import subprocess
import uuid as uuid_module

from .gpu import Settings, parse_uuids


def selected_uuid():
    identities = parse_uuids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if len(identities) != 1:
        raise ValueError("use the single-GPU wrapper with a GPU UUID")
    return selected_uuids()[0]


def selected_uuids():
    identities = parse_uuids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if len(identities) not in (1, 4):
        raise ValueError("use the GPU wrapper with one or four GPU UUIDs")
    settings = Settings.from_environment(
        {"OH_MY_VLLM_GPU_COUNT": str(len(identities)), **os.environ}
    )
    if len(identities) != settings.count or (
        settings.requested and identities != settings.requested
    ):
        raise ValueError("CUDA selection differs from the requested GPU group")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGetUuid_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    count = ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)):
        raise RuntimeError("cannot read the CUDA device count")
    if count.value != len(identities):
        raise ValueError("CUDA device count differs from the GPU wrapper")
    for ordinal, expected in enumerate(identities):
        handle = ctypes.c_int()
        identity = (ctypes.c_ubyte * 16)()
        if driver.cuDeviceGet(
            ctypes.byref(handle), ordinal
        ) or driver.cuDeviceGetUuid_v2(identity, handle.value):
            raise RuntimeError("cannot read the CUDA device UUID")
        actual = "GPU-" + str(uuid_module.UUID(bytes=bytes(identity)))
        if actual != expected:
            raise ValueError(
                "CUDA logical device UUID differs from the GPU wrapper order"
            )
    return identities


def device_identity(uuid):
    keys = ("uuid", "name", "driver_version", "clocks.mem", "power.limit")
    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                uuid,
                f"--query-gpu={','.join(keys)}",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    return dict(zip(keys, (value.strip() for value in values), strict=True))
