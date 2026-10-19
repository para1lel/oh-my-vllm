import ctypes
import os
import subprocess
import uuid as uuid_module


def selected_uuid():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("use the single-GPU wrapper with a GPU UUID")
    driver = ctypes.CDLL("libcuda.so.1")
    identity = (ctypes.c_ubyte * 16)()
    driver.cuDeviceGetUuid_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if driver.cuInit(0) or driver.cuDeviceGetUuid_v2(identity, 0):
        raise RuntimeError("cannot read the CUDA device UUID")
    actual = "GPU-" + str(uuid_module.UUID(bytes=bytes(identity)))
    if actual != uuid:
        raise ValueError("CUDA device UUID differs from the GPU wrapper")
    return uuid


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
