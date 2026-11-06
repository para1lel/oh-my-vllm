import ctypes
import subprocess
import uuid as uuid_module


def cuda_device_identity(ordinal=0):
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGetUuid_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    handle = ctypes.c_int()
    identity = (ctypes.c_ubyte * 16)()
    if (
        driver.cuInit(0)
        or driver.cuDeviceGet(ctypes.byref(handle), ordinal)
        or driver.cuDeviceGetUuid_v2(identity, handle.value)
    ):
        raise RuntimeError("cannot read the CUDA device UUID")
    result = {"uuid": "GPU-" + str(uuid_module.UUID(bytes=bytes(identity)))}
    for name, attribute in (("multiprocessor_count", 16), ("l2_cache_bytes", 38)):
        value = ctypes.c_int()
        if driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle.value):
            raise RuntimeError(f"cannot read CUDA device attribute: {name}")
        if value.value <= 0:
            raise RuntimeError(f"invalid CUDA device attribute: {name}")
        result[name] = value.value
    return result


def device_identity(uuid=None):
    properties = cuda_device_identity()
    if uuid is not None and properties["uuid"] != uuid:
        raise ValueError("CUDA device UUID differs from the selected GPU")
    keys = ("uuid", "name", "driver_version", "clocks.mem", "power.limit")
    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                properties["uuid"],
                f"--query-gpu={','.join(keys)}",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    result = dict(zip(keys, (value.strip() for value in values), strict=True))
    if result["uuid"] != properties["uuid"]:
        raise ValueError("device metadata came from a different UUID")
    return result | properties
