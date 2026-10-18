import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import uuid as uuid_module
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Peaks:
    hbm: float
    fp8: float
    bf16: float
    tf32: float
    fp32: float
    fp64: float

    def __post_init__(self):
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in vars(self).values()
        ):
            raise ValueError("measured peaks must be finite and positive")


def profile_peaks(profile, uuid):
    if profile.get("version") != 1 or profile.get("uuid") != uuid or not uuid:
        raise ValueError("calibration version or GPU UUID does not match")
    names = {field.name for field in fields(Peaks)}
    if set(profile.get("peaks", {})) != names:
        raise ValueError("calibration requires all six resource peaks")
    peaks = Peaks(**profile["peaks"])
    if set(profile.get("measurements", {})) != names:
        raise ValueError("calibration measurements are incomplete")
    hardware = profile.get("hardware", {})
    sm_count, l2_bytes = hardware.get("sm_count", 0), hardware.get("l2_bytes", 0)
    if (
        not isinstance(sm_count, int)
        or sm_count <= 0
        or not isinstance(l2_bytes, int)
        or l2_bytes <= 0
    ):
        raise ValueError("calibration requires valid SM count and L2 capacity")
    for name, cases in profile["measurements"].items():
        rates = []
        for case in cases:
            size, option = case.get("size"), case.get("option")
            if name == "hbm":
                valid = (
                    size in (1 << 30, 2 << 30)
                    and option in (4, 8, 16)
                    and size > 4 * l2_bytes
                )
                work = 2 * size if valid else 0
            elif name in {"fp8", "bf16", "tf32"}:
                valid = size in (4096, 8192, 16384) and option in range(8)
                work = 2 * size**3 if valid else 0
            else:
                valid = size == 4096 and option in (2, 4, 8)
                work = sm_count * option * 256 * 16 * size * 2 if valid else 0
            if not valid or case.get("work") != work:
                raise ValueError(
                    "calibration work differs from its kernel configuration"
                )
            samples = case["samples_seconds"]
            if (
                len(samples) != 7
                or not math.isfinite(case["work"])
                or case["work"] <= 0
                or any(not math.isfinite(value) or value <= 0 for value in samples)
            ):
                raise ValueError("calibration requires seven positive timing samples")
            rates.append(case["work"] / statistics.median(samples))
        if not rates or not math.isclose(
            max(rates), getattr(peaks, name), rel_tol=1e-12
        ):
            raise ValueError("peak differs from its best measured median")
    return peaks


def source_id():
    digest = hashlib.sha256()
    for name in ("peaks.py", "peak_measurement.py", "peak_kernels.cu"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_profile(profile, uuid):
    profile_peaks(profile, uuid)
    if profile.get("source_id") != source_id():
        raise ValueError("calibration source has changed; measure the GPU again")
    native = profile["native"]
    if file_digest(native["library"]) != native["binary_id"]:
        raise ValueError("calibration binary has changed")
    if file_digest(native["cublaslt_library"]) != native["cublaslt"]:
        raise ValueError("calibration cuBLASLt library has changed")


def load_profile(path, uuid):
    profile = json.loads(Path(path).read_text())
    verify_profile(profile, uuid)
    return profile


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if arguments.output.resolve().is_relative_to(root):
        parser.error("calibration output must be external to the repository")
    if arguments.output.exists():
        parser.error("use a new calibration output file")
    from .peak_measurement import measure

    profile = measure(selected_uuid())
    profile_peaks(profile, profile["uuid"])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"uuid": profile["uuid"], "peaks": profile["peaks"]}), flush=True)


if __name__ == "__main__":
    main()
