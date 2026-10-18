import ctypes
import hashlib
import json
import os
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .peaks import device_identity, file_digest, source_id


class LibraryInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def verify_vendor(module, native):
    info = LibraryInfo()
    loader = ctypes.CDLL(None)
    loader.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(LibraryInfo)]
    symbol = ctypes.cast(module.cublasLtGetVersion, ctypes.c_void_p)
    if not loader.dladdr(symbol, ctypes.byref(info)) or not info.path:
        raise RuntimeError("cannot identify the loaded cuBLASLt library")
    path = Path(os.fsdecode(info.path)).resolve()
    if (
        str(path) != native["cublaslt_library"]
        or file_digest(path) != native["cublaslt"]
    ):
        raise ValueError("loaded cuBLASLt image differs from the calibration build")


def build():
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    source = Path(__file__).with_name("peak_kernels.cu")
    compiler = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True)
    flags = ["-O3", "-std=c++17", "-arch=sm_100a", "-shared", "-Xcompiler=-fPIC"]
    vendor = cuda / "lib64/libcublasLt.so"
    identity = dict(
        source_id=source_id(),
        compiler=compiler,
        flags=flags,
        cublaslt=file_digest(vendor),
        cublaslt_library=str(vendor.resolve()),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ["OH_MY_VLLM_KERNEL_CACHE"]) / "calibration" / key
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "peaks.so"
    if not library.exists():
        temporary = cache / f"peaks.{os.getpid()}.so"
        subprocess.run(
            [
                str(cuda / "bin/nvcc"),
                *flags,
                str(source),
                "-L",
                str(cuda / "lib64"),
                "-Xlinker",
                f"-rpath={cuda / 'lib64'}",
                "-lcublasLt",
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(library)
    sass = subprocess.check_output(
        [str(cuda / "bin/cuobjdump"), "--dump-sass", str(library)], text=True
    )
    if "FFMA" not in sass or "DFMA" not in sass:
        raise ValueError(
            "calibration binary does not contain FP32 and FP64 FMA instructions"
        )
    return library, {
        **identity,
        "library": str(library),
        "binary_id": file_digest(library),
        "scalar_instructions": ["FFMA", "DFMA"],
    }


def measure(uuid):
    identity = source_id()
    library, native = build()
    module = ctypes.CDLL(str(library))
    verify_vendor(module, native)
    function = module.measure_peak
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    details = (ctypes.c_int64 * 5)()
    module.gpu_details.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    if module.gpu_details(details):
        raise RuntimeError("cannot read CUDA device properties")
    clocks = []

    def record_clock():
        clocks.append(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id",
                    uuid,
                    "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        )

    before = device_identity(uuid)
    record_clock()
    measurements, peaks = {}, {}
    names = ("hbm", "fp8", "bf16", "tf32", "fp32", "fp64")
    for resource, name in enumerate(names):
        cases = []
        if resource == 0:
            configurations = [
                (size, blocks) for size in (1 << 30, 2 << 30) for blocks in (4, 8, 16)
            ]
        elif resource <= 3:
            configurations = [
                (size, candidate)
                for size in (4096, 8192, 16384)
                for candidate in range(8)
            ]
        else:
            configurations = [(4096, blocks) for blocks in (2, 4, 8)]
        for size, option in configurations:
            output = (ctypes.c_double * 8)()
            error = ctypes.create_string_buffer(1024)
            status = function(resource, size, option, output, error, len(error))
            if status < 0:
                raise RuntimeError(f"{name}/{size}/{option}: {error.value.decode()}")
            if status == 1:
                continue
            cases.append(
                dict(
                    size=size,
                    option=option,
                    work=output[0],
                    samples_seconds=list(output)[1:],
                )
            )
        if not cases:
            raise ValueError(f"no supported calibration case for {name}")
        measurements[name] = cases
        peaks[name] = max(
            case["work"] / statistics.median(case["samples_seconds"]) for case in cases
        )
        print(
            json.dumps({"uuid": uuid, "resource": name, "peak": peaks[name]}),
            flush=True,
        )
        record_clock()
    after = device_identity(uuid)
    if before != after:
        raise ValueError(
            "GPU driver, memory clock, or power limit changed during calibration"
        )
    if source_id() != identity:
        raise ValueError("calibration source changed during measurement")
    verify_vendor(module, native)
    if file_digest(library) != native["binary_id"]:
        raise ValueError("calibration binary changed during measurement")
    return dict(
        version=1,
        uuid=uuid,
        device=after,
        peaks=peaks,
        measurements=measurements,
        source_id=identity,
        native=native,
        hardware=dict(
            zip(
                (
                    "sm_count",
                    "l2_bytes",
                    "compute_major",
                    "compute_minor",
                    "cublaslt_version",
                ),
                details,
                strict=True,
            )
        ),
        clock_snapshots=clocks,
        measured_at=datetime.now(UTC).isoformat(),
        method=dict(
            hbm=(
                "16-byte copy; pseudorandom input; read plus write bytes; "
                "buffers exceed four L2 capacities"
            ),
            tensor=(
                "dense cuBLASLt TN GEMM; FP32 accumulation/output; "
                "FP8 E4M3 tensor scales 1; BF16; FAST_TF32"
            ),
            scalar=(
                "16 independent native FMA accumulators per thread; two FLOPs per FMA"
            ),
            timing=(
                "same-stream CUDA events; 200 ms warmup per case; "
                "seven samples of ten invocations"
            ),
            selection="maximum work / median seconds across configurations",
        ),
    )
