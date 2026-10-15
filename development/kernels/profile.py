import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SECTIONS = (
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
    "MemoryWorkloadAnalysis",
    "WarpStateStats",
    "SchedulerStats",
    "SourceCounters",
)
METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes.sum.per_second",
    "lts__t_sector_hit_rate.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "derived__local_spilling_requests",
    "derived__local_spilling_requests_pct",
)


def run(command, log, environment):
    with log.open("w") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see {log}")


def counter_summary(path):
    with path.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("profiler supplied no counter table")
    units = rows[0]
    fields = [
        *METRICS,
        *(
            name
            for name in units
            if name.startswith("smsp__average_warps_issue_stalled")
        ),
    ]
    kernels = [
        {
            "id": row["ID"],
            "kernel": row["Kernel Name"],
            "metrics": {
                name: {
                    "value": row.get(name) or "unavailable",
                    "unit": units.get(name) or "unavailable",
                }
                for name in fields
            },
        }
        for row in rows
        if row.get("ID", "").isdigit()
    ]
    if not kernels:
        raise ValueError("profiler supplied no kernel counters")
    return kernels


def profile(arguments):
    directory = arguments.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "temporary"
    temporary.mkdir()
    environment = {**os.environ, "TMPDIR": str(temporary)}
    cuda = Path(environment.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    application = [
        sys.executable,
        "-m",
        "development.kernels",
        "bench",
        "--model",
        str(arguments.model),
        "--operator",
        arguments.operator,
        "--mode",
        arguments.mode,
        "--rows",
        str(arguments.rows),
        "--profile",
    ]
    manifest = {"accepted": False, "commands": [], "metrics": "unavailable"}
    try:
        command = [
            str(cuda / "nsys"),
            "profile",
            "--trace=cuda",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={directory / 'systems'}",
            *application,
            "--output",
            str(directory / "systems-check.json"),
        ]
        manifest["commands"].append(command)
        run(command, directory / "systems.log", environment)
        run(
            [
                str(cuda / "nsys"),
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                "--output",
                str(directory / "systems"),
                str(directory / "systems.nsys-rep"),
            ],
            directory / "systems-stats.log",
            environment,
        )
        command = [str(cuda / "ncu"), "--profile-from-start", "off"]
        for section in SECTIONS:
            command += ["--section", section]
        command += [
            "--metrics",
            "dram__bytes_read.sum,dram__bytes_write.sum",
            "--export",
            str(directory / "compute"),
            *application,
            "--output",
            str(directory / "compute-check.json"),
        ]
        manifest["commands"].append(command)
        run(command, directory / "compute.log", environment)
        run(
            [
                str(cuda / "ncu"),
                "--import",
                str(directory / "compute.ncu-rep"),
                "--csv",
                "--page",
                "raw",
            ],
            directory / "counters.csv",
            environment,
        )
        manifest["metrics"] = counter_summary(directory / "counters.csv")
        print(f"Profiler artifacts: {directory}")
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        (directory / "profile.json").write_text(json.dumps(manifest, indent=2) + "\n")
