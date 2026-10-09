"""Development-only TileFoundry analysis and Nsight CSV integration."""

import csv
import io
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def analyze(symbol):
    """Return explicitly estimated HIR metrics, not native-kernel measurements."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="oh-my-vllm-foundry-") as directory:
        output = Path(directory) / "analysis.txt"
        command = [
            str(Path(sys.executable).parent / "tilefoundry"),
            "analyze",
            f"development/kernels/semantics.py:Operators.{symbol}",
            str(output),
            "--compute-cost",
            "--memory",
            "--roofline",
        ]
        environment = {**os.environ}
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        subprocess.run(
            command, cwd=ROOT, env=environment, check=True, capture_output=True
        )
        return dict(
            kind="estimated",
            source="TileFoundry HIR",
            symbol=symbol,
            report=output.read_text(),
            limitation="Representative HIR shape; not native layout or FP64 phase cost",
        )


def read_nsight(path, required_metrics=()):
    """Import separate profiler observations without using its latency as a gate."""
    lines = Path(path).read_text().splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header is None:
        raise ValueError(
            "Nsight CSV metrics missing; check profiler errors/permissions"
        )
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    required = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("incomplete Nsight metric report")
    import math

    launches = {}
    for row in rows:
        try:
            value = float(row["Metric Value"].replace(",", ""))
        except ValueError as error:
            raise ValueError("Nsight returned an unavailable metric value") from error
        if not math.isfinite(value):
            raise ValueError("Nsight returned a nonfinite metric value")
        key = (row.get("Process ID"), row.get("ID"), row["Kernel Name"])
        launches.setdefault(key, set()).add(row["Metric Name"])
    if any(not set(required_metrics) <= found for found in launches.values()):
        raise ValueError("Nsight launch has incomplete requested metrics")
    return dict(
        kind="measured", source="Nsight Compute (separate replay)", metrics=rows
    )


METRICS = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__inst_executed.sum",
)


def _profile_run(command):
    """Own a separate process group and reap profiler workers even on cancellation."""
    import os
    import signal
    from contextlib import suppress

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=dict(os.environ),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        # NCU normally waits for its worker. Also stop any child left after an
        # abnormal exit; only this newly-created process group is addressed.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def collect(case_id, ncu):
    """Collect both backends separately; profiler times never enter comparison."""
    import tempfile

    from benchmarks.measurement import hardware_identity
    from development.kernels.cases import cases
    from development.kernels.reference import source_hashes, verify_reference

    case = next(c for c in cases() if c["id"] == case_id)
    initial_sources = source_hashes()
    report = dict(
        sources_sha256=initial_sources,
        case=case,
        hardware=hardware_identity(),
        reference=verify_reference(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        estimated=analyze(
            "fp8_silu_linear"
            if case["configuration"]["operation"] == "fp8_linear"
            and case["configuration"]["silu"]
            else case["configuration"]["operation"]
        ),
        measured={},
        limitation="Profiler replay only; not timing or performance acceptance",
    )
    with tempfile.TemporaryDirectory(prefix="oh-my-vllm-ncu-") as directory:
        for backend in ("tilelang", "cuda"):
            path = Path(directory) / f"{backend}.csv"
            command = [
                str(ncu),
                "--target-processes",
                "all",
                "--nvtx",
                "--nvtx-include",
                "profile-operation/",
                "--metrics",
                ",".join(METRICS),
                "--csv",
                "--log-file",
                str(path),
                sys.executable,
                "-m",
                "development.kernels.profile_worker",
                "--case",
                case_id,
                "--backend",
                backend,
            ]
            try:
                result = _profile_run(command)
                if source_hashes() != initial_sources:
                    raise RuntimeError("source changed during profiler collection")
                if result.returncode:
                    raise RuntimeError(
                        f"profiler exit {result.returncode}: "
                        + result.stderr[-4000:]
                        + result.stdout[-4000:]
                    )
                report["measured"][backend] = read_nsight(path, METRICS)
            except (OSError, ValueError, RuntimeError) as error:
                report["measured"][backend] = dict(
                    kind="unavailable",
                    error=str(error),
                    profiler_report=path.read_text()[-4000:] if path.exists() else "",
                )
    return report


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--ncu", type=Path, default=Path(shutil.which("ncu") or "ncu"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import signal

    def interrupted(*_):
        raise KeyboardInterrupt("profiler collection interrupted")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        report = collect(args.case, args.ncu)
    finally:
        signal.signal(signal.SIGTERM, previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(r["kind"] != "measured" for r in report["measured"].values()):
        raise SystemExit("Some counters unavailable; see report (no gate waiver)")


if __name__ == "__main__":
    main()
