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
            limitation=(
                "Representative unrolled GDN recurrence; not native CuTe chunk "
                "arithmetic or strided-copy traffic"
                if symbol == "gdn_prefill"
                else "Representative HIR shape; not native layout or FP64 phase cost"
            ),
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
    "dram__bytes.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "lts__t_bytes.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
)


def profile_provenance(stdout, stderr=""):
    """Keep load identities emitted by this worker, without inferring other loads."""
    import json

    records = {}
    for marker, key in (
        ("CUDA_BUILD_PROVENANCE ", "pointwise"),
        ("CUDA_GEMM_BUILD_PROVENANCE ", "gemm"),
        ("PROFILE_WORKER_IDENTITY ", "worker"),
    ):
        found = [
            json.loads(line[len(marker) :])
            for line in (stdout + "\n" + stderr).splitlines()
            if line.startswith(marker)
        ]
        if len(found) > 1 or any(not isinstance(row, dict) for row in found):
            raise ValueError("invalid or duplicate profiler worker load identity")
        if found:
            records[key] = found[0]
    return records


def validate_profile_worker(metrics, worker, case_id, backend):
    """Reject counters from another process, case, or selected backend."""
    if (
        worker.get("case") != case_id
        or worker.get("backend") != backend
        or type(worker.get("pid")) is not int
        or {row["Process ID"] for row in metrics} != {str(worker["pid"])}
    ):
        raise ValueError("profiler metrics do not match the selected worker")


def validate_profile_loads(records, backend, config, sources, gemm_inputs):
    """Bind required owned providers to source, build inputs, and actual SO bytes."""
    import hashlib
    import json
    import re

    from development.kernels.fixtures import ENTRIES

    if backend != "cuda":
        return
    if config["operation"] not in ENTRIES:
        raise ValueError("unknown profiler provider requirement")
    required = {"pointwise"}
    if (
        config["operation"]
        in {
            "fp8_linear",
            "add_norm_fp8_linear",
            "gated_norm_fp8_linear",
        }
        and config["tokens"] > 32
    ):
        required.add("gemm")
    for key in required:
        record = records.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"missing profiler {key} load identity")
        for field in ("source_sha256", "build_input_sha256", "so_sha256"):
            if not isinstance(record.get(field), str) or not re.fullmatch(
                r"[0-9a-f]{64}", record[field]
            ):
                raise ValueError(f"invalid profiler {key} {field}")
        filename = "kernels.cu" if key == "pointwise" else "groupwise_fp8.cu"
        source = "python/oh_my_vllm/kernels/cuda_backend/" + filename
        if record["source_sha256"] != sources.get(source):
            raise ValueError(f"profiler {key} source does not match collection")
        if key == "pointwise":
            prefix = "python/oh_my_vllm/kernels/cuda_backend/"
            headers = {
                path.removeprefix(prefix): digest
                for path, digest in sources.items()
                if path.startswith(prefix + "operators/") and path.endswith(".cuh")
            }
            if not headers or record.get("project_headers") != headers:
                raise ValueError(
                    "profiler CUDA operator headers differ from collection"
                )
        if key == "gemm" and any(record.get(k) != v for k, v in gemm_inputs.items()):
            raise ValueError("profiler GEMM templates or build inputs differ")
        if key == "gemm":
            identity = {
                k: v
                for k, v in record.items()
                if k not in {"so_path", "so_sha256", "build_input_sha256"}
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            if digest != record["build_input_sha256"]:
                raise ValueError("profiler GEMM build digest does not match its inputs")
        compiler = record.get("nvcc_path" if key == "pointwise" else "compiler")
        version = record.get(
            "nvcc_version" if key == "pointwise" else "compiler_version"
        )
        if not compiler or not version or str(version).startswith("nvcc-unavailable:"):
            raise ValueError(f"missing profiler {key} compiler identity")
        if key == "pointwise" and (
            record.get("compiler_identity_source") not in {"live", "manifest"}
            or record.get("nvcc_version") == "nvcc-changed-during-build"
            or type(record.get("pdl")) is not bool
        ):
            raise ValueError("invalid profiler pointwise compiler or PDL identity")
        if not isinstance(record.get("so_path"), str):
            raise ValueError(f"missing profiler {key} library path")
        if (
            hashlib.sha256(Path(record["so_path"]).read_bytes()).hexdigest()
            != record["so_sha256"]
        ):
            raise ValueError(f"profiler {key} loaded library bytes changed")


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

    from oh_my_vllm.performance.coverage import runtime_contract

    from benchmarks.measurement import hardware_identity
    from development.kernels.cases import cases
    from development.kernels.reference import source_hashes, verify_reference

    case = next(c for c in cases() if c["id"] == case_id)
    initial_sources = source_hashes()
    gemm_inputs = runtime_contract()["gemm_inputs"]
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
                observation = read_nsight(path, METRICS)
                observation["loaded_cuda"] = profile_provenance(
                    result.stdout, result.stderr
                )
                worker = observation["loaded_cuda"].get("worker", {})
                validate_profile_worker(
                    observation["metrics"], worker, case_id, backend
                )
                validate_profile_loads(
                    observation["loaded_cuda"],
                    backend,
                    case["configuration"],
                    initial_sources,
                    gemm_inputs,
                )
                observation["worker_stdout"] = result.stdout
                observation["worker_stderr"] = result.stderr
                report["measured"][backend] = observation
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
