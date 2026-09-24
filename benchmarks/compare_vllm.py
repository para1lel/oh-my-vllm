"""Compare framework measurements against the original frozen EngineCore data.

Run under scripts/with-gpu.sh. Only framework processes are launched; the baseline
is immutable data and there is no code path that installs or runs vLLM.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "941aef26a73048ae9759a0303fdc94c92d1cefd6cc5ee532b99f622710b3884e"


def runtime_identity():
    """Describe the independent environment; reject accidental legacy execution."""
    from oh_my_vllm.worker.runtime import identity

    worker = Path(os.environ.get("OH_MY_VLLM_WORKER_PYTHON", sys.executable))
    if worker.resolve() != Path(sys.executable).resolve():
        raise RuntimeError("benchmark worker must use the current independent Python")
    return identity()


def historical_baseline(path, args, batch, *, data_bytes=None):
    """Select an exact workload from frozen data without starting vLLM."""
    data_bytes = Path(path).read_bytes() if data_bytes is None else data_bytes
    if hashlib.sha256(data_bytes).hexdigest() != BASELINE_SHA256:
        raise ValueError("baseline hash differs from the original frozen artifact")
    artifact = json.loads(data_bytes)
    expected_protocol = {
        "model": args.model,
        "block_size": 784,
        "max_model_len": 65536,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 32768,
        "language_model_only": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "async_scheduling": False,
        "temperature": 0,
        "ignore_eos": True,
        "detokenize": False,
        "mtp_ssm_dtype": "bfloat16",
        "ordinary_prefix_ssm_dtype": "auto",
    }
    if not isinstance(artifact, dict) or any(
        artifact.get("protocol", {}).get(key) != value
        for key, value in expected_protocol.items()
    ):
        raise ValueError("historical baseline protocol is missing or does not match")
    if artifact.get("warmup_per_engine") != args.warmup:
        raise ValueError("historical baseline warmup count does not match")
    rows = artifact.get("rows", [])
    matches = [
        row
        for row in rows
        if all(
            row.get(key) == value
            for key, value in {
                "mode": args.mode,
                "batch_size": batch,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "num_gpu_blocks": args.num_gpu_blocks,
                "speculative_tokens": args.speculative_tokens
                if args.mode == "mtp"
                else 0,
            }.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "historical baseline must contain exactly one matching workload"
        )
    selected = matches[0]
    if len(selected["baseline"]) < 3:
        raise ValueError("historical baseline has fewer than three measurements")
    if any(
        row["output_tokens"] != batch * args.output_len or row["output_tps"] <= 0
        for row in selected["baseline"]
    ):
        raise ValueError("historical baseline contains incomplete measurements")
    return selected


def assert_gpu_exclusive(gpu, process_group):
    """Reject measurements contaminated by non-cooperating GPU clients."""
    snapshot = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    for line in snapshot.splitlines():
        uuid, pid = (part.strip() for part in line.split(",", 1))
        if uuid != gpu:
            continue
        try:
            group = os.getpgid(int(pid))
        except ProcessLookupError:
            continue  # The process exited between the snapshot and lookup.
        if group != process_group:
            raise RuntimeError(f"GPU contention on {gpu}: external process {pid}")


def run_engine(command, timeout=3600, stderr=None, log_path=None):
    """Own the entire engine process group, including model-worker children."""
    with ExitStack() as stack:
        output = stack.enter_context(Path(log_path).open("w")) if log_path else None
        return _run_engine(command, timeout, stderr, output, log_path)


def _run_engine(command, timeout, stderr, output, log_path):
    process = subprocess.Popen(
        command,
        text=True,
        stdout=output if output is not None else subprocess.PIPE,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + timeout
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        while True:
            if gpu.startswith("GPU-"):
                assert_gpu_exclusive(gpu, process.pid)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                if output is not None:
                    process.wait(timeout=min(1, remaining))
                    stdout = Path(log_path).read_text()
                else:
                    stdout, _ = process.communicate(timeout=min(1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if gpu.startswith("GPU-"):
            assert_gpu_exclusive(gpu, process.pid)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout)
        return stdout
    finally:
        # A successful parent can also leave descendants behind.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if process.stdout is not None:
            process.stdout.close()


def source_identity(binary):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--porcelain").decode(),
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "python_sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "python").rglob("*.py"))
        },
    }


def parse_rows(stdout: str, marker: str):
    return [
        json.loads(line[len(marker) :])
        for line in stdout.splitlines()
        if line.startswith(marker)
    ]


def compare(args):
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected_gpu.startswith("GPU-") or "," in selected_gpu:
        raise RuntimeError(
            "Run this script through scripts/with-gpu.sh to select one idle GPU UUID"
        )
    if os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") == "1" or os.environ.get(
        "OH_MY_VLLM_WORKER_PYTHON", ""
    ).endswith("probe_worker.py"):
        raise RuntimeError(
            "Disable accuracy probes/eager override for matched benchmarks"
        )
    args.binary = args.binary.resolve()
    identity = source_identity(args.binary)
    runtime = runtime_identity()
    baseline_bytes = args.baseline_json.read_bytes()
    all_results = []
    for batch in args.batch_sizes:
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-bench-") as directory:
            binary_snapshot = Path(directory) / "worker"
            shutil.copy2(args.binary, binary_snapshot)
            if (
                hashlib.sha256(binary_snapshot.read_bytes()).hexdigest()
                != identity["binary_sha256"]
            ):
                raise RuntimeError("binary changed after identity capture")
            historical = historical_baseline(
                args.baseline_json, args, batch, data_bytes=baseline_bytes
            )
            base_rows = [
                {"version": historical["vllm_version"], "runs": historical["baseline"]}
            ]
            command = [
                str(binary_snapshot),
                "--model",
                args.model,
                "--socket",
                str(Path(directory) / "worker.ipc"),
                "--num-gpu-blocks",
                str(args.num_gpu_blocks),
                "--num-speculative-tokens",
                str(args.speculative_tokens if args.mode == "mtp" else 0),
                "bench",
                "--batch-size",
                str(batch),
                "--input-len",
                str(args.input_len),
                "--output-len",
                str(args.output_len),
                "--warmup",
                str(args.warmup),
                "--repetitions",
                str(args.repetitions),
            ]
            if args.mode == "prefix":
                command.append("--prefix-hit")
            ours_run = run_engine(command)
            ours = parse_rows(ours_run, "BENCH_RESULT ")
        if len(ours) != args.repetitions:
            raise RuntimeError(f"missing framework measurements: {ours_run}")
        if any(row["output_tokens"] != batch * args.output_len for row in ours):
            raise RuntimeError("incomplete framework output")
        base = base_rows[0]["runs"]
        expected_hits = (
            batch * ((args.input_len - 1) // 784 * 784) if args.mode == "prefix" else 0
        )
        if any(row["prefix_hit_tokens"] != expected_hits for row in base + ours):
            raise RuntimeError("engines did not use the expected matched cache state")
        if args.mode == "mtp" and any(
            row["proposed_draft_tokens"] <= 0 or row["accepted_draft_tokens"] <= 0
            for row in base + ours
        ):
            raise RuntimeError("MTP did not produce draft tokens")
        base_median = statistics.median(row["output_tps"] for row in base)
        ours_median = statistics.median(row["output_tps"] for row in ours)
        for label, runs in (("baseline", base), ("candidate", ours)):
            values = [row["output_tps"] for row in runs]
            med = statistics.median(values)
            spread = (max(values) - min(values)) / med if med > 0 else 0
            if spread > 0.10:
                print(
                    f"WARNING: {label} batch={batch} spread={spread:.3f} > 0.10"
                    " — run is unstable, results are unreliable",
                    file=sys.stderr,
                )
        result = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "gpu": selected_gpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source": identity,
            "vllm_version": base_rows[0]["version"],
            "framework_runtime": runtime,
            "comparison": "historical",
            "baseline_artifact": str(args.baseline_json),
            "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "mode": args.mode,
            "batch_size": batch,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_gpu_blocks": args.num_gpu_blocks,
            "speculative_tokens": args.speculative_tokens if args.mode == "mtp" else 0,
            "baseline": base,
            "ours": ours,
            "ratio": ours_median / base_median,
            "passed": ours_median >= 0.95 * base_median,
        }
        all_results.append(result)
        print(json.dumps(result), flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps(all_results, indent=2) + "\n")
    if not all(result["passed"] for result in all_results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data0/shared/Qwen3.8-27B-FP8")
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["ordinary", "mtp", "prefix"], default="ordinary"
    )
    parser.add_argument("--speculative-tokens", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=ROOT / "bench/baseline/2026-09-19-acceptance.json",
        help="Use frozen workload rows; run only the framework, never vLLM baseline",
    )
    args = parser.parse_args()
    if args.repetitions < 5 or args.warmup < 2:
        parser.error("acceptance requires >=5 measurements and >=2 warmups")
    if args.mode == "mtp" and args.speculative_tokens <= 0:
        parser.error("MTP requires positive --speculative-tokens")
    if min(args.batch_sizes) <= 0 or min(args.input_len, args.output_len) <= 0:
        parser.error("batch sizes and token lengths must be positive")
    compare(args)


if __name__ == "__main__":
    main()
