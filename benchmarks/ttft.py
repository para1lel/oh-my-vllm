"""Collect independent EngineCore measurements and compare frozen baseline JSON."""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from compare_vllm import ROOT, parse_rows, run_engine, runtime_identity, source_identity
from measurement import FROZEN_SHA, audit, hardware_identity


def summarize(runs, batch_size, output_len, minimum=5):
    if len(runs) < minimum:
        raise ValueError("at least five repetitions required")
    for row in runs:
        times = row["ttft_s"]
        if len(times) != batch_size or any(
            not math.isfinite(t) or t <= 0 for t in times
        ):
            raise ValueError("invalid per-request TTFT")
        if row["output_tokens"] != batch_size * output_len or row["preemptions"]:
            raise ValueError("incomplete or preempted measurement")
        if not math.isfinite(row["output_tps"]) or row["output_tps"] <= 0:
            raise ValueError("invalid throughput")
    values = {
        "ttft_s": [max(r["ttft_s"]) for r in runs],
        "output_tps": [r["output_tps"] for r in runs],
    }
    return {
        key: {
            "median": statistics.median(v),
            "spread": (max(v) - min(v)) / statistics.median(v),
        }
        for key, v in values.items()
    }


def compare(baseline, candidate):
    if baseline["workload"] != candidate["workload"]:
        raise ValueError("workload mismatch")
    workload = baseline["workload"]
    for artifact in (baseline, candidate):
        hw = artifact.get("hardware", {})
        if (
            not hw.get("gpu", "").startswith("GPU-")
            or not hw.get("cpu_affinity")
            or not hw.get("gpu_info")
        ):
            raise ValueError("missing pinned hardware identity")
    if baseline["hardware"]["cpu_affinity"] != candidate["hardware"]["cpu_affinity"]:
        raise ValueError("CPU affinity mismatch")
    if (
        baseline["hardware"]["gpu_info"].split(",", 1)[1]
        != candidate["hardware"]["gpu_info"].split(",", 1)[1]
    ):
        raise ValueError("GPU model/memory/driver mismatch")
    if not baseline.get("cache_config") or not candidate.get("capacities"):
        raise ValueError("missing actual cache configuration")
    for key in (
        "model",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "async_scheduling",
        "mamba_ssm_cache_dtype",
    ):
        if key not in baseline.get("config", {}) or baseline["config"][
            key
        ] != candidate.get("config", {}).get(key):
            raise ValueError(f"configuration mismatch: {key}")
    if baseline.get("engine") != "vllm" or baseline.get("commit") != FROZEN_SHA:
        raise ValueError("unrecognized baseline identity")
    for artifact in (baseline, candidate):
        if len(artifact.get("warmups", [])) < 2:
            raise ValueError("missing full warmups")
        summarize(
            artifact["warmups"],
            workload["batch_size"],
            workload["output_len"],
            minimum=2,
        )
        for row in artifact["warmups"] + artifact["runs"]:
            expected = (
                (workload["input_len"] - 1) // 784 * 784 * workload["batch_size"]
                if workload["mode"] == "prefix"
                else 0
            )
            if row.get("prefix_hit_tokens") != expected:
                raise ValueError("unexpected prefix hit count")
        if not artifact.get("measurement_audit", {}).get("steady_state_verified"):
            raise ValueError("measurement compilation/capture audit missing or failed")
    base = summarize(baseline["runs"], workload["batch_size"], workload["output_len"])
    ours = summarize(candidate["runs"], workload["batch_size"], workload["output_len"])
    latency = ours["ttft_s"]["median"] / base["ttft_s"]["median"]
    throughput = ours["output_tps"]["median"] / base["output_tps"]["median"]
    stable = all(m["spread"] <= 0.05 for group in (base, ours) for m in group.values())
    return dict(
        baseline=base,
        candidate=ours,
        ttft_ratio=latency,
        throughput_ratio=throughput,
        stable=stable,
        passed=stable and latency <= 1.1 and throughput >= 0.95,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gpu-blocks", type=int, default=4200)
    parser.add_argument("--mamba-blocks", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    hardware = hardware_identity()
    if args.warmup < 2 or args.repetitions < 5:
        raise ValueError("at least two warmups and five repetitions required")
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    workload = baseline["workload"]
    binary = ROOT / "target/release/oh-my-vllm-zmq-worker"
    command = [
        str(binary),
        "--socket",
        f"/tmp/oh-my-vllm-ttft-{os.getpid()}.ipc",
        "--num-gpu-blocks",
        str(args.num_gpu_blocks),
        "--mamba-blocks",
        str(args.mamba_blocks),
        "--max-model-len",
        str(baseline["config"]["max_model_len"]),
        "--num-speculative-tokens",
        "4" if workload["mode"] == "mtp" else "0",
        "bench",
        "--batch-size",
        str(workload["batch_size"]),
        "--input-len",
        str(workload["input_len"]),
        "--output-len",
        str(workload["output_len"]),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
    ]
    if workload["mode"] == "prefix":
        command.append("--prefix-hit")
    identity = runtime_identity()
    with tempfile.TemporaryDirectory(prefix="oh-my-vllm-ttft-") as temporary:
        snapshot = Path(temporary) / binary.name
        shutil.copy2(binary, snapshot)
        source = source_identity(snapshot)
        command[0] = str(snapshot)
        stdout = run_engine(command, timeout=14400, stderr=subprocess.STDOUT)
    log = args.output.with_suffix(".log")
    log.write_text(stdout)
    candidate = dict(
        schema=1,
        engine="oh-my-vllm",
        workload=workload,
        runtime=identity,
        source=source,
        command=command,
        hardware=hardware,
        config={
            "model": "/data0/shared/Qwen3.8-27B-FP8",
            "max_model_len": baseline["config"]["max_model_len"],
            "max_num_seqs": 32,
            "max_num_batched_tokens": 32768,
            "block_size": 784,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "async_scheduling": False,
            "mamba_ssm_cache_dtype": "bfloat16"
            if workload["mode"] == "mtp"
            else "auto",
        },
        capacities={"fa": args.num_gpu_blocks // 3, "mamba": args.mamba_blocks},
        baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        warmups=parse_rows(stdout, "WARMUP_RESULT "),
        measurement_audit=audit(
            log,
            [
                os.environ.get("FLASHINFER_WORKSPACE_BASE"),
                os.environ.get("TRITON_CACHE_DIR"),
            ],
        ),
        runs=parse_rows(stdout, "BENCH_RESULT "),
        stdout=stdout,
    )
    args.output.write_text(json.dumps(candidate, indent=2) + "\n")
    candidate["comparison"] = compare(baseline, candidate)
    args.output.write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps(candidate["comparison"], indent=2))


if __name__ == "__main__":
    main()
