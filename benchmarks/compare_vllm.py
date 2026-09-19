"""Matched token-ID benchmarks; invoke under scripts/with-gpu.sh.

Each engine lives in a separate child process. A pair shares one GPU, inputs,
configuration, warmup/cache policy, output limits and request-to-completion timer.
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
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def run_engine(command, timeout=3600):
    """Own the entire engine process group, including model-worker children."""
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, start_new_session=True
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
        process.stdout.close()


def source_identity(binary):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--porcelain").decode(),
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }


def prompts(batch: int, length: int) -> list[list[int]]:
    return [
        [(i + request * 997) % 32000 + 1 for i in range(length)]
        for request in range(batch)
    ]


def draft_metrics(llm):
    metrics = llm.get_metrics()
    return {
        key: sum(metric.value for metric in metrics if metric.name == name)
        for key, name in {
            "proposed_draft_tokens": "vllm:spec_decode_num_draft_tokens",
            "accepted_draft_tokens": "vllm:spec_decode_num_accepted_tokens",
        }.items()
    }


def baseline(args):
    import vllm
    from vllm import LLM, SamplingParams

    speculative = args.speculative_tokens if args.mode == "mtp" else 0
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        language_model_only=True,
        block_size=784,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="bfloat16" if speculative else "auto",
        max_model_len=65536,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        num_gpu_blocks_override=args.num_gpu_blocks,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        disable_log_stats=False,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": speculative}
            if speculative
            else None
        ),
    )
    inputs = [
        {"prompt_token_ids": ids}
        for ids in prompts(args.batch_sizes[0], args.input_len)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=args.output_len, ignore_eos=True, detokenize=False
    )
    results = []
    for iteration in range(args.warmup + args.repetitions):
        if not llm.reset_prefix_cache():
            raise RuntimeError("baseline prefix cache reset failed")
        if args.mode == "prefix":
            llm.generate(
                inputs,
                SamplingParams(
                    temperature=0, max_tokens=1, ignore_eos=True, detokenize=False
                ),
                use_tqdm=False,
            )
        before = draft_metrics(llm)
        start = time.perf_counter()
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        after = draft_metrics(llm)
        counts = [len(output.outputs[0].token_ids) for output in outputs]
        if counts != [args.output_len] * len(inputs):
            raise RuntimeError(f"incomplete baseline output: {counts}")
        if iteration >= args.warmup:
            results.append(
                {
                    **{key: after[key] - before[key] for key in before},
                    "output_tokens": sum(counts),
                    "prefix_hit_tokens": sum(
                        output.num_cached_tokens for output in outputs
                    ),
                    "elapsed_s": elapsed,
                    "output_tps": sum(counts) / elapsed,
                }
            )
    print(
        "BASELINE_RESULT " + json.dumps({"version": vllm.__version__, "runs": results}),
        flush=True,
    )
    llm.llm_engine.engine_core.shutdown()


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
    all_results = []
    for batch in args.batch_sizes:
        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--baseline",
            "--model",
            args.model,
            "--num-gpu-blocks",
            str(args.num_gpu_blocks),
            "--batch-sizes",
            str(batch),
            "--input-len",
            str(args.input_len),
            "--output-len",
            str(args.output_len),
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--mode",
            args.mode,
            "--speculative-tokens",
            str(args.speculative_tokens),
        ]
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-bench-") as directory:
            binary_snapshot = Path(directory) / "worker"
            shutil.copy2(args.binary, binary_snapshot)
            if (
                hashlib.sha256(binary_snapshot.read_bytes()).hexdigest()
                != identity["binary_sha256"]
            ):
                raise RuntimeError("binary changed after identity capture")
            baseline_run = run_engine(base_cmd)
            base_rows = parse_rows(baseline_run, "BASELINE_RESULT ")
            if len(base_rows) != 1:
                raise RuntimeError(f"missing baseline result: {baseline_run[-2000:]}")
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
        if len(base) != args.repetitions or any(
            row["prefix_hit_tokens"] != expected_hits for row in base + ours
        ):
            raise RuntimeError("engines did not use the expected matched cache state")
        if args.mode == "mtp" and any(
            row["proposed_draft_tokens"] <= 0 for row in base + ours
        ):
            raise RuntimeError("MTP did not produce draft tokens")
        base_median = statistics.median(row["output_tps"] for row in base)
        ours_median = statistics.median(row["output_tps"] for row in ours)
        result = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "gpu": selected_gpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source": identity,
            "vllm_version": base_rows[0]["version"],
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
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["ordinary", "mtp", "prefix"], default="ordinary"
    )
    parser.add_argument("--speculative-tokens", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 1:
        parser.error("acceptance requires >=3 measurements and >=1 warmup")
    if args.mode == "mtp" and args.speculative_tokens <= 0:
        parser.error("MTP requires positive --speculative-tokens")
    if min(args.batch_sizes) <= 0 or min(args.input_len, args.output_len) <= 0:
        parser.error("batch sizes and token lengths must be positive")
    if args.baseline:
        baseline(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
