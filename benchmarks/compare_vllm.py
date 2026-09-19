"""Benchmark comparison: oh-my-vllm vs vLLM EngineCore.

Runs both engines on the same workloads and prints a side-by-side throughput
table.  Each workload is run with vLLM's offline LLM API first, then with the
oh-my-vllm binary.

Usage::

    python benchmarks/compare_vllm.py \
        --model /data0/shared/Qwen3.8-27B-FP8 \
        --num-gpu-blocks 4096 \
        --batch-sizes 1 2 4 \
        --input-len 32768 \
        --output-len 4096

Requirements:
    - vLLM must be importable (run inside the vllm conda env, or ensure vllm is
      on the Python path).
    - The oh-my-vllm-zmq-worker binary must be on PATH or at
      ../target/release/oh-my-vllm-zmq-worker relative to this script.
    - A single GPU with enough VRAM for the model at --num-gpu-blocks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# vLLM baseline
# ---------------------------------------------------------------------------


def run_vllm(
    model: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_gpu_blocks: int,
) -> float:
    """Run vLLM offline and return output tokens per second."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not importable — skipping vLLM baseline")
        return float("nan")

    llm = LLM(
        model=model,
        num_gpu_blocks_override=num_gpu_blocks,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
    )
    prompts = [" ".join(["the"] * input_len)] * batch_size
    sampling_params = SamplingParams(max_tokens=output_len, temperature=0.0)

    # Warmup.
    llm.generate(prompts[:1], sampling_params)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_output / elapsed
    del llm
    return tps


# ---------------------------------------------------------------------------
# oh-my-vllm baseline
# ---------------------------------------------------------------------------

# Socket path reused across calls within one process.
_SOCKET_PATH = "/tmp/oh-my-vllm-bench.ipc"


def _find_binary() -> Path:
    """Locate the oh-my-vllm-zmq-worker binary."""
    candidates = [
        Path(sys.argv[0]).parent.parent
        / "target"
        / "release"
        / "oh-my-vllm-zmq-worker",
        Path("target/release/oh-my-vllm-zmq-worker"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Fall back to PATH.
    import shutil

    found = shutil.which("oh-my-vllm-zmq-worker")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "oh-my-vllm-zmq-worker binary not found. Build with: "
        "cargo build --release --bin oh-my-vllm-zmq-worker"
    )


def run_oh_my_vllm(
    model: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_gpu_blocks: int,
) -> float:
    """Run oh-my-vllm bench subcommand and parse its output tokens per second."""
    binary = _find_binary()
    cmd = [
        str(binary),
        "--model",
        model,
        "--socket",
        _SOCKET_PATH,
        "--num-gpu-blocks",
        str(num_gpu_blocks),
        "--block-size",
        "784",
        "bench",
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--warmup",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  oh-my-vllm failed:\n{result.stderr[-2000:]}")
        return float("nan")

    # Parse "throughput    : <X> output tok/s  |  <Y> total tok/s"
    for line in result.stdout.splitlines():
        if "output tok/s" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "output":
                    return float(parts[i - 1])
    print(f"  could not parse throughput from output:\n{result.stdout}")
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare oh-my-vllm vs vLLM throughput"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=512)
    args = parser.parse_args()

    cols = f"{'batch':>6}  {'input':>6}  {'output':>6}"
    cols += f"  {'vLLM (tok/s)':>14}  {'ours (tok/s)':>14}  {'ratio':>7}  {'±5%':>5}"
    header = cols
    print(header)
    print("-" * len(header))

    for bs in args.batch_sizes:
        print(f"  bs={bs} prefilling…", flush=True)
        vllm_tps = run_vllm(
            args.model, bs, args.input_len, args.output_len, args.num_gpu_blocks
        )
        print(f"  bs={bs} running oh-my-vllm…", flush=True)
        ours_tps = run_oh_my_vllm(
            args.model, bs, args.input_len, args.output_len, args.num_gpu_blocks
        )

        if vllm_tps != vllm_tps or ours_tps != ours_tps:  # nan check
            ratio_str = "  N/A"
            ok_str = "  N/A"
        else:
            ratio = ours_tps / vllm_tps
            ratio_str = f"{ratio:>7.3f}"
            ok_str = "  ✓" if abs(ratio - 1.0) <= 0.05 else "  ✗"

        vllm_str = f"{vllm_tps:>14.1f}" if vllm_tps == vllm_tps else f"{'N/A':>14}"
        ours_str = f"{ours_tps:>14.1f}" if ours_tps == ours_tps else f"{'N/A':>14}"
        print(
            f"{bs:>6}  {args.input_len:>6}  {args.output_len:>6}  "
            f"{vllm_str}  {ours_str}  {ratio_str}  {ok_str}"
        )


if __name__ == "__main__":
    main()
