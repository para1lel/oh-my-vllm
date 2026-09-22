"""Isolated, explicitly authorized vLLM reference collector; never a runtime dependency.

Run only with the separately maintained baseline Python, via with-gpu.sh. The
independent comparison process consumes JSON; it must not import this module.
"""

import argparse
import dataclasses
import datetime
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

FROZEN_SHA = "e9f169d16b9408bb9ae44f75072b91a5521d733c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="/data0/shared/Qwen3.8-27B-FP8")
    parser.add_argument("--checkout", default="/data0/shared/dongwu.chen/vllm")
    parser.add_argument("--mode", choices=["ordinary", "mtp", "prefix"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compare_vllm import run_engine
    from measurement import audit, hardware_identity

    hardware = hardware_identity()
    if args.warmup < 2 or args.repetitions < 5:
        raise ValueError(
            "formal collection requires at least two warmups and five runs"
        )
    if not args.worker:
        os.environ.setdefault(
            "VLLM_CACHE_ROOT",
            "/data0/shared/dongwu.chen/.cache/oh-my-vllm/baseline-main",
        )
        os.environ.setdefault(
            "FLASHINFER_WORKSPACE_BASE", os.environ["VLLM_CACHE_ROOT"] + "/flashinfer"
        )
        os.environ.setdefault(
            "TRITON_CACHE_DIR", os.environ["VLLM_CACHE_ROOT"] + "/triton"
        )
        log = args.output.with_suffix(".log")
        run_engine(
            [sys.executable, __file__, *sys.argv[1:], "--worker"],
            timeout=14400,
            stderr=subprocess.STDOUT,
            log_path=log,
        )
        artifact = json.loads(args.output.read_text())
        artifact["measurement_audit"] = audit(
            log,
            [
                os.environ["VLLM_CACHE_ROOT"],
                os.environ["FLASHINFER_WORKSPACE_BASE"],
                os.environ["TRITON_CACHE_DIR"],
            ],
        )
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        return

    sha = subprocess.check_output(
        ["git", "-C", args.checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != FROZEN_SHA:
        raise RuntimeError("baseline checkout differs from frozen official main")
    if subprocess.check_output(["git", "-C", args.checkout, "diff", "HEAD"]):
        raise RuntimeError("baseline checkout has source modifications")
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError("workload exceeds context")
    import vllm

    if Path(vllm.__file__).resolve().parent != Path(args.checkout).resolve() / "vllm":
        raise RuntimeError("imported vLLM is not the frozen checkout")
    from vllm import EngineArgs, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.metrics.loggers import StatLoggerBase

    class Recorder(StatLoggerBase):
        finished: ClassVar[list[dict]] = []

        def __init__(self, vllm_config, engine_index=0):
            pass

        def log_engine_initialized(self):
            pass

        def record(
            self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
        ):
            if iteration_stats:
                self.finished.extend(
                    dataclasses.asdict(r) for r in iteration_stats.finished_requests
                )

    spec = (
        {"method": "mtp", "num_speculative_tokens": 4} if args.mode == "mtp" else None
    )
    config = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        block_size=784,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        language_model_only=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="bfloat16" if spec else "auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=spec,
        disable_log_stats=False,
    )
    engine = LLMEngine.from_engine_args(
        EngineArgs(**config), stat_loggers=[Recorder], enable_multiprocessing=True
    )
    prompts = [
        [(i + request * 997) % 32000 + 1 for i in range(args.input_len)]
        for request in range(args.batch_size)
    ]
    counter = 0

    def batch(length):
        nonlocal counter
        Recorder.finished.clear()
        first, counts, request_ids = {}, {}, []
        params = SamplingParams(
            temperature=0, max_tokens=length, ignore_eos=True, detokenize=False
        )
        start = time.monotonic()
        for prompt in prompts:
            # Outputs use the external ID, not add_request's randomized internal ID.
            rid = str(counter)
            engine.add_request(rid, {"prompt_token_ids": prompt}, params)
            counter += 1
            request_ids.append(rid)
        while engine.has_unfinished_requests():
            for result in engine.step():
                if result.outputs and result.outputs[0].token_ids:
                    first.setdefault(result.request_id, time.monotonic() - start)
                    counts[result.request_id] = len(result.outputs[0].token_ids)
        elapsed = time.monotonic() - start
        if len(counts) != args.batch_size or any(n != length for n in counts.values()):
            raise RuntimeError(f"incomplete output: {counts}")
        if len(Recorder.finished) != args.batch_size:
            raise RuntimeError("missing request completion statistics")
        return dict(
            elapsed_s=elapsed,
            output_tokens=sum(counts.values()),
            output_tps=sum(counts.values()) / elapsed,
            ttft_s=[first[r] for r in request_ids],
            preemptions=sum(r["num_preemptions"] for r in Recorder.finished),
            prefix_hit_tokens=sum(r["num_cached_tokens"] for r in Recorder.finished),
            finished=list(Recorder.finished),
        )

    runs, warmups = [], []
    try:
        for iteration in range(args.warmup + args.repetitions):
            stamp = datetime.datetime.now(datetime.UTC).isoformat()
            measured = str(iteration >= args.warmup).lower()
            print(
                f"{stamp} BENCH_PHASE measured={measured}", file=sys.stderr, flush=True
            )
            if not engine.reset_prefix_cache():
                raise RuntimeError("prefix cache reset failed")
            if args.mode == "prefix":
                batch(1)
            row = batch(args.output_len)
            (warmups if iteration < args.warmup else runs).append(row)
            print("BASELINE_RESULT " + json.dumps(row, default=str), flush=True)
        artifact = dict(
            schema=1,
            engine="vllm",
            commit=sha,
            python=sys.executable,
            packages={
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            hardware=hardware,
            imported_vllm=str(Path(vllm.__file__).resolve()),
            config=config,
            cache_config=dataclasses.asdict(engine.vllm_config.cache_config),
            workload={
                k: getattr(args, k)
                for k in ("mode", "batch_size", "input_len", "output_len")
            },
            warmups=warmups,
            runs=runs,
        )
        args.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    finally:
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
