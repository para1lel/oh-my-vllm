import argparse
import json
import math
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

from .deepseek_batch_capture import TargetCapture as TargetCapture
from .deepseek_batch_capture import _read_cache_state as _read_cache_state
from .deepseek_batch_capture import cache_state as cache_state
from .deepseek_batch_capture import cpu as cpu
from .deepseek_batch_capture import differences as differences
from .deepseek_batch_capture import fingerprint as fingerprint
from .deepseek_batch_capture import request_state as request_state
from .deepseek_batch_capture import row as row
from .deepseek_batch_capture import target_signature as target_signature
from .deepseek_batch_rank import BatchRank as BatchRank
from .deepseek_batch_rank import rank_main as rank_main
from .deepseek_batch_rank import write_identity as write_identity
from .deepseek_batch_rounds import MultiRound as MultiRound
from .deepseek_batch_rounds import multiround as multiround
from .deepseek_batch_schedule import continuation_rows as continuation_rows
from .deepseek_batch_schedule import execute_message as execute_message
from .deepseek_batch_schedule import execute_scheduled as execute_scheduled
from .deepseek_batch_schedule import execute_single_round as execute_single_round
from .deepseek_batch_schedule import (
    require_multiround_gates as require_multiround_gates,
)
from .deepseek_batch_schedule import (
    require_scheduled_requests as require_scheduled_requests,
)
from .deepseek_batch_schedule import (
    require_single_execution as require_single_execution,
)
from .deepseek_batch_schedule import reset_graphs as reset_graphs
from .deepseek_batch_schedule import update_boundaries as update_boundaries
from .deepseek_batch_schedule import write_round_report as write_round_report


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--strategy", choices=("tp4", "tep4"), default="tp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, choices=(4, 16, 64), default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=(130, 134, 138, 142))
    parser.add_argument(
        "--suite", choices=("single", "multiround", "all"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--batch-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable target groups; disabling uses scheduled individual requests.",
    )
    parser.add_argument(
        "--require-verify-continuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require verify continuation; disabling gives diagnostic results only.",
    )
    parser.add_argument("--require-boundaries", action="store_true")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if not args.require_verify_continuation:
        print(
            "DIAGNOSTIC ONLY: continuation disabled; not valid for acceptance.",
            file=sys.stderr,
        )
    args.graph = args.strategy == "tp4" if args.graph is None else args.graph
    if args.rounds < 6:
        parser.error(
            "multiround checks need at least six steps for slot reuse and fallback"
        )
    args.lengths = tuple(
        args.lengths[index % len(args.lengths)] for index in range(args.batch)
    )
    output_tokens = 32 if args.suite == "single" else max(32, 6 * args.rounds + 8)
    if min(args.lengths) < 1 or max(args.lengths) + output_tokens > args.context:
        parser.error("context must cover every prompt and its full output budget")
    sys.path[:0] = [str(args.tree / "python"), str(args.tree)]
    from oh_my_vllm.models.deepseek_v41.model import context_token_bucket

    configuration = json.loads((args.model / "inference/config.json").read_text())
    cycle = math.lcm(
        *(ratio for ratio in configuration["compress_ratios"] if ratio > 1)
    )
    window = configuration["window_size"]
    for length in (1, 6):
        groups = {
            (
                start % cycle,
                context_token_bucket(start + length),
                min(window, start) if length > 1 else window,
            )
            for start in args.lengths
        }
        if args.suite != "multiround" and len(groups) != 1:
            parser.error(
                "single-round lengths must share a context bucket and compressor branch"
            )
    if args.output.resolve().is_relative_to(args.tree.resolve()):
        parser.error("checkpoint artifacts must be outside the source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 4:
        raise ValueError("four leased GPUs are required")
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="batch-check-", dir=os.environ.get("TMPDIR")
    ) as directory:
        processes = [
            context.Process(
                target=rank_main,
                args=(rank, devices, args, str(Path(directory) / "nccl")),
            )
            for rank in range(4)
        ]
        try:
            for process in processes:
                process.start()
            while any(process.is_alive() for process in processes):
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise RuntimeError("a checkpoint rank failed")
                for process in processes:
                    process.join(timeout=0.1)
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("a checkpoint rank failed")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()


if __name__ == "__main__":
    main()
