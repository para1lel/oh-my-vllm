"""Compare DEP4 exchange paths with complete target and restoration endpoints."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
import uuid
from multiprocessing.connection import wait
from pathlib import Path

from .deepseek_dep4_evidence import ExchangeEvidence as ExchangeEvidence
from .deepseek_dep4_evidence import _rank_worker as _rank_worker
from .deepseek_dep4_evidence import _write_json as _write_json
from .deepseek_dep4_evidence import checker_identity
from .deepseek_dep4_evidence import request_owners as request_owners
from .deepseek_dep4_evidence import tensor_difference as tensor_difference
from .deepseek_dep4_rank import Dep4Rank as Dep4Rank
from .deepseek_dep4_rank import initialize_rank as initialize_rank
from .deepseek_dep4_rank import run as run


class RankGroupFailure(RuntimeError):
    pass


def _record_group_failure(args, processes, manifest, error):
    shutdown_started = time.monotonic()
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
    stopped = time.monotonic()
    failure_times = []
    for rank in range(4):
        path = args.output / f"proof-rank{rank}.json"
        record = json.loads(path.read_text())
        if record["failure_time_monotonic"] is not None:
            failure_times.append(record["failure_time_monotonic"])
        record["outcome"] = "failed"
        record["group_error"] = dict(type=type(error).__name__, message=str(error))
        _write_json(path, record)
    manifest.update(
        outcome="failed",
        error=dict(type=type(error).__name__, message=str(error)),
        shutdown_seconds=stopped - shutdown_started,
        failure_to_group_exit_seconds=stopped - min(failure_times)
        if failure_times
        else None,
        child_exitcodes=[process.exitcode for process in processes],
        remaining_child_pids=[
            process.pid
            for process in processes
            if process.pid is not None and process.is_alive()
        ],
    )
    _write_json(args.output / "run.json", manifest)


def supervise(args, *, operation=None):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("DEP4 check output must be a new or empty directory")
    run_id = uuid.uuid4().hex
    manifest = dict(
        run_id=run_id,
        mode=args.mode,
        outcome="initializing",
        checker_modules=checker_identity(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lengths=getattr(args, "lengths", None),
        context=getattr(args, "context", None),
        owner_ranks=getattr(args, "owner_ranks", None),
        execution="eager",
        batch_decode_enabled=False,
        evidence="cross-path-consistency-not-frozen-reference",
    )
    _write_json(args.output / "run.json", manifest)
    for rank in range(4):
        ExchangeEvidence(args.output, rank, mode=args.mode, run_id=run_id).write()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_rank_worker, args=(rank, args, run_id, operation or run)
        )
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        pending = {
            process.sentinel: (rank, process) for rank, process in enumerate(processes)
        }
        deadline = time.monotonic() + args.timeout_seconds
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DEP4 check exceeded its process deadline")
            for sentinel in wait(pending, timeout=min(remaining, 0.2)):
                rank, process = pending.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise RankGroupFailure(
                        f"rank {rank} exited with code {process.exitcode}"
                    )
    except BaseException as error:
        _record_group_failure(args, processes, manifest, error)
        raise
    else:
        manifest["outcome"] = "completed"
        _write_json(args.output / "run.json", manifest)
    finally:
        for process in processes:
            if process.pid is not None and not process.is_alive():
                process.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--mode", choices=("tep4", "legacy", "fixed", "shadow"), required=True
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[600, 300, 17])
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    try:
        owners = request_owners(args.lengths)
    except ValueError as error:
        parser.error(str(error))
    args.owner_ranks = [0] if args.mode == "tep4" else owners
    if min(args.lengths) < 1 or max(133, max(args.lengths) + 3) > args.context:
        parser.error(
            "context must cover the prompt, continuation, and verification endpoints"
        )
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    if "RANK" in os.environ:
        parser.error("launch this driver once through with-gpu.sh, without torchrun")
    args.output = args.output.resolve()
    supervise(args)


if __name__ == "__main__":
    main()
