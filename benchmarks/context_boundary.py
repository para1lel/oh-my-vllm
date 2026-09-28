"""Run the six real 262144-token boundary rows on one UUID-pinned B200."""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from benchmarks.compare_vllm import parse_rows, run_engine, source_identity
    from benchmarks.measurement import hardware_identity
else:
    # Direct `python benchmarks/context_boundary.py` has benchmarks/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.compare_vllm import parse_rows, run_engine, source_identity
    from benchmarks.measurement import hardware_identity

ROOT = Path(__file__).resolve().parents[1]
INPUT_LEN = 258048
OUTPUT_LEN = 4096
BATCH_SIZES = (1, 2, 4)
MODES = ("ordinary", "mtp4")


@contextmanager
def _reap_on_termination():
    """Defer Ctrl-C/SIGTERM until run_engine owns its spawned process group."""
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in signals}
    requested = False

    def interrupt(signum, frame):
        nonlocal requested
        requested = True

    for signum in signals:
        signal.signal(signum, interrupt)
    try:
        yield lambda: requested
        if requested:
            raise InterruptedError("boundary collector was terminated")
    finally:
        for signum in signals:
            signal.signal(signum, previous[signum])


def validate_row(log: str, batch_size: int, mode: str, gpu_total_bytes: int) -> dict:
    """Fail closed on missing bench output, memory accounting, OOM, or MTP drafts."""
    if mode not in MODES or batch_size not in BATCH_SIZES:
        raise ValueError("unknown context-boundary row")
    if re.search(r"OutOfMemoryError|out of memory|out-of-memory|\bOOM\b", log, re.I):
        raise ValueError("boundary run reported GPU OOM")
    rows = parse_rows(log, "BENCH_RESULT ")
    if len(rows) != 1:
        raise ValueError(f"expected one BENCH_RESULT, found {len(rows)}")
    row = rows[0]
    expected = {
        "batch_size": batch_size,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "output_tokens": batch_size * OUTPUT_LEN,
        "preemptions": 0,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"boundary {key}: expected {value}, got {row.get(key)}")
    proposed = row.get("proposed_draft_tokens")
    if (
        not isinstance(proposed, int)
        or (mode == "mtp4" and proposed <= 0)
        or (mode == "ordinary" and proposed != 0)
    ):
        raise ValueError(f"boundary {mode} proposed_draft_tokens={proposed}")
    for key in ("elapsed_s", "output_tps"):
        if (
            not isinstance(row.get(key), (int, float))
            or not math.isfinite(row[key])
            or row[key] <= 0
        ):
            raise ValueError(f"invalid boundary {key}")
    for key in ("steps", "accepted_draft_tokens"):
        minimum = 1 if key == "steps" else 0
        if not isinstance(row.get(key), int) or row[key] < minimum:
            raise ValueError(f"invalid boundary {key}")
    if mode == "ordinary" and row["accepted_draft_tokens"] != 0:
        raise ValueError("ordinary boundary accepted MTP drafts")
    if row["accepted_draft_tokens"] > proposed:
        raise ValueError("boundary accepted more MTP drafts than proposed")
    memories = []
    for line in log.splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("message") == "GPU memory high water"
        ):
            memories.append(record)
    if len(memories) != 1:
        raise ValueError(
            f"expected one worker GPU memory record, found {len(memories)}"
        )
    memory = memories[0]
    allocated, reserved = (
        memory.get("max_allocated_bytes"),
        memory.get("max_reserved_bytes"),
    )
    if not (
        isinstance(allocated, int)
        and isinstance(reserved, int)
        and 0 < allocated <= reserved <= gpu_total_bytes
    ):
        raise ValueError("invalid worker peak GPU memory")
    return {
        "mode": mode,
        "batch_size": batch_size,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "output_tokens": row["output_tokens"],
        "elapsed_s": row["elapsed_s"],
        "output_tps": row["output_tps"],
        "steps": row["steps"],
        "preemptions": row["preemptions"],
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": row["accepted_draft_tokens"],
        "max_allocated_bytes": allocated,
        "max_reserved_bytes": reserved,
        "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
    }


def _gpu_total_bytes(hardware: dict) -> int:
    text = hardware["gpu_info"].split(",")[2].strip()
    if not text.endswith(" MiB"):
        raise ValueError(f"unexpected GPU memory unit: {text}")
    return int(text.removesuffix(" MiB")) * 1024 * 1024


def run_row(
    binary: Path,
    batch_size: int,
    mode: str,
    *,
    gpu_total_bytes: int | None = None,
    raw_dir: Path | None = None,
    timeout: int = 1800,
) -> dict:
    """Inherit the outer GPU lock and let run_engine own/reap the full process group."""
    hardware = hardware_identity()
    total = (
        gpu_total_bytes if gpu_total_bytes is not None else _gpu_total_bytes(hardware)
    )
    with tempfile.TemporaryDirectory(prefix="oh-my-vllm-boundary-") as directory:
        socket = Path(directory) / "worker.ipc"
        log_path = (
            raw_dir / f"{mode}-{batch_size}.log"
            if raw_dir is not None
            else Path(directory) / "bench.log"
        )
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--socket",
            str(socket),
            "--num-gpu-blocks",
            "4200",
            "--mamba-blocks",
            "128",
            "--max-model-len",
            "262144",
            "--num-speculative-tokens",
            "4" if mode == "mtp4" else "0",
            "bench",
            "--batch-size",
            str(batch_size),
            "--input-len",
            str(INPUT_LEN),
            "--output-len",
            str(OUTPUT_LEN),
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
        with _reap_on_termination() as cancelled:
            log = run_engine(
                command,
                timeout=timeout,
                stderr=subprocess.STDOUT,
                log_path=log_path,
                cancelled=cancelled,
            )
        result = validate_row(log, batch_size, mode, total)
        result["command"] = command
        result["gpu_uuid"] = hardware["gpu"]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_dir.resolve().is_relative_to(ROOT):
        raise ValueError("raw boundary logs must stay outside the repository")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"build the boundary bench binary first: {binary}")
    hardware = hardware_identity()
    source = source_identity(binary)
    rows = []
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            row = run_row(binary, batch_size, mode, raw_dir=args.raw_dir)
            rows.append(row)
            print(
                mode,
                batch_size,
                row["output_tokens"],
                row["max_reserved_bytes"],
                flush=True,
            )
    source_end = source_identity(binary)
    same_source = source == source_end
    same_gpu = all(row["gpu_uuid"] == hardware["gpu"] for row in rows)
    artifact = {
        "source": source,
        "hardware": hardware,
        "run_id": os.environ.get("OH_MY_VLLM_RUN_ID"),
        "protocol": (
            "ordinary/MTP4 x batch 1/2/4; 258048 input + 4096 output; "
            "zero preemption; MTP drafts; worker peak memory"
        ),
        "rows": rows,
        "source_end_matches_start": same_source,
        "gpu_uuid_matches_start": same_gpu,
        "passed": (
            len(rows) == 6
            and same_source
            and source_end["git_status"] == ""
            and same_gpu
        ),
    }
    if not same_source:
        artifact["source_end"] = source_end
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    if not artifact["passed"]:
        raise ValueError("boundary rows require a clean source for acceptance")


if __name__ == "__main__":
    main()
