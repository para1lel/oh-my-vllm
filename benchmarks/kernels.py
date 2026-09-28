"""Formal static-shape CUDA versus frozen TileLang performance collection."""

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_vllm import assert_gpu_exclusive  # noqa: E402
from measurement import hardware_identity  # noqa: E402

from development.kernels.cases import UNUSED, cases  # noqa: E402
from development.kernels.comparison import compare  # noqa: E402
from development.kernels.reference import source_hashes, verify_reference  # noqa: E402
from development.kernels.variant import (  # noqa: E402
    REQUIRED_FAST,
    verify_fast_dispatch,
)

sources = source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operations", nargs="+")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    matrix = cases()
    selected = [
        c
        for c in matrix
        if not args.operations or c["configuration"]["operation"] in args.operations
    ]
    if not selected:
        parser.error("no matching cases")
    if args.list:
        print(json.dumps(dict(cases=selected, unused=UNUSED), indent=2))
        return
    if args.output is None:
        parser.error("--output required for measurements")
    if os.environ.get("OH_MY_VLLM_KERNEL_BACKEND") != "cuda":
        parser.error("set OH_MY_VLLM_KERNEL_BACKEND=cuda before starting Python")
    import torch
    from oh_my_vllm.kernels.cuda_backend import provenance, variant_launch_counts

    from development.kernels.fixtures import fixture
    from development.kernels.observations import analyze
    from development.kernels.output_verification import verify
    from development.kernels.timing import measure

    reference_manifest = verify_reference()
    hardware = hardware_identity()
    initial_sources = sources()
    failures = []
    stop = threading.Event()

    def watch():
        while not stop.wait(0.5):
            try:
                assert_gpu_exclusive(hardware["gpu"], os.getpgrp())
            except Exception as error:
                failures.append(str(error))
                return

    assert_gpu_exclusive(hardware["gpu"], os.getpgrp())
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    report = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_status=status,
        sources_sha256=initial_sources,
        hardware=hardware,
        reference=reference_manifest,
        protocol=(
            "Complete static maximum-shape operations; three independent warm rounds, "
            "twenty alternating-order pairs/round,100 graph repetitions/sample. "
            "No profiler during timing. Fixtures use immutable sources and isolated "
            "repeatable destinations. Before timing, compare both backends' "
            "returns and written cache/state slots at existing tolerances."
        ),
        coverage=dict(
            complete=len(selected) == len(matrix),
            required=[c["id"] for c in matrix],
            selected=[c["id"] for c in selected],
        ),
        diagnostic=bool(status),
        rows=[],
        estimates={},
        passed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for case in selected:
            if failures:
                raise RuntimeError(failures[0])
            reference, candidate, witnesses = fixture(
                case["configuration"], observe=True
            )
            operation = case["configuration"]["operation"]
            variant_before = (
                variant_launch_counts() if operation in REQUIRED_FAST else None
            )
            output_verification = verify(operation, reference, candidate, witnesses)
            variant_dispatch = (
                verify_fast_dispatch(operation, variant_before, variant_launch_counts())
                if variant_before is not None
                else None
            )
            raw = measure(reference, candidate)
            report["cuda_build_provenance"] = provenance(
                require_loaded=True, require_compiler=True
            )
            decision = compare(raw)
            if failures or sources() != initial_sources:
                raise RuntimeError(
                    failures[0] if failures else "source changed during measurement"
                )
            report["rows"].append(
                dict(
                    **case,
                    output_verification=output_verification,
                    variant_dispatch=variant_dispatch,
                    raw_pairs_ms=raw,
                    comparison=decision,
                )
            )
            print(
                case["id"],
                "PASS" if decision["passed"] else "FAIL",
                decision["paired_mean_gain_ms"],
                flush=True,
            )
            del reference, candidate
            torch.cuda.empty_cache()
            save()
        assert_gpu_exclusive(hardware["gpu"], os.getpgrp())
        stop.set()
        thread.join()
        if failures:
            raise RuntimeError(failures[0])
        for operation in sorted({r["configuration"]["operation"] for r in selected}):
            symbol = {"convolution": "convolution", "recurrent": "recurrent"}.get(
                operation, operation
            )
            report["estimates"][operation] = analyze(symbol)
        if sources() != initial_sources:
            raise RuntimeError("source changed during analysis")
        report["selected_passed"] = all(
            r["comparison"]["passed"] for r in report["rows"]
        )
        report["passed"] = (
            not report["diagnostic"]
            and report["coverage"]["complete"]
            and report["selected_passed"]
        )
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        stop.set()
        thread.join()
        report["interference"] = failures
        save()
    if not all(r["comparison"]["passed"] for r in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
