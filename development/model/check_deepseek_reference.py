"""Collect native Worker endpoints against the immutable independent MP4 reference."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from .check_deepseek_dep4 import _write_json, supervise
from .deepseek_reference_capture import CandidateCapture as CandidateCapture
from .deepseek_reference_contract import CASE_NAMES, digest, load_reference
from .deepseek_reference_inputs import BUNDLE_SHA256 as BUNDLE_SHA256
from .deepseek_reference_inputs import ROOT as ROOT
from .deepseek_reference_inputs import bundle_members as bundle_members
from .deepseek_reference_inputs import checkpoint_inputs as checkpoint_inputs
from .deepseek_reference_inputs import checkpoint_stats as checkpoint_stats
from .deepseek_reference_rank import _candidate_identity as _candidate_identity
from .deepseek_reference_rank import collect_rank as collect_rank
from .deepseek_reference_rank import rank_entry as rank_entry
from .deepseek_reference_report import _rank_summaries as _rank_summaries
from .deepseek_reference_report import coverage_matches as coverage_matches
from .deepseek_reference_report import summarize as summarize


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", type=Path)
    result.add_argument("--strategy", choices=("tp4", "tep4", "dep4"), default="tp4")
    result.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=CASE_NAMES)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--checkpoint-digests", type=Path)
    result.add_argument("--evidence", type=Path, action="append", default=[])
    result.add_argument("--timeout-seconds", type=float, default=1800)
    return result


def _write_validation(args, reference, manifest, evidence):
    args.output.mkdir(parents=True)
    _write_json(
        args.output / "validation.json",
        dict(
            reference=reference,
            cases=list(CASE_NAMES),
            tensors=len(manifest["tensors"]),
            evidence=evidence,
            candidate_executed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                reference=reference,
                tensors=len(manifest["tensors"]),
                candidate_executed=False,
            )
        )
    )
    return


def main():
    arguments = parser()
    args = arguments.parse_args()
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if (
        args.output.is_relative_to(ROOT)
        or args.reference.is_relative_to(args.output)
        or args.output.is_relative_to(args.reference)
    ):
        arguments.error(
            "reports must be external and separate from the read-only reference"
        )
    if args.output.exists():
        arguments.error("output must be a new external directory")
    if len(set(args.cases)) != len(args.cases) or args.timeout_seconds <= 0:
        arguments.error("cases must be distinct and timeout-seconds must be positive")
    args.cases = tuple(name for name in CASE_NAMES if name in args.cases)
    manifest, _, reference = load_reference(args.reference)
    members = bundle_members(args.reference)
    evidence = []
    for path in (args.reference / "bundle-manifest.json", *args.evidence):
        sha256 = digest(path)
        member = next(
            (name for name, value in members.items() if value == sha256), None
        )
        if sha256 != BUNDLE_SHA256 and member is None:
            raise ValueError("evidence is not a reviewed recovery bundle member")
        evidence.append(dict(path=str(path), sha256=sha256, bundle_member=member))
    if args.validate_only:
        _write_validation(args, reference, manifest, evidence)
        return
    if args.model is None or args.checkpoint_digests is None:
        arguments.error(
            "candidate collection requires --model and --checkpoint-digests"
        )
    args.model = args.model.resolve()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    selected = os.environ.get("OH_MY_VLLM_GPU_UUIDS", "").split(",")
    if (
        os.environ.get("OH_MY_VLLM_GPU_COUNT") != "4"
        or devices != selected
        or len(set(devices)) != 4
        or any(not value.startswith("GPU-") for value in devices)
    ):
        arguments.error("use with-gpu.sh with four explicit ordered GPU UUIDs")
    if os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "1") != "1":
        arguments.error("native sparse attention must be enabled")
    if "RANK" in os.environ:
        arguments.error("launch this entry once, without torchrun")
    observed, stats = checkpoint_inputs(args)
    args.devices = devices
    args.mode = "legacy" if args.strategy == "dep4" else args.strategy
    try:
        with patch("development.model.check_deepseek_dep4._rank_worker", rank_entry):
            supervise(args, operation=collect_rank)
    finally:
        if args.output.exists():
            _write_json(
                args.output / "candidate-inputs.json",
                dict(
                    reference=reference,
                    checkpoint=observed,
                    checkpoint_stats=stats,
                    checkpoint_hash_method="full SHA256 of all current shards",
                    checkpoint_manifest_sha256=digest(args.checkpoint_digests),
                    evidence=evidence,
                    arguments=json.loads(json.dumps(vars(args), default=str)),
                ),
            )
    if checkpoint_stats(args.model, stats) != stats:
        raise ValueError("checkpoint files changed during candidate execution")
    summarize(args, reference)


if __name__ == "__main__":
    main()
