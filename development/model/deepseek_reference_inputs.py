import json
from pathlib import Path

from .deepseek_reference_contract import digest

ROOT = Path(__file__).resolve().parents[2]


BUNDLE_SHA256 = "7f82d0e2dcca8243f7dcfea71c96b204f78e1ddbf720a8843a0e48baea82faed"


def checkpoint_stats(model, names):
    return {
        name: dict(
            bytes=(model / name).stat().st_size,
            mtime_ns=(model / name).stat().st_mtime_ns,
            inode=(model / name).stat().st_ino,
            ctime_ns=(model / name).stat().st_ctime_ns,
            device=(model / name).stat().st_dev,
        )
        for name in sorted(names)
    }


def bundle_members(root):
    path = root / "bundle-manifest.json"
    if digest(path) != BUNDLE_SHA256:
        raise ValueError("reviewed recovery bundle SHA256 differs")
    return {
        record["path"]: record["sha256"]
        for record in json.loads(path.read_text())["files"]
    }


def checkpoint_inputs(args):
    from development.serving.deepseek_benchmark import checkpoint_identity

    members = bundle_members(args.reference)
    if (
        digest(args.checkpoint_digests)
        != members["provenance/current-original-checkpoint-sha256.json"]
    ):
        raise ValueError(
            "checkpoint digest record is not the reviewed recovery bundle member"
        )
    manifest = json.loads(args.checkpoint_digests.read_text())
    expected = {name: value["sha256"] for name, value in manifest["files"].items()}
    before = checkpoint_stats(
        args.model, (*expected, "config.json", "model.safetensors.index.json")
    )
    observed = checkpoint_identity(args.model)
    after = checkpoint_stats(args.model, before)
    if after != before:
        raise ValueError("checkpoint files changed during the SHA256 check")
    if {
        name: value
        for name, value in observed.items()
        if name not in ("config", "index")
    } != expected:
        raise ValueError(
            "candidate checkpoint bytes differ from the reference conversion inputs"
        )
    return observed, before


def checker_identity():
    names = (
        "check_deepseek_reference.py",
        "deepseek_reference_contract.py",
        "deepseek_reference_capture.py",
        "deepseek_reference_rank.py",
        "deepseek_reference_report.py",
        "deepseek_reference_inputs.py",
        "check_deepseek_batch.py",
        "deepseek_batch_capture.py",
        "deepseek_batch_rank.py",
        "deepseek_batch_rounds.py",
        "deepseek_batch_schedule.py",
        "check_deepseek_dep4.py",
        "deepseek_dep4_evidence.py",
        "deepseek_dep4_rank.py",
    )
    return {name: digest(Path(__file__).with_name(name)) for name in names}
