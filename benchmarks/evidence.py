"""Checkpoint, loaded-model, service-log, and artifact evidence helpers."""

import hashlib
import json
import re
from pathlib import Path

from benchmarks.common import source_identity
from scripts.export_evidence import derive

SPECULATIVE_MODES = ("mtp", "dspark")


def checkpoint_identity(directory: str | Path) -> dict:
    """Hash checkpoint content, with relative names and configuration bytes."""
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("checkpoint must be a directory with config.json")
    files = {}
    tree = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        encoded = relative.encode()
        tree.update(len(encoded).to_bytes(8, "little"))
        tree.update(encoded)
        tree.update(digest.digest())
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("checkpoint has no safetensors weights")
    return {
        "path": str(directory),
        "files": files,
        "tree_sha256": tree.hexdigest(),
        "config": json.loads((directory / "config.json").read_text()),
    }


def read_service_evidence(
    path: Path, offset: int, mode: str, *, request_ids=None, expected=None
) -> list[dict]:
    """Match new completed requests to the configured speculative method."""
    if mode not in SPECULATIVE_MODES:
        raise ValueError("service evidence requires mtp or dspark mode")
    with path.open("rb") as source:
        if source.seek(0, 2) < offset:
            raise ValueError("server log was truncated during acceptance")
        source.seek(0)
        full = source.read().decode("utf-8")
        source.seek(offset)
        tail = source.read().decode("utf-8")
    configured = [
        re.search(r'\bspeculative_mode="?(mtp|dspark|none)"?\b', line)
        for line in full.splitlines()
        if "BENCH_CONFIG" in line
    ]
    if len(configured) != 1 or configured[0] is None or configured[0][1] != mode:
        raise ValueError("server speculative mode identity is missing or differs")
    rows = []
    seen = set()
    for line in tail.splitlines():
        if "request completed" not in line:
            continue
        fields = {}
        for key in ("request_id", "verified_draft_tokens", "accepted_draft_tokens"):
            match = re.search(rf"\b{key}=(\d+)\b", line)
            if match is None:
                raise ValueError(f"completed request has no {key}")
            fields[key] = int(match[1])
        rid = fields["request_id"]
        if rid in seen:
            raise ValueError("duplicate completed request identity")
        seen.add(rid)
        if fields["accepted_draft_tokens"] > fields["verified_draft_tokens"]:
            raise ValueError("accepted drafts exceed actual verified drafts")
        rows.append(fields)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} completed requests, found {len(rows)}")
    if request_ids is not None and seen != set(request_ids):
        raise ValueError("completed requests do not match client response identities")
    if not rows or sum(row["verified_draft_tokens"] for row in rows) <= 0:
        raise ValueError("service has no actual verified drafts")
    return rows


def write_artifact(path: Path, artifact: dict, portable_output: Path | None = None):
    """Keep a complete raw artifact and an explicitly derived portable envelope."""
    if portable_output is not None and portable_output.resolve() == path.resolve():
        raise ValueError("portable output must not replace raw evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(artifact, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    if portable_output is not None:
        portable_output.parent.mkdir(parents=True, exist_ok=True)
        portable_output.write_text(json.dumps(derive(raw, path.name), indent=2) + "\n")


def runtime_source_identity(binary: Path) -> dict:
    result = source_identity(binary)
    root = Path(__file__).resolve().parents[1]
    result["runtime_sources_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("crates", "python", "benchmarks", "scripts")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file() and path.suffix in (".rs", ".py", ".sh", ".cu", ".cuh")
    }
    return result


def observed_configuration(log: str) -> dict:
    lines = [line for line in log.splitlines() if "BENCH_CONFIG" in line]
    if len(lines) != 1:
        raise ValueError("expected one observed BENCH_CONFIG record")
    return {
        key: value.strip('"')
        for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', lines[0])
    }


def loaded_draft_provenance(log: str, expected: dict) -> dict:
    """Match the worker's loaded tensors to the checkpoint's actual source bytes."""
    rows = []
    prefix = "DSPARK_CHECKPOINT_PROVENANCE "
    for line in log.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            message = line
        else:
            message = record.get("message", "") if isinstance(record, dict) else ""
        if prefix in message:
            rows.append(json.loads(message.split(prefix, 1)[1]))
    if len(rows) != 1:
        raise ValueError("expected one actually loaded DSpark checkpoint provenance")
    row = rows[0]
    if Path(row.get("path", "")).resolve() != Path(expected["path"]).resolve():
        raise ValueError("loaded DSpark checkpoint path differs")
    config_hash = expected["files"].get("config.json", {}).get("sha256")
    weights = row.get("weights_sha256", {})
    if (
        row.get("config_sha256") != config_hash
        or not weights
        or any(
            name not in expected["files"] or digest != expected["files"][name]["sha256"]
            for name, digest in weights.items()
        )
    ):
        raise ValueError("loaded DSpark configuration or weight hashes differ")
    return row
