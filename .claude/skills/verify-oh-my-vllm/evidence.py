import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def timestamp():
    return datetime.now(UTC).isoformat()


def source_identity():
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    prefixes = ("crates/", "python/", "tests/", ".claude/skills/verify-oh-my-vllm/")
    files = {}
    for name in sorted(set(paths)):
        path = ROOT / name
        if (
            name.startswith(prefixes)
            and path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(ROOT)
        ):
            files[name] = digest(path)
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "files": files,
        "digest": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "snapshot_atomic": False,
    }


def manifest(binary, feature):
    return {
        "started_at": timestamp(),
        "scope": "CPU fixture: Rust service and transport only",
        "gpu_inference": False,
        "numerical_acceptance": False,
        "performance_acceptance": False,
        "binary_source_match": "unverified; rebuild and rerun after source freeze",
        "feature": feature,
        "binary": {"path": str(binary), "sha256": digest(binary)},
        "source": source_identity(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": {name: version(name) for name in ("pyzmq", "msgpack", "torch")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
