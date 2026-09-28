"""Evidence checks shared by the isolated collector and independent comparator."""

import datetime
import hashlib
import os
import re
import subprocess
from pathlib import Path

FROZEN_SHA = "e9f169d16b9408bb9ae44f75072b91a5521d733c"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def hardware_identity():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        raise RuntimeError("select one UUID-pinned GPU with scripts/with-gpu.sh")
    return {
        "gpu": gpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_info": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }


def audit(log, cache_roots):
    """Reject capture/compile activity after measured start."""
    log = Path(log)
    text = log.read_text()
    log_path = log.resolve()
    start = None
    problems = []
    for line in text.splitlines():
        if "BENCH_PHASE" in line and "measured=true" in line and start is None:
            match = re.search(r"\d{4}-\d\d-\d\dT[\d:.]+(?:Z|\+00:00)", line)
            if match:
                start = datetime.datetime.fromisoformat(
                    match[0].replace("Z", "+00:00")
                ).timestamp()
        if start and re.search(
            r"captur(?:e|ing).*graph|compil(?:ing|ation started)", line, re.I
        ):
            problems.append(line)
    cache_tree_sha256 = {}
    if start is None:
        problems.append("missing measured phase timestamp")
    file_digests = {}
    for root in cache_roots:
        if not root:
            problems.append("cache root is unset — steady-state cannot be verified")
            continue
        root = Path(root)
        if not root.is_dir():
            problems.append(f"cache root is missing: {root}")
            continue
        tree = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved == log_path:
                continue
            relative = str(path.relative_to(root)).encode()
            tree.update(len(relative).to_bytes(8, "little"))
            tree.update(relative)
            digest = file_digests.get(resolved)
            if digest is None:
                digest = _sha256_file(path)
                file_digests[resolved] = digest
            tree.update(digest)
            if start is not None and path.stat().st_mtime_ns >= int(start * 1e9):
                problems.append(f"cache file changed during measurements: {path}")
        cache_tree_sha256[str(root)] = tree.hexdigest()
    return {
        "steady_state_verified": not problems,
        "problems": problems,
        "log_path": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "cache_roots": cache_roots,
        "cache_tree_sha256": cache_tree_sha256,
    }
