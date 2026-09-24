"""Evidence checks shared by the isolated collector and independent comparator."""

import datetime
import hashlib
import os
import re
import subprocess
from pathlib import Path

FROZEN_SHA = "e9f169d16b9408bb9ae44f75072b91a5521d733c"


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
    text = Path(log).read_text()
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
    if start is None:
        problems.append("missing measured phase timestamp")
    else:
        for root in cache_roots:
            if not root:
                problems.append("cache root is unset — steady-state cannot be verified")
                continue
            for path in Path(root).rglob("*"):
                if (
                    path.is_file()
                    and path.suffix in {".so", ".cubin", ".ptx", ".o"}
                    and path.stat().st_mtime >= start
                ):
                    problems.append(
                        f"compiled file changed during measurements: {path}"
                    )
    return {
        "steady_state_verified": not problems,
        "problems": problems,
        "log_path": str(log),
        "log_sha256": hashlib.sha256(Path(log).read_bytes()).hexdigest(),
        "cache_roots": cache_roots,
    }
