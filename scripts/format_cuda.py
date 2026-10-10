"""Format owned CUDA modules and enforce their 800-line size limit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "python/oh_my_vllm/kernels/cuda_backend"
MAX_LINES = 800


def oversized_files(directory: Path) -> list[tuple[Path, int]]:
    """Count physical lines in all owned native-backend code files."""
    return [
        (path, lines)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in {".cu", ".cuh", ".py"}
        and (lines := len(path.read_text().splitlines())) > MAX_LINES
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject formatting drift")
    options = parser.parse_args()
    files = sorted(
        path for path in BACKEND.rglob("*") if path.suffix in {".cu", ".cuh"}
    )
    flags = ["--dry-run", "--Werror"] if options.check else ["-i"]
    result = subprocess.run(
        ["clang-format", "--style=file", "--fail-on-incomplete-format", *flags, *files],
        check=False,
    )
    oversized = oversized_files(BACKEND)
    for path, lines in oversized:
        print(f"{path.relative_to(ROOT)}: {lines} lines exceeds {MAX_LINES}")
    if result.returncode or oversized:
        return 1
    print(f"CUDA formatting and module size passed ({len(files)} native files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
