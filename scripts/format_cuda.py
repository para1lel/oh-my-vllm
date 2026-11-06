"""Format owned C, C++, and CUDA sources with clang-format."""

from __future__ import annotations

import argparse
import subprocess

from scripts.repo_files import ROOT, owned_files

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject formatting drift")
    options = parser.parse_args()
    files = [
        str(ROOT / path) for path in owned_files() if path.suffix in SOURCE_SUFFIXES
    ]
    flags = ["--dry-run", "--Werror"] if options.check else ["-i"]
    result = (
        subprocess.run(
            [
                "clang-format",
                "--style=file",
                "--fail-on-incomplete-format",
                *flags,
                *files,
            ],
            check=False,
        )
        if files
        else None
    )
    if result is not None and result.returncode:
        return 1
    print(f"Native formatting passed ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
