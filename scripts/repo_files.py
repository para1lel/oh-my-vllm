"""Select owned working-tree files for repository checks."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    "3rdparty",
    "third_party",
    "third-party",
    "vendor",
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
}


def _git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=root, input=data, capture_output=True, check=False
    )
    if result.returncode not in (0, 1) or (
        result.returncode == 1 and arguments[0] != "check-ignore"
    ):
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def owned_files(root: Path = ROOT) -> list[Path]:
    root = root.resolve()
    output = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = set(filter(None, output.split(b"\0")))
    ignored = _git(root, "check-ignore", "--no-index", "--stdin", "-z", data=output)
    names.difference_update(filter(None, ignored.split(b"\0")))
    attributes = _git(
        root,
        "check-attr",
        "-z",
        "--stdin",
        "linguist-generated",
        "linguist-vendored",
        data=b"\0".join(sorted(names)) + b"\0" if names else b"",
    ).split(b"\0")[:-1]
    for name, _, value in zip(
        attributes[0::3], attributes[1::3], attributes[2::3], strict=True
    ):
        if value in {b"set", b"true"}:
            names.discard(name)
    return [
        Path(os.fsdecode(name))
        for name in sorted(names)
        if is_owned_path(root, Path(os.fsdecode(name)))
    ]


def is_owned_path(root: Path, relative: Path) -> bool:
    if relative.is_absolute() or ".." in relative.parts:
        return False
    if EXCLUDED_PARTS.intersection(relative.parts):
        return False
    path = root / relative
    if any((root / parent).is_symlink() for parent in [relative, *relative.parents]):
        return False
    return path.is_file() and path.resolve().is_relative_to(root.resolve())
