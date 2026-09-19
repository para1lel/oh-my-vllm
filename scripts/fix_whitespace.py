#!/usr/bin/env python3
"""Strip trailing whitespace and enforce exactly one final newline.

Stands in for pre-commit-hooks' trailing-whitespace/end-of-file-fixer so the hook
set stays `repo: local` and needs no network. Binary files are skipped; so are
files whose content is already clean, which keeps mtimes stable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def fix(path: Path) -> bool:
    """Rewrite ``path`` if it needs cleaning. Returns True if it was changed."""
    try:
        original = path.read_bytes()
    except OSError as exc:
        print(f"{path}: cannot read: {exc}", file=sys.stderr)
        return False

    if b"\x00" in original:
        return False

    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        return False

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    cleaned = "".join(f"{line}\n" for line in lines)

    if cleaned.encode("utf-8") == original:
        return False

    path.write_text(cleaned, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    changed = [str(p) for arg in argv if fix(p := Path(arg))]
    for name in changed:
        print(f"fixed {name}")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
