#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def fix(path: Path, *, check: bool = False) -> bool:
    original = path.read_bytes()

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

    if not check:
        path.write_text(cleaned, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    check = bool(argv and argv[0] == "--check")
    changed = [str(p) for arg in argv[int(check) :] if fix(p := Path(arg), check=check)]
    for name in changed:
        print(f"{'needs whitespace fix:' if check else 'fixed'} {name}")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
