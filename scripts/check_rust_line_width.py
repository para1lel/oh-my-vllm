"""Enforce rustfmt's configured width even inside macros, strings and comments."""

import sys
import tomllib
from pathlib import Path

from scripts.repo_files import ROOT, owned_files


def violations(path: Path, max_width: int, tab_spaces: int) -> list[str]:
    return [
        f"{path}:{number}: {width} columns exceeds Rust max_width={max_width}"
        for number, line in enumerate(path.read_text().split("\n"), 1)
        if (width := len(line.expandtabs(tab_spaces))) > max_width
    ]


def main() -> int:
    config = tomllib.loads((ROOT / "rustfmt.toml").read_text())
    owned = {ROOT / path for path in owned_files() if path.suffix == ".rs"}
    paths = sorted(owned)
    if len(sys.argv) > 1:
        requested = {Path(arg).absolute() for arg in sys.argv[1:]}
        paths = sorted(owned & requested)
    errors = [
        error
        for path in paths
        for error in violations(path, config["max_width"], config.get("tab_spaces", 4))
    ]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
