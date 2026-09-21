"""Enforce rustfmt's configured width even inside macros, strings and comments."""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def violations(path: Path, max_width: int, tab_spaces: int) -> list[str]:
    return [
        f"{path}:{number}: {width} columns exceeds Rust max_width={max_width}"
        for number, line in enumerate(path.read_text().split("\n"), 1)
        if (width := len(line.expandtabs(tab_spaces))) > max_width
    ]


def main() -> int:
    config = tomllib.loads((ROOT / "rustfmt.toml").read_text())
    paths = (
        [Path(arg) for arg in sys.argv[1:]]
        if len(sys.argv) > 1
        else [
            ROOT / name
            for name in subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                    "*.rs",
                ],
                cwd=ROOT,
            )
            .decode()
            .split("\0")
            if name and (ROOT / name).is_file()
        ]
    )
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
