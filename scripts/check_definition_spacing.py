import argparse
import ast
import subprocess
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def python_spacing_gaps(source: str) -> dict[int, int]:
    syntax = ast.parse(source)
    lines = source.splitlines()
    missing = {}
    for container in ast.walk(syntax):
        for _, body in ast.iter_fields(container):
            if not isinstance(body, list) or not all(
                isinstance(item, ast.stmt) for item in body
            ):
                continue
            for previous, current in pairwise(body):
                _record_gap(container, previous, current, lines, missing)
    return missing


def _record_gap(container, previous, current, lines, missing):
    if not isinstance(previous, DEFINITIONS) and not isinstance(current, DEFINITIONS):
        return
    start = current.lineno - 1
    if isinstance(current, DEFINITIONS) and current.decorator_list:
        start = current.decorator_list[0].lineno - 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    while start > previous.end_lineno:
        line = lines[start - 1]
        if (
            not line.lstrip().startswith("#")
            or len(line) - len(line.lstrip()) != indent
        ):
            break
        start -= 1
    required = 2 if isinstance(container, ast.Module) else 1
    present = sum(not line.strip() for line in lines[previous.end_lineno : start])
    if present < required:
        missing[start] = max(missing.get(start, 0), required - present)


def check_python(path: Path, fix: bool) -> bool:
    source = path.read_text()
    missing = python_spacing_gaps(source)
    if missing and fix:
        lines = source.splitlines()
        for line, count in sorted(missing.items(), reverse=True):
            lines[line:line] = [""] * count
        path.write_text("\n".join(lines) + "\n")
        return False
    for line, count in missing.items():
        print(
            f"{path.relative_to(ROOT)}:{line + 1}: "
            f"add {count} blank line(s) before this definition"
        )
    return bool(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    options = parser.parse_args()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    files = sorted(
        {
            ROOT / name.decode()
            for name in names.split(b"\0")
            if name
            and not name.startswith(b"3rdparty/")
            and (ROOT / name.decode()).is_file()
            and not (ROOT / name.decode()).is_symlink()
            and (ROOT / name.decode()).resolve().is_relative_to(ROOT)
        }
    )
    failed = False
    for path in files:
        if path.suffix == ".py":
            failed |= check_python(path, options.fix)
    rust = [str(path) for path in files if path.suffix == ".rs"]
    result = (
        subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "--manifest-path",
                str(ROOT / "development/style-check/Cargo.toml"),
                "--",
                *(["--fix"] if options.fix else []),
                *rust,
            ],
            cwd=ROOT,
            check=False,
        )
        if rust
        else None
    )
    javascript = [
        str(path) for path in files if path.suffix in {".js", ".mjs", ".cjs", ".ts"}
    ]
    javascript_result = (
        subprocess.run(
            [
                "node",
                str(ROOT / "development/javascript/check-spacing.mjs"),
                *(["--fix"] if options.fix else []),
                *javascript,
            ],
            cwd=ROOT,
            check=False,
        )
        if javascript
        else None
    )
    if (
        (result is not None and result.returncode)
        or (javascript_result is not None and javascript_result.returncode)
        or failed
    ):
        return 1
    print("Function and module definition spacing passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
