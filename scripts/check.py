"""Repository checks: doctor, size, read-only static checks, CPU tests, or hooks."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.code_size_syntax import external_spans, validate_versions
from scripts.repo_files import ROOT, owned_files

PYTHON = sys.executable


def run(arguments: list[str], *, env=None) -> int:
    print("+ " + " ".join(arguments), flush=True)
    try:
        return subprocess.run(arguments, cwd=ROOT, env=env, check=False).returncode
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1


def doctor() -> int:
    validate_versions()
    tools = {
        "git": ["--version"],
        "cargo": ["--version"],
        "node": ["--version"],
        "clang-format": ["--version"],
        "ruff": ["--version"],
        "pre-commit": ["--version"],
        "rustfmt": ["--version"],
    }
    for tool, flags in tools.items():
        if shutil.which(tool) is None:
            raise RuntimeError(f"missing development tool: {tool}")
        subprocess.run([tool, *flags], cwd=ROOT, check=True)
    node_version = subprocess.check_output(["node", "--version"], text=True).strip()
    if int(node_version.lstrip("v").split(".")[0]) < 24:
        raise RuntimeError("Node 24 or later required")
    for language, path in (
        ("rust", Path("development/style-check/src/main.rs")),
        ("javascript", Path("development/javascript/check-spacing.mjs")),
    ):
        results = external_spans(ROOT, language, [path])
        if not results or any(result["errors"] for result in results):
            raise RuntimeError(f"{language} parser self-check failed")
    hook = subprocess.check_output(
        ["git", "rev-parse", "--git-path", "hooks/pre-commit"], cwd=ROOT, text=True
    ).strip()
    if not os.access(ROOT / hook, os.X_OK):
        raise RuntimeError("pre-commit hook is not installed or executable")
    print("Development tools and pinned parsers are ready")
    return 0


def static_checks() -> int:
    paths = owned_files()
    python = [str(path) for path in paths if path.suffix in {".py", ".pyi"}]
    commands = [
        [PYTHON, "scripts/check_code_size.py", "--all"],
        [PYTHON, "scripts/format_cuda.py", "--check"],
        [PYTHON, "scripts/run_cargo_hook.py", "fmt"],
        [PYTHON, "scripts/check_rust_line_width.py"],
        [PYTHON, "scripts/check_definition_spacing.py"],
        ["ruff", "format", "--check", "--force-exclude", *python],
        ["ruff", "check", "--force-exclude", *python],
        [
            PYTHON,
            "scripts/fix_whitespace.py",
            "--check",
            *(str(path) for path in paths),
        ],
        [PYTHON, "scripts/check_docs.py"],
        [PYTHON, "scripts/build_glossary.py", "--check"],
    ]
    results = [run(command) for command in commands]
    return int(any(results))


def cpu_tests(arguments) -> int:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if not arguments.python_only and not arguments.list:
        result = run([PYTHON, "scripts/run_cargo_hook.py", "test"], env=environment)
        if result:
            return result
    return run(
        [
            PYTHON,
            "scripts/run_cpu_tests.py",
            *(["--list"] if arguments.list else []),
            *arguments.modules,
        ],
        env=environment,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "doctor", help="check tools, locked parsers, and installed hooks"
    )
    size = subcommands.add_parser(
        "size", help="audit all owned files and functions (500/80 physical lines)"
    )
    size.add_argument(
        "--all", action="store_true", help="explicit full audit (the default)"
    )
    size.add_argument(
        "--inventory",
        action="store_true",
        help="list spans without failing on size violations",
    )
    size.add_argument(
        "--json", action="store_true", help="emit complete JSON inventory"
    )
    subcommands.add_parser(
        "static",
        help="read-only formatting, lint, size, spacing, and docs; no Clippy or tests",
    )
    cpu = subcommands.add_parser(
        "cpu", help="Rust tests and all owned Python tests with CUDA disabled"
    )
    cpu.add_argument(
        "--python-only",
        action="store_true",
        help="Python tests only; Cargo owns service tests",
    )
    cpu.add_argument(
        "--list",
        action="store_true",
        help="list unique Python test IDs without running tests or Cargo",
    )
    cpu.add_argument(
        "modules",
        nargs="*",
        help="optional focused Python unittest modules or test IDs",
    )
    subcommands.add_parser(
        "hooks", help="run complete pre-commit --all-files once, with its normal fixes"
    )
    options = parser.parse_args(argv)
    try:
        if options.command == "size":
            from scripts.check_code_size import main as size_main

            return size_main(
                [
                    "--all",
                    *(["--inventory"] if options.inventory else []),
                    *(["--json"] if options.json else []),
                ]
            )
        if options.command == "doctor":
            return doctor()
        if options.command == "static":
            return static_checks()
        if options.command == "cpu":
            return cpu_tests(options)
        return run(["pre-commit", "run", "--all-files"])
    except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
