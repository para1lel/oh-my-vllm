"""Audit all owned code: files <=500 and functions <=80 physical lines."""

import argparse
import json
import sys
from pathlib import Path

from scripts.code_size_syntax import (
    external_spans,
    inspect_source,
    parser_for,
    source_language,
    validate_versions,
)
from scripts.repo_files import ROOT, owned_files

FILE_LIMIT = 500
FUNCTION_LIMIT = 80


def physical_lines(source: bytes) -> int:
    if b"\r" in source.replace(b"\r\n", b""):
        raise ValueError("bare CR line endings are not supported; use LF or CRLF")
    return source.count(b"\n") + int(bool(source) and not source.endswith(b"\n"))


def audit(root: Path) -> dict:
    validate_versions()
    report = {
        "limits": {"file": FILE_LIMIT, "function": FUNCTION_LIMIT},
        "files": [],
        "violations": [],
        "errors": [],
    }
    parsers, external = {}, {}
    for relative in owned_files(root):
        path = root / relative
        try:
            language = source_language(path)
            if language is None:
                continue
            source = path.read_bytes()
            source.decode("utf-8")
            record = {
                "path": relative.as_posix(),
                "language": language,
                "lines": physical_lines(source),
                "functions": [],
            }
            report["files"].append(record)
            if language in {"rust", "javascript", "typescript"}:
                external.setdefault(language, []).append(record)
                continue
            if language != "python" and language not in parsers:
                parsers[language] = parser_for(language)
            functions, errors = inspect_source(source, language, parsers.get(language))
            record["functions"] = functions
            report["errors"].extend(
                {"path": relative.as_posix(), **error} for error in errors
            )
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            report["errors"].append(
                {"path": relative.as_posix(), "message": str(error)}
            )
    for language, records in external.items():
        collect_external(root, language, records, report)
    report["violations"] = [
        issue for record in report["files"] for issue in size_violations(record)
    ]
    return report


def collect_external(root: Path, language: str, records: list, report: dict) -> None:
    try:
        results = external_spans(
            root, language, [Path(record["path"]) for record in records]
        )
        by_path = {str(root / record["path"]): record for record in records}
        if {item["path"] for item in results} != set(by_path):
            raise RuntimeError(f"{language} parser returned incomplete file coverage")
        for item in results:
            record = by_path[item["path"]]
            record["functions"] = item["functions"]
            report["errors"].extend(
                {"path": record["path"], **error} for error in item["errors"]
            )
    except (ValueError, OSError, RuntimeError) as error:
        report["errors"].append({"path": ".", "message": str(error)})


def size_violations(record: dict) -> list[dict]:
    path, count = record["path"], record["lines"]
    violations = []
    if count > FILE_LIMIT:
        violations.append(
            {
                "path": path,
                "kind": "file",
                "line": 1,
                "lines": count,
                "limit": FILE_LIMIT,
            }
        )
    for function in record["functions"]:
        if function["lines"] > FUNCTION_LIMIT:
            violations.append(
                {
                    "path": path,
                    "kind": "function",
                    "line": function["start"],
                    "lines": function["lines"],
                    "limit": FUNCTION_LIMIT,
                    "name": function["name"],
                }
            )
    return violations


def print_report(report: dict, inventory: bool) -> None:
    for issue in report["errors"]:
        print(
            f"{issue['path']}:{issue.get('line', 1)}: {issue['message']}",
            file=sys.stderr,
        )
    for issue in report["violations"]:
        name = f" {issue['name']}" if "name" in issue else ""
        print(
            f"{issue['path']}:{issue['line']}: {issue['kind']}{name}: "
            f"{issue['lines']} physical lines exceeds {issue['limit']}"
        )
    if inventory:
        for record in report["files"]:
            print(f"{record['path']}: {record['lines']} lines, {record['language']}")
            for function in record["functions"]:
                print(
                    f"  {function['start']}-{function['end']} "
                    f"{function['name']}: {function['lines']}"
                )
    print(
        f"Audited {len(report['files'])} code files; "
        f"{len(report['violations'])} size violations; {len(report['errors'])} errors"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="explicit full audit (also the default)"
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="list functions; size violations do not fail",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit complete machine-readable inventory"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Git repository to audit"
    )
    options = parser.parse_args(argv)
    try:
        report = audit(options.root.resolve())
    except (ImportError, OSError, RuntimeError) as error:
        report = {
            "files": [],
            "violations": [],
            "errors": [{"path": ".", "message": str(error)}],
        }
    if options.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, options.inventory)
    return int(
        bool(report["errors"] or (report["violations"] and not options.inventory))
    )


if __name__ == "__main__":
    raise SystemExit(main())
