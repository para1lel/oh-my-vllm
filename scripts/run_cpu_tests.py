"""Discover owned Python CPU tests; Cargo owns tests.test_service."""

import argparse
import os
import sys
import unittest

from scripts.repo_files import owned_files


def test_modules() -> list[str]:
    return [
        ".".join(path.with_suffix("").parts)
        for path in owned_files()
        if path.parts[0] in {"tests", "development"}
        and path.suffix == ".py"
        and path.name.startswith("test_")
        and path.as_posix() != "tests/test_service.py"
    ]


def cases(suite):
    for entry in suite:
        if isinstance(entry, unittest.TestSuite):
            yield from cases(entry)
        else:
            yield entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="collect and list unique test IDs without running",
    )
    parser.add_argument(
        "modules", nargs="*", help="optional focused unittest modules or test IDs"
    )
    options = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error(
            "CPU tests require CUDA_VISIBLE_DEVICES=''; use scripts/check.py cpu"
        )
    modules = options.modules or test_modules()
    if "tests.test_service" in modules:
        parser.error(
            "tests.test_service runs through cargo test with its service binary"
        )
    loaded = unittest.defaultTestLoader.loadTestsFromNames(modules)
    unique = {case.id(): case for case in cases(loaded)}
    if options.list:
        print("\n".join(unique))
        print(
            f"Collected {len(unique)} unique Python CPU tests; "
            "service tests belong to Cargo",
            file=sys.stderr,
        )
        for error in unittest.defaultTestLoader.errors:
            print(error, file=sys.stderr)
        return int(bool(unittest.defaultTestLoader.errors))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(unique.values())
    )
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
