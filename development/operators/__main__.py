import argparse
import json
from pathlib import Path

from .cases import load_cases
from .fixtures import load_fixture


def main():
    parser = argparse.ArgumentParser(description="Fixed operator validation cases")
    parser.add_argument("command", choices=("list", "fixtures", "numerics"))
    parser.add_argument(
        "--purpose", choices=("numerics", "performance"), default="numerics"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "numerics":
        if arguments.model is None or arguments.report is None:
            parser.error("numerical acceptance requires --model and --report")
        from .validation import numerical

        raise SystemExit(0 if numerical(arguments.model, arguments.report) else 1)
    cases = [case for case in load_cases() if case.purpose == arguments.purpose]
    if arguments.command == "fixtures":
        for case in cases:
            load_fixture(case)
    print(
        json.dumps(
            [{"id": case.id, "identity": case.identity} for case in cases], indent=2
        )
    )


if __name__ == "__main__":
    main()
