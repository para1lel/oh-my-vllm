import argparse
import json

from .cases import load_cases
from .fixtures import load_fixture


def main():
    parser = argparse.ArgumentParser(description="Fixed operator validation cases")
    parser.add_argument("command", choices=("list", "fixtures"))
    parser.add_argument(
        "--purpose", choices=("numerics", "performance"), default="numerics"
    )
    arguments = parser.parse_args()
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
