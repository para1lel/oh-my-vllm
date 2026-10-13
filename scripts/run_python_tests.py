"""Keep context-boundary tests in a process without an earlier CUDA context."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import run_engine  # noqa: E402
from benchmarks.context_boundary import _reap_on_termination  # noqa: E402


def commands(pytest_args, evidence, *, test_root="tests"):
    return [
        (
            name,
            [
                sys.executable,
                "-m",
                "pytest",
                test_root,
                "-q",
                *pytest_args,
                "-p",
                "development.pytest_partition",
                "--oh-my-vllm-test-partition=" + name,
                "-ra",
                "--junitxml=" + str(evidence / (name + ".xml")),
                "--basetemp=" + str(evidence / (name + "-tmp")),
            ],
        )
        for name in ("boundary", "remaining")
    ]


def run(pytest_args, evidence):
    any_tests = False
    with _reap_on_termination() as cancelled:
        for name, command in commands(pytest_args, evidence):
            print("Running " + name + " tests in a fresh process", flush=True)
            log = evidence / (name + ".log")
            try:
                output = run_engine(
                    command,
                    timeout=14400,
                    stderr=subprocess.STDOUT,
                    log_path=log,
                    cancelled=cancelled,
                )
            except subprocess.CalledProcessError as error:
                print(log.read_text(), end="", flush=True)
                if error.returncode == 5:
                    continue  # A user filter can select only the other group.
                return error.returncode
            else:
                print(output, end="", flush=True)
                any_tests = True
    return 0 if any_tests else 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path)
    args, pytest_args = parser.parse_known_args()
    if args.evidence_dir is not None:
        evidence = args.evidence_dir.resolve()
        if evidence.is_relative_to(ROOT):
            parser.error("test evidence must be outside the repository")
        evidence.mkdir(parents=True, exist_ok=False)
        return run(pytest_args, evidence)
    evidence = Path(tempfile.mkdtemp(prefix="oh-my-vllm-tests-"))
    print("Test evidence: " + str(evidence), flush=True)
    return run(pytest_args, evidence)


if __name__ == "__main__":
    sys.exit(main())
