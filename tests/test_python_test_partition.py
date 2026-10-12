"""Compare isolated pytest collection with the original selected node multiset."""

import importlib.util
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "python_test_runner", ROOT / "scripts/run_python_tests.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def collect(command):
    result = subprocess.run(
        [*command, "--collect-only"],
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return Counter(
        line for line in result.stdout.splitlines() if line.startswith("tests/")
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "gpu"],
        [
            "-m",
            "gpu",
            "tests/test_compiler_metadata_gpu.py::"
            "test_owned_metadata_retirement_preserves_capture_and_cold_recapture",
        ],
        ["-m", "gpu", "tests/test_context_boundary.py"],
        ["-m", "gpu", "-k", "not max_context_length_262144"],
        ["-m", "gpu", "--keep-duplicates", "tests/test_context_boundary.py"],
        ["-m", "gpu", "--ignore=tests/test_context_boundary.py"],
        ["-m", "gpu", "--ignore-glob=*context_boundary.py"],
        ["-m", "gpu", "-k", "no_such_test_selected"],
    ],
)
def test_isolated_collection_preserves_selected_nodes_and_multiplicity(
    arguments, tmp_path
):
    original = collect([sys.executable, "-m", "pytest", "tests", "-q", *arguments])
    partitions = [
        collect(command) for _, command in RUNNER.commands(arguments, tmp_path)
    ]
    assert not (partitions[0].keys() & partitions[1].keys())
    assert partitions[0] + partitions[1] == original
