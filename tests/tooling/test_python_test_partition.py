"""Compare isolated pytest collection with the original selected node multiset."""

import importlib.util
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "python_test_runner", ROOT / "scripts/run_python_tests.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def collect(command, directory):
    result = subprocess.run(
        [*command, "--collect-only"],
        cwd=directory,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
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
            "tests/numerical/test_compiler_metadata_gpu.py::test_graph_replay",
        ],
        ["-m", "gpu", "tests/acceptance/test_context_boundary.py"],
        ["-m", "gpu", "-k", "not max_context_length_262144"],
        ["-m", "gpu", "--keep-duplicates", "tests/acceptance/test_context_boundary.py"],
        ["-m", "gpu", "--ignore=tests/acceptance/test_context_boundary.py"],
        ["-m", "gpu", "--ignore-glob=*context_boundary.py"],
        ["-m", "gpu", "-k", "no_such_test_selected"],
    ],
)
def test_isolated_collection_preserves_selected_nodes_and_multiplicity(
    arguments, collection_project, tmp_path
):
    common = [
        "--rootdir=" + str(collection_project),
        "--confcutdir=" + str(collection_project),
        "-c",
        str(ROOT / "pytest.ini"),
    ]
    original = collect(
        [sys.executable, "-m", "pytest", "tests", "-q", *common, *arguments],
        collection_project,
    )
    partitions = [
        collect(command, collection_project)
        for _, command in RUNNER.commands([*common, *arguments], tmp_path)
    ]
    assert not (partitions[0].keys() & partitions[1].keys())
    assert partitions[0] + partitions[1] == original


@pytest.fixture(scope="module")
def collection_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("test-selection")
    modules = {
        "acceptance/test_context_boundary.py": (
            "@pytest.mark.gpu\n@pytest.mark.isolated_gpu\n"
            "@pytest.mark.parametrize('batch', [1, 2, 4])\n"
            "def test_max_context_length_262144(batch):\n    pass\n"
        ),
        "numerical/test_compiler_metadata_gpu.py": (
            "@pytest.mark.gpu\ndef test_graph_replay():\n    pass\n"
        ),
        "unit/test_cache.py": "def test_commit():\n    pass\n",
    }
    for name, body in modules.items():
        path = root / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import pytest\n\n" + body)
    return root
