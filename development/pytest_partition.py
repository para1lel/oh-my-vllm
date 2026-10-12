"""Partition context-boundary collection without losing explicit selectors."""

from pathlib import Path

import pytest

BOUNDARY = Path(__file__).resolve().parents[1] / "tests/test_context_boundary.py"


def pytest_addoption(parser):
    parser.addoption(
        "--oh-my-vllm-test-partition",
        choices=("boundary", "remaining"),
        default=None,
        help="select the boundary module or the remaining test modules",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    partition = config.getoption("--oh-my-vllm-test-partition")
    if partition is None:
        return
    selected, deselected = [], []
    for item in items:
        boundary = item.path.resolve() == BOUNDARY
        keep = boundary == (partition == "boundary")
        (selected if keep else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
