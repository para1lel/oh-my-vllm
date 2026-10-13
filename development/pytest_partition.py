"""Partition context-boundary collection without losing explicit selectors."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--oh-my-vllm-test-partition",
        choices=("boundary", "remaining"),
        default=None,
        help="select isolated_gpu cases or the remaining tests",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    partition = config.getoption("--oh-my-vllm-test-partition")
    if partition is None:
        return
    selected, deselected = [], []
    for item in items:
        boundary = item.get_closest_marker("isolated_gpu") is not None
        keep = boundary == (partition == "boundary")
        (selected if keep else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
