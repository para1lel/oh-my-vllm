"""Actual FX metadata distinguishes address overhead from unknown GPU work."""

import json
import operator

import pytest
import torch
from oh_my_vllm.performance import execution
from oh_my_vllm.performance.coverage import metadata_covers
from torch._subclasses import fake_tensor
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture
def inventory(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(fake_tensor, "init_gpu_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(execution, "_GRAPH_OPERATIONS", set())
    monkeypatch.setattr(execution, "_GRAPH_METADATA", set())
    yield lambda: [json.loads(row) for row in execution._GRAPH_METADATA]
    assert torch.cuda.is_initialized() == initialized


def address_graph(*, dtype=torch.int64, divisor=784, extra_consumer=False):
    graph = torch.fx.Graph()
    with FakeTensorMode():
        value = torch.empty(2, dtype=dtype, device="cuda:0")
        positions = graph.placeholder("host_positions")
        positions.meta["example_value"] = value
        for operation, consumer in (
            (operator.floordiv, operator.getitem),
            (operator.mod, operator.add),
        ):
            address = graph.call_function(operation, (positions, divisor))
            address.meta["example_value"] = value
            graph.call_function(consumer, (positions, address))
            if extra_consumer:
                graph.call_function(torch.sin, (address,))
    return graph


def test_reviewed_proposal_addresses_need_typed_evidence(inventory):
    execution.observe_graph("proposal_graph", address_graph())
    for operation in ("_operator.floordiv", "_operator.mod"):
        assert metadata_covers("proposal_graph", operation, inventory())
        assert not metadata_covers("proposal_graph", operation, [])


@pytest.mark.parametrize(
    "unit,options",
    [
        ("target", {}),
        ("proposal_graph", {"dtype": torch.float32}),
        ("proposal_graph", {"divisor": 128}),
        ("proposal_graph", {"extra_consumer": True}),
    ],
)
def test_non_address_arithmetic_stays_unknown(inventory, unit, options):
    execution.observe_graph(unit, address_graph(**options))
    assert not metadata_covers(unit, "_operator.floordiv", inventory())
    assert not metadata_covers(unit, "_operator.mod", inventory())


def test_one_invalid_occurrence_cannot_hide_behind_a_valid_name(inventory):
    execution.observe_graph("proposal_graph", address_graph())
    execution.observe_graph("proposal_graph", address_graph(divisor=128))
    assert not metadata_covers("proposal_graph", "_operator.mod", inventory())


@pytest.mark.parametrize("result", [torch.Size([2, 4]), 2])
def test_tensor_size_is_host_shape_metadata(inventory, result):
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    tensor.meta["example_value"] = torch.empty(2, 4)
    query = graph.call_method("size", (tensor,))
    query.meta["example_value"] = result
    execution.observe_graph("dspark", graph)
    assert metadata_covers("dspark", "call_method:size", inventory())


@pytest.mark.parametrize("result", [None, torch.empty(1), 1.0, True, (2, 1.0)])
def test_missing_or_device_value_size_metadata_blocks_acceptance(inventory, result):
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    tensor.meta["example_value"] = torch.empty(2, 4)
    query = graph.call_method("size", (tensor,))
    query.meta["example_value"] = result
    execution.observe_graph("dspark", graph)
    assert not metadata_covers("dspark", "call_method:size", inventory())
