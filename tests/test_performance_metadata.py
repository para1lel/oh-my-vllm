"""Actual FX metadata distinguishes address overhead from unknown GPU work."""

import json
import operator
from copy import deepcopy

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


def test_loaded_gemm_must_bind_the_reviewed_owned_accumulator_header(monkeypatch):
    from oh_my_vllm.performance import coverage

    reviewed = {
        "sources": {
            "kernels/cuda_backend/kernels.cu": "a" * 64,
            "kernels/cuda_backend/groupwise_fp8.cu": "b" * 64,
        },
        "gemm_inputs": {"project_headers": {"groupwise_accum.cuh": "c" * 64}},
    }
    monkeypatch.setattr(coverage, "validate_weights", lambda weights: None)
    monkeypatch.setattr(coverage.Path, "read_text", lambda path: json.dumps(reviewed))
    trace = {
        "runtime_contract": reviewed,
        "cuda_provenance": {
            "source_sha256": "a" * 64,
            "so_sha256": "d" * 64,
            "build_input_sha256": "e" * 64,
        },
        "cuda_gemm_provenance": {
            "source_sha256": "b" * 64,
            "so_sha256": "f" * 64,
            "build_input_sha256": "1" * 64,
            "project_headers": {"groupwise_accum.cuh": "c" * 64},
        },
        "graph_operations": [("target", "_operator.add")],
        "steps": [],
    }
    coverage.validate_inventory(trace)
    for headers in ({}, {"groupwise_accum.cuh": "2" * 64}):
        altered = deepcopy(trace)
        altered["cuda_gemm_provenance"]["project_headers"] = headers
        with pytest.raises(
            ValueError, match="templates or flags require semantic review"
        ):
            coverage.validate_inventory(altered)
