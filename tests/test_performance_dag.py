"""Analytic examples distinguish data dependencies from launch ordering."""

import pytest
from oh_my_vllm.performance.dag import Dag, Hardware, Node, b200, retention_bytes
from oh_my_vllm.performance.model import unique_context_tokens


def test_independent_branches_overlap_and_join_uses_longest_path():
    graph = Dag(Hardware({"simt_fp32": 10}, 10))
    a = graph.add(Node("a", work={"simt_fp32": 20}))
    b = graph.add(Node("b", work={"simt_fp32": 30}))
    join = graph.add(Node("join", (a, b), {"simt_fp32": 10}))
    assert graph.bound(join) == 4
    assert graph.bound(b) == 3


def test_memory_and_compute_overlap_within_an_atomic_operation():
    graph = Dag(Hardware({"simt_fp32": 10}, 10))
    node = graph.add(Node("atomic", work={"simt_fp32": 20}, external_bytes=30))
    assert graph.bound(node) == 3


def test_unknown_operations_and_invalid_dependency_fail_closed():
    graph = Dag(Hardware({"simt_fp32": 10}, 10))
    with pytest.raises(ValueError, match="unrecognized"):
        graph.add(Node("unknown", work={"tensor_tf32": 20}))
    with pytest.raises(ValueError, match="dependency"):
        graph.add(Node("cycle", (0,)))
    assert graph.nodes == []


def test_prefetch_requires_an_independent_memory_node():
    graph = Dag(Hardware({"simt_fp32": 1}, 1))
    weights = graph.add(Node("independent weights", external_bytes=10))
    a = graph.add(Node("activation", work={"simt_fp32": 10}))
    b = graph.add(Node("uses activation and weights", (weights, a), {"simt_fp32": 1}))
    assert graph.bound(b) == 11
    # Charging weight transfer after activation would produce 20, which
    # exceeds a possible prefetched execution and is not a general lower bound.


def test_cache_credit_cannot_be_applied_independently_to_every_node():
    graph = Dag(Hardware({"simt_fp32": 10}, 10, cache_bytes=10))
    a = graph.add(Node("unique parameter range a", external_bytes=20))
    b = graph.add(Node("unique parameter range b", (a,), external_bytes=20))
    assert graph.bound(b, cache_credit_bytes=10) == 3
    with pytest.raises(ValueError, match="capacity"):
        graph.bound(b, cache_credit_bytes=11)


def test_peak_values_are_dense_single_gpu_rates():
    hardware = b200(148, 1965e6, 132644864)
    assert hardware.rates["tensor_fp8"] == 4.5e15
    assert hardware.rates["tensor_bf16"] == 2.25e15
    assert hardware.rates["simt_fp32"] == 75e12
    assert hardware.rates["simt_fp64"] == 37e12
    assert hardware.hbm_bytes_per_s == 7.7e12


def test_resource_model_copies_mappings_and_rejects_invalid_bytes():
    rates = {"simt_fp32": 10}
    hardware = Hardware(rates, 10)
    rates["simt_fp32"] = 1
    assert hardware.rates["simt_fp32"] == 10
    work = {"simt_fp32": 20}
    node = Node("fixed", work=work)
    work["simt_fp32"] = 200
    assert node.work["simt_fp32"] == 20
    for value in (float("nan"), float("inf"), 1.5, True, -1):
        with pytest.raises(ValueError):
            Node("invalid bytes", external_bytes=value)


def test_shared_prefix_pages_and_partial_ranges_are_read_once():
    assert unique_context_tokens([1000, 1000], [[1, 2], [1, 3]]) == 784 + 216 * 2


@pytest.mark.parametrize("length", [784, 785])
def test_mtp_incoming_ranges_keep_physical_position_one(length):
    assert unique_context_tokens(
        [length, length], [[1, 2], [3, 2]], first=1
    ) == 2 * 783 + (length - 784)
    assert unique_context_tokens(
        [length, length], [[1, 2], [1, 3]], first=1
    ) == 783 + 2 * (length - 784)
    assert unique_context_tokens([900, 1000], [[1, 2], [1, 2]]) == 1000
    with pytest.raises(ValueError):
        unique_context_tokens([1000], [[1]])


def test_retention_counts_unified_carveouts_once_with_generous_tmem_credit():
    assert (
        retention_bytes(
            {
                "name": "NVIDIA B200",
                "compute_capability": [10, 0],
                "register_bytes_per_sm": 262144,
                "l2_bytes": 132644864,
                "sm_count": 148,
            }
        )
        == 250249216
    )


@pytest.mark.parametrize(
    "shape,dtype,size",
    [
        ([2, 3], "torch.bfloat16", 999),
        ([2, 3], "torch.int64", 48),
        ([0, 3], "torch.float32", 0),
        ([True, 3], "torch.float32", 12),
    ],
)
def test_invalid_parameter_bytes_and_dtype_fail_before_cost_analysis(
    shape, dtype, size
):
    from oh_my_vllm.performance.coverage import validate_weights

    with pytest.raises(ValueError, match="byte geometry"):
        validate_weights(
            {"target.embedding": dict(shape=shape, dtype=dtype, bytes=size)}
        )


@pytest.mark.parametrize(
    "sm,clock", [(1, 1965e6), (148, 1), (147, 1965e6), (148, 1965), (148, 2e9)]
)
def test_incomplete_b200_or_wrong_clock_units_are_rejected(sm, clock):
    with pytest.raises(ValueError, match="full B200"):
        b200(sm, clock, 0)


@pytest.mark.parametrize("l2", [-1, 0, 132644864 * 2])
def test_wrong_b200_cache_capacity_is_rejected(l2):
    with pytest.raises(ValueError, match="capacity"):
        retention_bytes(
            dict(
                name="NVIDIA B200",
                compute_capability=[10, 0],
                register_bytes_per_sm=262144,
                sm_count=148,
                l2_bytes=l2,
            )
        )
