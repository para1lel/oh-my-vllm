"""Phase aggregation and the exact fifteen-row acceptance statistics."""

import copy

import pytest

from benchmarks.framework import WORKLOADS, phase_times, phase_verdict


def test_fifteen_fixed_workloads_keep_synthetic_lengths_and_prefix_scope():
    assert len(WORKLOADS) == 15
    assert len({(w["mode"], w["input_len"], w["batch_size"]) for w in WORKLOADS}) == 15
    assert {w["output_len"] for w in WORKLOADS} == {4096}
    assert {w["batch_size"] for w in WORKLOADS} == {1, 2, 4}
    assert [w["input_len"] for w in WORKLOADS if w["mode"] == "dspark"] == [32768] * 3


def run():
    return {
        "batch_size": 2,
        "input_len": 32768,
        "output_len": 4096,
        "speculative_mode": "none",
        "output_tokens": 8192,
        "preemptions": 0,
        "initial_prefix_hit_tokens": 0,
        "elapsed_s": 15.5,
        "steps": 11,
        "request_phases": [
            {
                "request_id": 1,
                "submitted_s": 0,
                "first_token_s": 2,
                "last_token_s": 15,
                "first_step": 1,
                "last_step": 10,
            },
            {
                "request_id": 2,
                "submitted_s": 0,
                "first_token_s": 6,
                "last_token_s": 14,
                "first_step": 3,
                "last_step": 9,
            },
        ],
    }


def test_mixed_phases_use_own_last_minus_own_first_and_exclude_cleanup():
    assert phase_times(
        run(),
        {"batch_size": 2, "mode": "ordinary", "output_len": 4096, "input_len": 32768},
    ) == {
        "prefill": 6,
        "decode": 13,
    }
    # Whole batch minus maximum TTFT is 9.5, so it would hide the slow decode.


def test_ratio_of_medians_is_distinct_from_median_of_ratios():
    verdict = phase_verdict([2.9, 2.95, 3, 3.05, 3.1], [1, 100, 1, 100, 1])
    assert verdict["latency_ratio"] == 3
    assert verdict["passed"]
    assert not phase_verdict([3.01] * 5, [1] * 5)["passed"]


def test_wall_spread_is_a_phase_gate_with_exact_boundary():
    assert phase_verdict([2.9, 3, 3, 3, 3.1], [2] * 5)["passed"]
    assert not phase_verdict([2.8, 3, 3, 3, 3.2], [2] * 5)["passed"]
    with pytest.raises(ValueError, match="five"):
        phase_verdict([1] * 4, [1] * 4)


@pytest.mark.parametrize("mode", ["ordinary", "mtp4", "prefix", "dspark"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_only_prefix_prefill_reports_ratio_without_a_latency_gate(mode, phase):
    verdict = phase_verdict([4] * 5, [1] * 5, mode=mode, phase=phase)
    exempt = mode == "prefix" and phase == "prefill"
    assert verdict["latency_ratio"] == 4
    assert not verdict["latency_passed"]
    assert verdict["latency_gate"] is not exempt
    assert verdict["spread_passed"]
    assert verdict["passed"] is exempt


def test_prefix_prefill_spread_and_complete_measurement_gates_stay_active():
    args = {"mode": "prefix", "phase": "prefill"}
    verdict = phase_verdict([3.7, 4, 4, 4, 4.3], [1] * 5, **args)
    assert not verdict["latency_gate"]
    assert not verdict["spread_passed"]
    assert not verdict["passed"]
    with pytest.raises(ValueError, match="five"):
        phase_verdict([4] * 4, [1] * 4, **args)
    with pytest.raises(ValueError, match="invalid"):
        phase_verdict([4] * 5, [0] * 5, **args)


@pytest.mark.parametrize("args", [{"mode": "unknown"}, {"phase": "cleanup"}])
def test_unknown_gate_scope_is_rejected(args):
    with pytest.raises(ValueError, match="unknown"):
        phase_verdict([4] * 5, [1] * 5, **args)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "last", "partial", "preempt"]
)
def test_missing_or_corrupt_phase_evidence_blocks_acceptance(corruption):
    r = copy.deepcopy(run())
    if corruption == "duplicate":
        r["request_phases"][1]["request_id"] = 1
    elif corruption == "missing":
        r["request_phases"].pop()
    elif corruption == "last":
        r["request_phases"][0]["last_token_s"] = float("nan")
    elif corruption == "partial":
        r["output_tokens"] -= 1
    else:
        r["preemptions"] = 1
    with pytest.raises(ValueError):
        phase_times(
            r,
            {
                "batch_size": 2,
                "mode": "ordinary",
                "output_len": 4096,
                "input_len": 32768,
            },
        )


@pytest.mark.parametrize(
    "field", ["input_len", "output_len", "batch_size", "speculative_mode"]
)
def test_mislabeled_runtime_workload_is_rejected(field):
    r = run()
    r[field] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        phase_times(
            r,
            {
                "batch_size": 2,
                "mode": "ordinary",
                "output_len": 4096,
                "input_len": 32768,
            },
        )


def test_gpu_endpoint_tail_is_trimmed_per_request_without_added_sync():
    from oh_my_vllm.performance.analysis import phase_operations

    target = {"kind": "target", "requests": [{"request_id": 1}, {"request_id": 2}]}
    commit = {"kind": "commit", "outputs": []}
    inject = {
        "kind": "dspark_inject",
        "requests": [1, 2],
        "queries": [1, 1],
        "contexts": [10, 20],
        "rows": 2,
    }
    step = {"operations": [target, commit, inject], "completed_operations": 1}
    boundaries = {1: (0, 0, 0), 2: (0, 0, 1)}
    selected = phase_operations(step, {1, 2}, boundaries, 0, "decode")
    assert selected[0]["requests"] == target["requests"]
    assert selected[1]["requests"] == [2]
    assert selected[1]["contexts"] == [20]
    step["completed_operations"] = 3
    assert phase_operations(step, {1, 2}, boundaries, 0, "decode")[1]["requests"] == [
        1,
        2,
    ]
    del step["completed_operations"]
    with pytest.raises(ValueError, match="feedback"):
        phase_operations(step, {1, 2}, boundaries, 0, "decode")
    assert phase_operations(step, {2}, boundaries, 0, "decode")[1]["requests"] == [2]


def test_plain_greedy_readback_certifies_only_already_enqueued_operations():
    import torch
    from oh_my_vllm.performance import execution
    from oh_my_vllm.worker.sampler import greedy_rows

    before = execution._CURRENT
    try:
        execution._CURRENT = {"operations": [{"kind": "target"}]}
        assert greedy_rows(torch.tensor([[0.0, 1.0]])) == [[1, 1]]
        assert execution._CURRENT["completed_operations"] == 1
        execution.record("dspark_inject")
        assert execution._CURRENT["completed_operations"] == 1
    finally:
        execution._CURRENT = before
