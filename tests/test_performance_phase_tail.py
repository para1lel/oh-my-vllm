"""Contained phase work remains a bound when requests start at different times."""

import pytest
from oh_my_vllm.performance import analysis
from oh_my_vllm.performance.dag import Hardware
from oh_my_vllm.performance.memory import WriteLedger


def setup_model(monkeypatch, cache=0):
    monkeypatch.setattr(analysis, "validate_inventory", lambda trace: None)
    monkeypatch.setattr(analysis, "validate_progress", lambda trace, run: None)
    monkeypatch.setattr(analysis, "retention_bytes", lambda device: cache)
    monkeypatch.setattr(
        analysis,
        "b200",
        lambda *args: Hardware({"tensor_fp8": 1, "tensor_bf16": 1}, 1, cache),
    )

    class NoFinalWrites(WriteLedger):
        # Isolate the resource-interval proof; real physical writes have their
        # separate lifetime and endpoint tests in test_performance_memory.py.
        def bytes(self):
            return 0

    monkeypatch.setattr(analysis, "WriteLedger", NoFinalWrites)

    def one_shared_read(*args, **kwargs):
        return dict(
            work={},
            incoming_ranges={"feedback:shared_weight": 1},
            critical_path_s=0,
            relaxed_native_operations={},
        )

    monkeypatch.setattr(analysis, "step_work", one_shared_read)


def step(ids, *, kept=1, completed=2, extra=None):
    requests = [
        dict(
            request_id=rid,
            context=0,
            query=1,
            drafts=0,
            state_row_writes=[rid],
            mamba_block_table=[rid],
            fa_block_table=[rid],
        )
        for rid in ids
    ]
    operations = [
        dict(kind="target", requests=requests, state_dtype="torch.float32"),
        dict(
            kind="commit",
            outputs=[dict(request_id=r, kept=kept, computed=1) for r in ids],
        ),
    ]
    if extra:
        operations.append(extra)
    return dict(mode="none", operations=operations, completed_operations=completed)


def run(trace, times):
    request_phases = []
    for rid, first_time in times.items():
        indices = [
            i
            for i, s in enumerate(trace["steps"])
            if any(r["request_id"] == rid for r in s["operations"][0]["requests"])
        ]
        request_phases.append(
            dict(
                request_id=rid,
                submitted_s=0,
                first_token_s=first_time,
                first_step=indices[0] + 1,
                last_step=indices[-1] + 1,
            )
        )
    return dict(speculative_mode="none", output_len=2, request_phases=request_phases)


def test_disjoint_phases_do_not_sum_into_longest_request(monkeypatch):
    setup_model(monkeypatch)
    trace = dict(
        device=dict(sm_count=148, max_clock_hz=1965e6),
        weights={},
        steps=[step([1]), step([1]), step([2]), step([2])],
    )
    result = analysis.analyze_trace(trace, [run(trace, {1: 0, 2: 100})])[0]
    model = result["decode_model"]
    assert model["resource_span_s"] == 2
    assert model["skew_adjusted_lower_bound_s"] == 0
    assert model["common_tail_model"]["selected_steps"] == 1
    assert model["common_tail_model"]["first_step_exclusive"] == 2
    assert result["decode"] == 1


@pytest.mark.parametrize("cache,expected", [(0, 1), (1, 0)])
def test_shared_weights_and_prefetched_cache_have_one_credit(
    monkeypatch, cache, expected
):
    setup_model(monkeypatch, cache)
    trace = dict(
        device=dict(sm_count=148, max_clock_hz=1965e6),
        weights={},
        steps=[step([1, 2]), step([1, 2])],
    )
    result = analysis.analyze_trace(trace, [run(trace, {1: 0, 2: 0})])[0]
    assert result["decode"] == expected
    assert (
        result["decode_model"]["common_tail_model"]["compulsory_hbm_bytes"] == expected
    )


def test_uncertified_endpoint_tail_is_removed_before_cost_model(monkeypatch):
    setup_model(monkeypatch)
    trace = dict(
        device=dict(sm_count=148, max_clock_hz=1965e6),
        weights={},
        steps=[step([1]), step([1], extra=dict(kind="uncertified", requests=[1]))],
    )
    called = []

    def inspect_cost(hardware, weights, operations, **kwargs):
        called.append([operation["kind"] for operation in operations])
        return dict(
            work={},
            incoming_ranges={"feedback:w": 1},
            critical_path_s=0,
            relaxed_native_operations={},
        )

    monkeypatch.setattr(analysis, "step_work", inspect_cost)
    analysis.analyze_trace(trace, [run(trace, {1: 0})])
    assert called and all(kinds == ["target"] for kinds in called)


def observed_ledgers(monkeypatch):
    ledgers = []
    base = analysis.WriteLedger

    class Observed(base):
        def __init__(self):
            super().__init__()
            ledgers.append(self)

    monkeypatch.setattr(analysis, "WriteLedger", Observed)
    return ledgers


def test_checkpoint_before_common_cut_is_not_a_tail_write(monkeypatch):
    setup_model(monkeypatch)
    ledgers = observed_ledgers(monkeypatch)
    first, last = step([1]), step([1])
    first["operations"][0]["requests"][0].update(context=782)
    last["operations"][0]["requests"][0].update(context=783)
    trace = dict(
        device=dict(sm_count=148, max_clock_hz=1965e6),
        weights={},
        steps=[first, last, step([2]), step([2])],
    )
    analysis.analyze_trace(trace, [run(trace, {1: 0, 2: 100})])
    decode, tail = ledgers[2:]
    assert decode.checkpoints == {1}
    assert not tail.checkpoints
    assert all(page != 1 for family, page in tail.ranges)


def test_other_request_reassignment_invalidates_tail_created_page(monkeypatch):
    setup_model(monkeypatch)
    ledgers = observed_ledgers(monkeypatch)
    first, middle, last = step([1, 2]), step([1]), step([2, 3])
    a = first["operations"][0]["requests"][0]
    a.update(query=783, state_row_writes=[-1] * 782 + [1])
    first["operations"][1]["outputs"][0]["computed"] = 783
    middle["operations"][0]["requests"][0].update(context=783)
    # Request3 is outside the measured batch. Its post-cut allocation reuses
    # the physical page written by request1's terminal full-page prefix.
    last["operations"][0]["requests"][1]["fa_block_table"] = [1]
    trace = dict(
        device=dict(sm_count=148, max_clock_hz=1965e6),
        weights={},
        steps=[first, middle, last],
    )
    analysis.analyze_trace(trace, [run(trace, {1: 0, 2: 0})])
    tail = ledgers[3]
    assert tail.page_owners[1] == (3, 0)
    assert ("target", 1) not in tail.ranges
