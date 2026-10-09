"""Draft work must match accepted rows, allocated pages, and output budgets."""

from copy import deepcopy

import pytest
from oh_my_vllm.performance.analysis import _validate_draft_step


def draft_step(mode, *, remaining=7):
    target = dict(context=4, fa_block_table=[1])
    outputs = {1: dict(computed=1, kept=1)}
    run = dict(speculative_mode=mode, output_len=remaining + 2)
    if mode == "dspark":
        ops = [
            dict(kind="dspark_inject", rows=1, requests=[1], queries=[1], contexts=[4])
        ]
        if remaining:
            ops.append(
                dict(
                    kind="dspark_backbone",
                    requests=[1],
                    contexts=[5],
                    incoming_contexts=[4],
                    tables=[[1]],
                    anchors=[7],
                    candidates=[[8] * 7],
                    rows_per_request=7,
                    greedy=True,
                )
            )
    else:
        ops = [
            dict(
                kind="mtp_forward",
                requests=[1],
                queries=[1],
                contexts=[5],
                tables=[[1]],
                unique_token_ids_per_request=[[7]],
                persistent=True,
                output_rows=[[0] if remaining else []],
                token_ids_per_request=[[7]],
            )
        ]
        if remaining == 4:
            ops.append(
                dict(
                    kind="mtp_proposal_graph",
                    requests=[1],
                    contexts=[5],
                    tables=[[1]],
                    rows=1,
                    draft_steps=3,
                    tokens=[[8] * 4],
                )
            )
        elif remaining:
            ops.append(dict(kind="mtp_logits", requests=[1], rows=1))
            for step in range(1, remaining):
                ops.extend(
                    [
                        dict(
                            kind="mtp_forward",
                            requests=[1],
                            queries=[1],
                            contexts=[5 + step],
                            tables=[[1]],
                            unique_token_ids_per_request=[[8]],
                            persistent=False,
                            output_rows=[[0]],
                            token_ids_per_request=[[8]],
                        ),
                        dict(kind="mtp_logits", requests=[1], rows=1),
                    ]
                )
    return ops, {1: target}, outputs, {1: 0}, run, {1: 5}


@pytest.mark.parametrize(
    "mode,remaining",
    [("dspark", 7), ("dspark", 0), ("mtp", 4), ("mtp", 3), ("mtp", 1), ("mtp", 0)],
)
def test_legal_draft_work(mode, remaining):
    _validate_draft_step(*draft_step(mode, remaining=remaining))


@pytest.mark.parametrize(
    "field,value",
    [
        ("contexts", [99]),
        ("incoming_contexts", [99]),
        ("rows_per_request", 14),
        ("tables", [[2]]),
        ("candidates", [[8] * 14]),
        ("anchors", [-1]),
    ],
)
def test_inflated_dspark_work_is_rejected(field, value):
    args = draft_step("dspark")
    args[0][1][field] = value
    with pytest.raises(ValueError):
        _validate_draft_step(*args)


@pytest.mark.parametrize("mode", ["mtp", "dspark"])
def test_duplicate_draft_invocation_is_rejected(mode):
    args = draft_step(mode, remaining=4)
    args[0].append(deepcopy(args[0][-1]))
    with pytest.raises(ValueError, match=r"count|budget"):
        _validate_draft_step(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("queries", [9]),
        ("contexts", [9]),
        ("persistent", False),
        ("unique_token_ids_per_request", [[1, 2]]),
    ],
)
def test_inflated_persistent_mtp_work_is_rejected(field, value):
    args = draft_step("mtp", remaining=4)
    args[0][0][field] = value
    with pytest.raises(ValueError):
        _validate_draft_step(*args)


def test_finished_request_cannot_run_a_backbone_or_proposal_graph():
    for mode in ("mtp", "dspark"):
        args = draft_step(mode, remaining=4)
        args[4]["output_len"] = 1
        with pytest.raises(ValueError):
            _validate_draft_step(*args)


def test_mtp_page_boundary_defers_then_restores_one_row():
    args = draft_step("mtp", remaining=0)
    args[1][1]["context"] = 783
    args[-1][1] = 784
    args[0].clear()
    _validate_draft_step(*args)
    assert args[-1][1] == 784
    args[1][1] = dict(context=784, fa_block_table=[1, 2])
    args[0].append(
        dict(
            kind="mtp_forward",
            requests=[1],
            queries=[2],
            contexts=[784],
            tables=[[1, 2]],
            unique_token_ids_per_request=[[7]],
            persistent=True,
            output_rows=[[]],
            token_ids_per_request=[[7, 7]],
        )
    )
    _validate_draft_step(*args)
    assert args[-1][1] == 786


def test_raw_mtp_cannot_override_derived_incoming_context():
    args = draft_step("mtp", remaining=4)
    args[0][0]["incoming_contexts"] = [784]
    with pytest.raises(ValueError, match="derived incoming"):
        _validate_draft_step(*args)
