"""Counterexamples for persistent versions, terminal release, and token DAGs."""

from oh_my_vllm.performance.dag import b200
from oh_my_vllm.performance.memory import WriteLedger
from oh_my_vllm.performance.model import Model

KV_BYTES = 16 * 2 * 4 * 256 * 2
STATE_BYTES = 48 * (48 * 128 * 128 * 2 + 10240 * 3 * 2)


def request(
    start,
    query,
    count,
    writes,
    *,
    terminal=False,
    drafts=0,
    rid=1,
    table=(1, 2),
    mamba=(10, 11),
    mtp=False,
):
    return dict(
        context=start,
        query=query,
        kept_rows=count,
        state_row_writes=writes,
        terminal=terminal,
        drafts=drafts,
        request_id=rid,
        fa_block_table=list(table),
        mamba_block_table=list(mamba),
        mtp=mtp,
    )


def test_recycling_physical_page_drops_old_tail_and_all_draft_versions():
    ledger = WriteLedger()
    ledger.tokens("target", [3], 0, 784, 1)
    ledger.tokens("dspark", [3], 0, 784, 1)
    ledger.tokens("mtp", [3], 0, 784, 1)
    ledger.ranges[("mtp_hidden", 3)] = [(0, 1)]
    ledger.tokens("target", [3], 0, 1, 2)
    assert ledger.bytes() == KV_BYTES


def test_terminal_discards_working_state_and_partial_kv_keeps_registered_checkpoint():
    ledger = WriteLedger()
    ledger.target([request(0, 784, 784, [-1] * 783 + [10])], "torch.bfloat16")
    ledger.target([request(784, 1, 1, [11], terminal=True)], "torch.bfloat16")
    assert ledger.working == {}
    assert ledger.checkpoints == {10}
    assert ledger.bytes() == 784 * KV_BYTES + STATE_BYTES


def test_newly_accepted_terminal_draft_boundary_is_not_registered_prefix():
    ledger = WriteLedger()
    ledger.target(
        [request(782, 3, 2, [9, 10, 11], terminal=True, drafts=2)], "torch.bfloat16"
    )
    assert not ledger.checkpoints
    assert ledger.bytes() == 0


def test_rejected_candidate_states_do_not_become_persistent_writes():
    ledger = WriteLedger()
    ledger.target([request(0, 4, 1, [7, 8, 9, 10], drafts=3)], "torch.bfloat16")
    assert ledger.bytes() == KV_BYTES + STATE_BYTES
    ledger.draft([dict(kind="mtp_forward", persistent=False)], [])
    assert ledger.bytes() == KV_BYTES + STATE_BYTES


def test_dspark_append_and_boundary_hidden_have_one_physical_version():
    ledger = WriteLedger()
    r = request(0, 784, 784, [-1] * 783 + [10], mtp=True)
    ledger.target([r], "torch.bfloat16")
    injection = dict(kind="dspark_inject", requests=[1], queries=[784])
    ledger.draft([injection, injection], [r])
    assert (
        ledger.bytes()
        == 784 * (KV_BYTES + 5 * 2 * 8 * 128 * 2) + STATE_BYTES + 5120 * 2
    )


def test_shared_angles_charge_position_union_across_models_and_phase_calls():
    hardware = b200(148, 1965e6, 0)
    seen = set()
    first = Model(hardware, {}, angle_seen=seen)
    first.rope_angles("target", [10, 11, 12], 32)
    first.rope_angles("target", [11, 12, 13], 32)
    assert sum(n.work.get("sfu", 0) for n in first.graph.nodes) == 4 * 32 * 2
    second = Model(hardware, {}, angle_seen=seen)
    second.rope_angles("target", [12, 13, 14], 32)
    assert sum(n.work.get("sfu", 0) for n in second.graph.nodes) == 32 * 2


def test_cached_markov_embedding_lookup_still_depends_on_current_token():
    hardware = b200(148, 1965e6, 0)
    model = Model(hardware, {})
    old = model.node("old sample")
    model.embedding([42], markov=True, parents=(old,))
    current = model.node("current sample")
    lookup = model.embedding([42], markov=True, parents=(current,))
    assert current in model.graph.nodes[lookup].parents


def test_unused_reserved_lookahead_invalidates_previous_cached_checkpoint():
    ledger = WriteLedger()
    ledger.target(
        [request(0, 784, 784, [-1] * 783 + [10], table=(3,))], "torch.bfloat16"
    )
    assert ledger.checkpoints == {10}
    new = request(0, 1, 1, [12], rid=2, table=(4,), mamba=(12, 10))
    new["state_source"] = 0
    ledger.target([new], "torch.bfloat16")
    assert not ledger.checkpoints


def test_reserved_read_only_source_keeps_its_cached_checkpoint():
    ledger = WriteLedger()
    ledger.target(
        [request(0, 784, 784, [-1] * 783 + [10], table=(3,))], "torch.bfloat16"
    )
    new = request(0, 1, 1, [12], rid=2, table=(4,), mamba=(12, 10))
    new["state_source"] = 10
    ledger.target([new], "torch.bfloat16")
    assert ledger.checkpoints == {10}


def test_unfinished_checkpoint_copy_is_excluded_at_token_boundary():
    ledger = WriteLedger()
    r = request(783, 2, 1, [9, 11], drafts=1)
    r["checkpoint_copies_completed"] = False
    ledger.target([r], "torch.bfloat16")
    assert not ledger.checkpoints
    assert ledger.bytes() == KV_BYTES + STATE_BYTES


def test_other_phase_execution_invalidates_old_working_and_checkpoint_versions():
    ledger = WriteLedger()
    r = request(0, 784, 784, [-1] * 783 + [10], table=(3, 4))
    ledger.target([r], "torch.bfloat16")
    assert ledger.checkpoints == {10} and ledger.working == {1: 10}
    later = request(784, 1, 1, [11], table=(3, 4), mamba=(0, 11, 10))
    ledger.invalidate(later)
    assert not ledger.checkpoints and not ledger.working
    assert ledger.bytes() == 784 * KV_BYTES
