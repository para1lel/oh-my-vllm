"""Budget masks use committed history and each speculative target prefix."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from oh_my_vllm.worker.sampler import RequestSampler, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams
from oh_my_vllm.worker.serving import Generation, ServingAdapter


def adapter(budget, *, count=0, end=31):
    instance = ServingAdapter.__new__(ServingAdapter)
    instance.vocab_size, instance._mask = 64, None
    generation = Generation(
        None,
        32,
        [],
        {63},
        reasoning_end=end,
        reasoning_tokens=count,
        thinking_token_budget=budget,
    )
    generation.decoder = SimpleNamespace(step=lambda backend, token: str(token))
    instance.generations = {1: generation}
    return instance


def allowed(row):
    return {token for token in range(64) if (int(row[token // 32]) >> (token % 32)) & 1}


@pytest.mark.parametrize("draft_count", [0, 1, 4, 7])
@pytest.mark.parametrize("budget", [0, 1, 3, 8])
def test_each_verification_and_bonus_row_respects_budget(draft_count, budget):
    instance = adapter(budget)
    # A draft that continues reasoning at the boundary must be rejected there.
    result = instance.masks({1: [5] * draft_count})
    if budget > draft_count:
        assert result is None  # Existing fast path stays available.
        return
    mask = result.grammar_bitmask.copy()
    assert allowed(mask[budget]) == {31}
    assert all(allowed(row) == set(range(64)) for row in mask[:budget])
    assert all(allowed(row) == set(range(64)) for row in mask[budget + 1 :])
    assert instance.generations[1].reasoning_tokens == 0
    assert instance.generations[1].reasoning_end == 31
    assert np.array_equal(mask, instance.masks({1: [5] * draft_count}).grammar_bitmask)


def test_accepted_end_in_drafts_restores_answer_and_bonus_distribution():
    instance = adapter(2, count=1)
    mask = instance.masks({1: [5, 31, 7]}).grammar_bitmask
    assert allowed(mask[1]) == {31}
    assert allowed(mask[2]) == allowed(mask[3]) == set(range(64))
    assert instance.generations[1].reasoning_tokens == 1


def test_natural_early_end_and_disabled_thinking_do_not_force_later_rows():
    instance = adapter(2)
    mask = instance.masks({1: [31, 5, 5]}).grammar_bitmask
    assert all(allowed(row) == set(range(64)) for row in mask)
    assert adapter(None, count=100).masks({1: [5] * 7}) is None
    assert adapter(0, end=None).masks({1: [5] * 7}) is None


def test_mixed_batch_keeps_budget_state_per_request():
    instance = adapter(0)
    instance.generations[2] = adapter(8).generations[1]
    result = instance.masks({2: [5] * 4, 1: []})
    assert result.structured_output_request_ids == ["1"]
    assert allowed(result.grammar_bitmask[0]) == {31}
    assert instance.generations[2].reasoning_tokens == 0


@pytest.mark.parametrize("stochastic", [False, True])
def test_wrong_mtp_draft_is_rejected_at_forced_boundary(stochastic):
    instance = adapter(1)
    drafts = [5, 6, 7, 8]
    mask = instance.masks({1: drafts}).grammar_bitmask.copy()
    logits = torch.full((5, 64), -100.0)
    logits[0, 5], logits[1:, 6] = 100.0, 100.0
    sampler = RequestSampler(
        SamplingParams(max_tokens=32, temperature=1 if stochastic else 0, seed=42),
        [],
        "cpu",
    )
    accepted = sampler.sample(logits, drafts, mask)
    assert accepted == [5, 31]
    generation = instance.generations[1]
    kept, _ = generation.consume(accepted, SimpleNamespace(backend_tokenizer=None))
    sampler.commit(kept)
    assert kept == sampler.generated == [5, 31]
    assert generation.reasoning_tokens == 2  # Existing usage includes the marker.
    assert generation.reasoning_end is None
    assert instance.masks({1: []}) is None


@pytest.mark.parametrize("draft", [5, 31])
def test_dspark_unconstrained_proposal_yields_forced_target_end(draft):
    instance = adapter(0)
    mask = instance.masks({1: [draft]}).grammar_bitmask.copy()
    sampler = RequestSampler(SamplingParams(max_tokens=32, seed=42), [], "cpu")
    logits = torch.zeros((2, 64))
    logits[1, 7] = 100
    proposal = torch.zeros((1, 64))
    proposal[0, 5], proposal[0, 31] = 0.75, 0.25
    rows = sampler.draw_speculative_rows(logits, [draft], proposal, mask).tolist()
    tokens = verify_rows(rows, [draft])
    assert tokens[0] == 31
    if draft == 5:
        assert tokens == [31]  # Rejection residual is the forced singleton.
    else:
        assert tokens == [31, 7]  # Accepted end and normal bonus answer.


def test_total_output_limit_wins_and_consume_detects_a_bypassed_mask():
    instance = adapter(8)
    generation = instance.generations[1]
    generation.max_tokens = 1
    kept, _ = generation.consume([5, 6], SimpleNamespace(backend_tokenizer=None))
    assert kept == [5] and generation.finished == "length"
    generation = adapter(0).generations[1]
    with pytest.raises(RuntimeError, match="thinking token budget"):
        generation.consume([5], SimpleNamespace(backend_tokenizer=None))
    assert generation.ids == [] and generation.reasoning_tokens == 0


def test_dspark_rejection_handles_rounded_singleton_proposal_mass():
    instance = adapter(0)
    mask = instance.masks({1: [5]}).grammar_bitmask.copy()
    proposal = torch.zeros((1, 64))
    proposal[0, 31], proposal[0, 5] = 1, 1e-9
    sampler = RequestSampler(SamplingParams(max_tokens=32, seed=42), [], "cpu")
    rows = sampler.draw_speculative_rows(torch.zeros((2, 64)), [5], proposal, mask)
    assert verify_rows(rows.tolist(), [5]) == [31]
    proposal[0, 5] = -1e-9
    rows = sampler.draw_speculative_rows(torch.zeros((2, 64)), [5], proposal, mask)
    with pytest.raises(RuntimeError, match="no valid token"):
        verify_rows(rows.tolist(), [5])
