"""Periodic output constraints apply before truncation and speculative selection."""

import numpy as np
import pytest
import torch
from oh_my_vllm.worker.repetition import cycle_continuations
from oh_my_vllm.worker.sampler import RequestSampler, probabilities
from oh_my_vllm.worker.sampling import SamplingParams


def check_cycle_guard_distributions(device):
    history = [0, 1] * 32
    logits = torch.tensor([[1000.0, 0, 1, -1]], device=device)
    params = SamplingParams(128, top_k=1, top_p=0.5)
    actual = probabilities(logits, params, [], history)
    torch.testing.assert_close(actual, torch.tensor([[0.0, 0, 1, 0]], device=device))
    # A required grammar continuation takes precedence over the cycle guard.
    mask = np.array([[1]], dtype=np.int32)
    forced = probabilities(logits, params, [], history, bitmask=mask)
    torch.testing.assert_close(forced, torch.tensor([[1.0, 0, 0, 0]], device=device))
    torch.testing.assert_close(
        logits, torch.tensor([[1000.0, 0, 1, -1]], device=device)
    )


def test_cycle_guard_runs_before_top_k_and_preserves_forced_grammar():
    check_cycle_guard_distributions("cpu")


def test_default_guard_uses_committed_history_with_all_penalties_disabled():
    sampler = RequestSampler(SamplingParams(128, temperature=0), [], "cpu")
    sampler.commit([0, 1] * 32)
    assert sampler.sample(torch.tensor([[1000.0, 0, 1]])) == [2]
    assert sampler.generated == [0, 1] * 32


def reference_continuations(history):
    tail = history[-768:]
    found = set()
    for period in range(1, min(256, len(tail) // 3) + 1):
        span = max(3 * period, 64)
        suffix = tail[-span:]
        if len(tail) >= span and all(
            suffix[index] == suffix[index + period] for index in range(span - period)
        ):
            found.add(tail[-period])
    return found


def test_period_detection_matches_an_independent_definition_at_bounds_and_phases():
    cases = [[], [0] * 63, [0] * 64, [0] * 10000]
    for period in (2, 3, 64, 128, 256, 257):
        pattern = list(range(period))
        for length in (63, 64, 65, 191, 192, 767, 768, 769, 1024):
            for phase in (0, 1):
                cases.append((pattern * 600)[phase : phase + length])
    rng = np.random.default_rng(784)
    cases.extend(rng.integers(0, 4, size=length).tolist() for length in range(1024))
    for history in cases:
        assert cycle_continuations(history) == reference_continuations(history)


def test_distinct_overlapping_cycles_are_suppressed_jointly_or_preserved_jointly():
    history = ([0] * 64 + [1] * 64) * 3
    assert cycle_continuations(history) == {0, 1}
    logits = torch.tensor([[1000.0, 999, 998]])
    params = SamplingParams(128)
    actual = probabilities(logits, params, [], history)
    torch.testing.assert_close(actual, torch.tensor([[0.0, 0, 1]]))
    mask = np.array([[0b011]], dtype=np.int32)
    forced = probabilities(logits, params, [], history, bitmask=mask)
    expected = torch.tensor([[1000.0, 999, -torch.inf]]).softmax(-1)
    torch.testing.assert_close(forced, expected)


def test_prompt_nonconsecutive_repetition_and_explicit_disable_are_unchanged():
    params = SamplingParams(128, temperature=0)
    logits = torch.tensor([[1000.0, 0, 1]])
    expected = torch.tensor([[1.0, 0, 0]])
    torch.testing.assert_close(probabilities(logits, params, [0, 1] * 64, []), expected)
    torch.testing.assert_close(
        probabilities(logits, params, [], [0, 1] * 32 + [2]), expected
    )
    torch.testing.assert_close(
        probabilities(
            logits,
            SamplingParams(128, temperature=0, repetition_guard=False),
            [],
            [0, 1] * 32,
        ),
        expected,
    )
    assert not RequestSampler(params, [], "cpu").plain_greedy
    assert RequestSampler(
        SamplingParams(128, temperature=0, repetition_guard=False), [], "cpu"
    ).plain_greedy


def check_cycle_guard_speculative_prefixes(device):
    history = [0, 1] * 31 + [0]
    logits = torch.tensor(
        [[0.0, 1000, 1], [1000.0, 0, 1], [0.0, 1000, 1]], device=device
    )
    drafts = [1, 0]
    params = SamplingParams(128, temperature=0)
    combined = probabilities(logits, params, [], history, drafts)
    for row in range(3):
        expected = probabilities(
            logits[row : row + 1], params, [], history + drafts[:row]
        )
        torch.testing.assert_close(combined[row : row + 1], expected)
    sampler = RequestSampler(params, [], device)
    sampler.commit(history)
    assert sampler.sample(logits, drafts) == [1, 2]
    assert sampler.generated == history
    # An unguarded proposal must be rejected against the guarded target PMF.
    q = torch.tensor([[0.0, 1, 0], [1.0, 0, 0]], device=device)
    rows = sampler.draw_speculative_rows(logits, drafts, q).tolist()
    assert rows[0] == [1, 1]
    assert rows[1] == [2, 1]
    assert sampler.generated == history
    sampler.commit([1, 2])
    assert sampler.sample(torch.tensor([[1000.0, 0, 1]], device=device)) == [0]


def test_mtp_and_dspark_target_use_each_prefix_without_mutating_committed_history():
    check_cycle_guard_speculative_prefixes("cpu")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_guard_does_not_hide_invalid_logits_or_all_masked_rows(invalid):
    sampler = RequestSampler(SamplingParams(128, temperature=0), [], "cpu")
    sampler.commit([0, 1] * 32)
    logits = torch.tensor([[invalid, -torch.inf, -torch.inf]])
    with pytest.raises(RuntimeError, match="reachable"):
        sampler.sample(logits)


def test_guard_only_history_work_is_bounded_and_has_no_penalty_count_allocation():
    sampler = RequestSampler(SamplingParams(128), [0] * 262144, "cpu")
    sampler.commit([0] * 32768)
    prompt, generated = sampler.sampling_history()
    assert not prompt
    assert len(generated) == 768
    assert sampler._history_counts is None
    assert cycle_continuations(generated) == {0}


@pytest.mark.parametrize("value", [0, 1, None, "true", []])
def test_guard_configuration_is_strictly_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        SamplingParams(128, repetition_guard=value)
