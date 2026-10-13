"""CPU proposal/cache contracts, including causal stopping and exact sampled q."""

from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.worker.batch_plan import PlannedRequest
from oh_my_vllm.worker.protocol import ScheduledRequest
from oh_my_vllm.worker.sampler import RequestSampler, probabilities
from oh_my_vllm.worker.sampling import SamplingParams

from tests.support.dspark import make_worker, proposal


def test_cache_injection_keeps_accepted_inputs_and_does_not_store_noise():
    worker, model = make_worker()
    drafts = proposal(worker, count=1, scheduled=[0, 7, 8, 9, 10, 1, 2, 3])
    features, positions, slots = model.injections[0]
    assert features.shape == (1, 16)
    assert positions.tolist() == [0]
    assert slots.tolist() == [784]
    assert worker.next_position == {1: 1}
    assert len(drafts[1]) == 7
    tokens, positions, _, lengths = model.calls[0]
    assert tokens.tolist() == [[1, 10, 10, 10, 10, 10, 10]]
    assert positions.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    assert lengths.tolist() == [1]


def test_prefix_reuse_only_writes_new_private_page():
    worker, model = make_worker()
    assert proposal(worker, count=784, output=False) == {1: []}
    prefix_positions = model.injections[0][1]
    assert prefix_positions.tolist() == list(range(784))
    proposal(worker, start=784, count=1, pages=[1, 2], rid=2)
    _, positions, slots = model.injections[-1]
    assert positions.tolist() == [784]
    assert slots.tolist() == [1568]
    assert model.calls[-1][2][0, :2].tolist() == [1, 2]


def test_first_confidence_failure_stops_even_if_later_scores_recover():
    worker, model = make_worker(threshold=0.5)
    result = proposal(worker)
    assert result == {1: [3, 4]}
    model.confidence[3:] = [0.0] * 4
    worker.forget(1)
    assert proposal(worker) == result
    model.confidence[0] = 0.1
    worker.forget(1)
    assert proposal(worker) == {1: []}


def test_zero_threshold_keeps_all_seven_and_context_limit_caps_candidates():
    worker, model = make_worker(max_tokens=9)
    model.confidence = [0.0] * 7
    assert len(proposal(worker)[1]) == 6
    worker, _ = make_worker()
    assert len(proposal(worker)[1]) == 7


def test_confidence_cutoff_uses_cumulative_prefix_survival():
    worker, model = make_worker(threshold=0.7)
    model.confidence = [0.8] * 7
    assert proposal(worker) == {1: [3]}
    worker, model = make_worker(threshold=0.7)
    model.confidence = [0.8] * 7
    sampler = RequestSampler(SamplingParams(max_tokens=30, seed=9), [0, 1], "cpu")
    sampler.commit([2])
    assert len(proposal(worker, samplers={1: sampler})[1]) == 1


def test_rejected_suffix_never_injects_across_page_boundary():
    worker, model = make_worker()
    proposal(worker, count=782, output=False)
    history = [position % 11 for position in range(784)]
    drafts = proposal(
        worker,
        start=782,
        count=1,
        scheduled=[1, 2, 3, 4, 5, 6, 7, 8],
        pages=[1, 2],
        history=history,
    )
    assert len(drafts[1]) == 7
    _, positions, slots = model.injections[-1]
    assert positions.tolist() == [782]
    assert slots.tolist() == [1 * 784 + 782]
    assert worker.next_position[1] == 783


def test_completed_and_intermediate_outputs_still_inject_without_proposals():
    for options in ({"output": False}, {"finished": True}):
        worker, model = make_worker()
        assert proposal(worker, **options) == {1: []}
        assert len(model.injections) == 1
        assert not model.calls


@pytest.mark.parametrize(
    "start,count,pages",
    [
        (1, 1, [1]),
        (0, 0, [1]),
        (0, 2, [0]),
        (0, 2, [8]),
        (0, 785, [1]),
    ],
)
def test_invalid_context_state_is_rejected(start, count, pages):
    worker, _ = make_worker()
    request = ScheduledRequest(1, [0] * max(1, count), start, pages, [])
    plan = PlannedRequest(request, 0, [], [], [], False)
    with pytest.raises(ValueError):
        worker.validate_state(plan, count)


def test_stale_request_state_and_forget_are_explicit():
    worker, _ = make_worker()
    proposal(worker)
    with pytest.raises(ValueError, match="position"):
        proposal(worker)
    worker.forget(1)
    assert 1 not in worker.next_position
    assert 1 not in worker.proposals
    assert len(proposal(worker)[1]) == 7


def test_stochastic_q_matches_sampling_filters_history_and_markov_chain():
    worker, _ = make_worker(threshold=0.5)
    params = SamplingParams(
        max_tokens=30,
        temperature=0.7,
        top_k=5,
        top_p=0.85,
        seed=63,
        repetition_penalty=1.1,
        frequency_penalty=0.3,
        presence_penalty=0.2,
    )
    sampler = RequestSampler(params, [0, 1], "cpu", vocab_size=11)
    sampler.commit([2])
    state = sampler.generator.get_state().clone()
    before = list(sampler.generated)
    drafts = proposal(worker, samplers={1: sampler})[1]
    assert len(drafts) == 2
    q = worker.draft_probabilities(1, drafts)
    previous = 2
    prefix = []
    for position, token in enumerate(drafts):
        base = torch.linspace(-1, 1, 11)[None]
        base[0, (previous + 1) % 11] += 3
        expected = probabilities(base, params, sampler.prompt, [*before, *prefix])[0]
        assert torch.equal(q[position], expected)
        assert q[position, token] > 0
        previous = token
        prefix.append(token)
    assert torch.equal(sampler.generator.get_state(), state)
    assert sampler.generated == before
    assert q.device == worker.device and q.dtype == torch.float32
    assert torch.equal(worker.draft_probabilities(1, drafts[:1]), q[:1])
    with pytest.raises(ValueError, match="prefix"):
        worker.draft_probabilities(1, [(drafts[0] + 1) % 11])
    returned = list(drafts)
    drafts[0] = (drafts[0] + 1) % 11
    with pytest.raises(ValueError, match="prefix"):
        worker.draft_probabilities(1, drafts)
    assert torch.equal(worker.draft_probabilities(1, returned), q)
    worker.forget(1)
    with pytest.raises(ValueError, match="unavailable"):
        worker.draft_probabilities(1, drafts)


@pytest.mark.parametrize("guard", [False, True])
def test_proposal_q_applies_cycle_guard_before_recording_each_prefix(guard):
    worker, _ = make_worker()
    params = SamplingParams(
        max_tokens=128, temperature=0.7, seed=9, repetition_guard=guard
    )
    sampler = RequestSampler(params, [0], "cpu", vocab_size=11)
    sampler.commit([0, 1] * 32)
    history = list(sampler.generated)
    rng = sampler.generator.get_state().clone()
    drafts = proposal(worker, count=1, history=[0, 1], samplers={1: sampler})[1]
    assert len(drafts) == 7
    q = worker.draft_probabilities(1, drafts)
    previous = 1
    for position, token in enumerate(drafts):
        scores = torch.linspace(-1, 1, 11)
        scores[(previous + 1) % 11] += 3
        if guard and position == 0:
            # The next 0 would extend 32 copies of [0, 1]. Any other ID breaks
            # that suffix, so none of the remaining seven-step prefixes cycle.
            scores[0] = -torch.inf
            assert token != 0
        torch.testing.assert_close(
            q[position], (scores / params.temperature).softmax(0)
        )
        previous = token
    assert (q[0, 0] == 0).item() == guard
    assert sampler.generated == history
    assert torch.equal(sampler.generator.get_state(), rng)


class Matcher:
    def __init__(self):
        self.accepted = []
        self.rolled_back = []

    def is_terminated(self):
        return len(self.accepted) == 2

    def fill_next_token_bitmask(self, mask, row):
        mask[row].zero_()
        mask[row, 0] = 1 << (3 + len(self.accepted))

    def accept_token(self, token):
        assert token == 3 + len(self.accepted)
        self.accepted.append(token)
        return True

    def rollback(self, count):
        self.rolled_back.append(count)
        del self.accepted[-count:]


def test_grammar_is_temporary_and_is_part_of_the_real_q():
    worker, model = make_worker()
    sampler = RequestSampler(SamplingParams(max_tokens=30, seed=8), [0, 1], "cpu")
    sampler.commit([2])
    matcher = Matcher()
    serving = SimpleNamespace(generations={1: SimpleNamespace(matcher=matcher)})
    drafts = proposal(worker, samplers={1: sampler}, serving=serving)[1]
    assert drafts == [3, 4]
    assert matcher.accepted == []
    assert matcher.rolled_back == [2]
    q = worker.draft_probabilities(1, drafts)
    assert q.shape == (2, 11)
    assert q[0, 3] == 1 and q[1, 4] == 1
    assert int((q != 0).sum()) == 2
    assert len(model.calls) == 1


def test_grammar_is_rolled_back_after_device_or_head_failure():
    worker, model = make_worker()
    matcher = Matcher()
    serving = SimpleNamespace(generations={1: SimpleNamespace(matcher=matcher)})
    original = model.step

    def fail_on_second(hidden, logits, previous):
        if int(hidden[0, 0]) == 1:
            raise RuntimeError("owned device failure")
        return original(hidden, logits, previous)

    worker.step_unit = fail_on_second
    with pytest.raises(RuntimeError, match="owned device failure"):
        proposal(worker, serving=serving)
    assert matcher.accepted == []
    assert matcher.rolled_back == [1]


def test_greedy_mode_does_not_keep_a_vocabulary_probability_matrix():
    worker, _ = make_worker()
    drafts = proposal(worker)[1]
    assert worker.proposals[1].probabilities is None
    with pytest.raises(ValueError, match="unavailable"):
        worker.draft_probabilities(1, drafts)


@pytest.mark.parametrize("remaining", [0, 1, 2, 4, 8])
def test_proposer_reserves_the_scheduler_bonus_output(remaining):
    worker, model = make_worker()
    sampler = RequestSampler(
        SamplingParams(max_tokens=10, temperature=0), [0, 1], "cpu", vocab_size=11
    )
    sampler.commit([2] * (10 - remaining))
    result = proposal(worker, samplers={1: sampler})
    assert len(result[1]) == min(7, max(0, remaining - 1))
    assert len(model.calls) == int(remaining > 1)
