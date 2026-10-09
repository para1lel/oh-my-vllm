"""CPU proposal/cache contracts, including causal stopping and exact sampled q."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from oh_my_vllm.models.dspark import DSparkConfig
from oh_my_vllm.worker.batch_plan import PlannedRequest
from oh_my_vllm.worker.dspark import DSpark
from oh_my_vllm.worker.protocol import RequestOutput, ScheduledRequest
from oh_my_vllm.worker.sampler import RequestSampler, probabilities
from oh_my_vllm.worker.sampling import SamplingParams


class TinyDraft:
    def __init__(self):
        self.config = DSparkConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=11,
            markov_rank=2,
            mask_token_id=10,
            target_layer_ids=(0, 1),
        )
        self.injections = []
        self.calls = []
        self.confidence = [0.9, 0.8, 0.1, 0.95, 0.95, 0.95, 0.95]

    def inject(self, features, positions, slots, caches):
        self.injections.append((features.clone(), positions.clone(), slots.clone()))

    def greedy(self, tokens, positions, tables, lengths, caches):
        self.calls.append(
            (tokens.clone(), positions.clone(), tables.clone(), lengths.clone())
        )
        candidates = (tokens[:, :1] + torch.arange(1, 8)[None, :]) % 11
        confidence = torch.tensor(self.confidence).expand(len(tokens), -1)
        return candidates, confidence, torch.ones_like(candidates, dtype=torch.bool)

    def backbone(self, tokens, positions, tables, lengths, caches):
        self.calls.append(
            (tokens.clone(), positions.clone(), tables.clone(), lengths.clone())
        )
        hidden = torch.arange(7).view(1, 7, 1).expand(len(tokens), 7, 8).bfloat16()
        base = torch.linspace(-1, 1, 11).view(1, 1, 11).expand(len(tokens), 7, 11)
        return hidden, base

    def step(self, hidden, base, previous):
        position = int(hidden[0, 0])
        logits = base.clone()
        logits[0, (int(previous[0]) + 1) % 11] += 3
        return logits, torch.tensor([self.confidence[position]])


def make_worker(*, threshold=0, max_tokens=4096):
    model = TinyDraft()
    target = SimpleNamespace(embedding=torch.zeros(11, 8, dtype=torch.bfloat16))
    with patch("oh_my_vllm.worker.dspark.DSparkModel", return_value=model):
        worker = DSpark(target, ".", 8, max_tokens, confidence_threshold=threshold)
    return worker, model


def proposal(
    worker,
    *,
    start=0,
    count=2,
    scheduled=None,
    pages=None,
    rid=1,
    output=True,
    finished=False,
    samplers=None,
    serving=None,
    history=None,
):
    history = history or [position % 11 for position in range(start + count + 1)]
    tokens = history[start : start + count] if scheduled is None else scheduled
    request = ScheduledRequest(rid, tokens, start, pages or [1], [])
    plan = PlannedRequest(request, 0, [], [], [], False)
    features = torch.arange(len(tokens) * 16).view(-1, 16).bfloat16()
    result = RequestOutput(
        rid,
        [history[start + count]] if output else [],
        finish_reason="length" if finished else None,
    )
    return worker.propose(
        [plan],
        [0],
        [count],
        features,
        {rid: history},
        [result],
        samplers=samplers,
        serving=serving,
    )


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
