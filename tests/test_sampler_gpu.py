"""GPU sampling cache and local top-p sort retain the full-sort distribution."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from oh_my_vllm.worker.model_runner import OhMyVllmWorker
from oh_my_vllm.worker.sampler import RequestSampler, probabilities, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams

from tests.test_independent_sampler import check_seeded_drafts_mask_partial_commit

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seeded_drafts_mask_and_partial_commit_on_gpu():
    check_seeded_drafts_mask_partial_commit("cuda")


def full_sort_top_p(
    logits: torch.Tensor, top_k: int, top_p: float, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scores = logits.float().clone()
    if mask is not None:
        ids = torch.arange(scores.shape[1], device=scores.device)
        allowed = ((mask[:, ids // 32] >> (ids % 32)) & 1).bool()
        scores.masked_fill_(~allowed, -torch.inf)
    threshold = scores.topk(top_k, dim=-1).values[:, -1:]
    scores.masked_fill_(scores < threshold, -torch.inf)
    sorted_scores, indices = scores.sort(dim=-1, descending=True, stable=True)
    sorted_probs = sorted_scores.softmax(-1)
    remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
    sorted_scores.masked_fill_(remove, -torch.inf)
    scores.scatter_(1, indices, sorted_scores)
    return scores.softmax(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_top_k_top_p_gpu_matches_full_sort_with_ties_and_invalid_rows():
    generator = torch.Generator(device="cuda").manual_seed(784)
    ordinary = torch.randn(1, 248320, device="cuda", generator=generator)
    tied = torch.full((1, 128), -100.0, device="cuda")
    tied[0, :49] = torch.arange(49, 0, -1, device="cuda")
    tied[0, 49:51] = 0
    masked = torch.tensor([[0.0, 3, 2, 1, 0, -1, -2, -3]], device="cuda")
    for logits, k, mask in (
        (ordinary, 50, None),
        (tied, 50, None),
        (
            torch.tensor([[0.0, float("nan"), 2, 1, 0, -1, -2, -3]], device="cuda"),
            2,
            None,
        ),
        (
            torch.tensor([[0.0, float("inf"), 2, 1, 0, -1, -2, -3]], device="cuda"),
            2,
            None,
        ),
        (torch.full((1, 8), -torch.inf, device="cuda"), 2, None),
        (masked, 2, torch.tensor([[0]], device="cuda", dtype=torch.int32)),
        (masked, 2, torch.tensor([[0b00101101]], device="cuda", dtype=torch.int32)),
    ):
        actual = probabilities(
            logits, SamplingParams(10, top_k=k, top_p=0.8), [], [], bitmask=mask
        )
        expected = full_sort_top_p(logits, k, 0.8, mask)
        torch.testing.assert_close(
            actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cached_gpu_penalties_preserve_seeded_samples_after_commit():
    params = SamplingParams(
        20,
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        repetition_penalty=1.2,
        frequency_penalty=0.2,
        presence_penalty=-0.1,
        seed=784,
    )
    prompt = [0, 1, 1, 2] * 128
    sampler = RequestSampler(params, prompt, "cuda", vocab_size=64)
    reference_generator = torch.Generator(device="cuda").manual_seed(784)
    logits = torch.randn(
        1, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)
    )
    generated = []
    for _ in range(5):
        expected_probs = probabilities(logits, params, prompt, generated)
        cached_probs = probabilities(
            logits,
            params,
            prompt,
            generated,
            history_counts=sampler._history_counts,
        )
        torch.testing.assert_close(cached_probs, expected_probs, rtol=1e-6, atol=1e-8)
        noise = torch.empty_like(expected_probs).exponential_(
            generator=reference_generator
        )
        expected = verify_rows([[int((expected_probs / noise).argmax()), 1]], [])
        actual = sampler.sample(logits)
        assert actual == expected
        sampler.commit(actual)
        generated.extend(actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_gpu_draw_reads_once_and_matches_seeded_requests():
    params = SamplingParams(10, temperature=0.8, top_k=4, seed=784)
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.serving = object()
    worker.samplers = {rid: RequestSampler(params, [], "cuda") for rid in (1, 2, 3)}
    reference = {rid: RequestSampler(params, [], "cuda") for rid in (1, 2, 3)}
    logits = torch.randn(
        6, 32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)
    )
    plans = [
        SimpleNamespace(
            request=SimpleNamespace(request_id=rid),
            sample_indices=list(range(count)),
            drafts=drafts,
        )
        for rid, count, drafts in ((1, 2, [1]), (2, 1, []), (3, 3, [2, 3]))
    ]
    expected = [
        reference[1].sample(logits[:2], [1]),
        reference[2].sample(logits[2:3]),
        reference[3].sample(logits[3:], [2, 3]),
    ]
    original_tolist = torch.Tensor.tolist
    readbacks = []

    def observed_tolist(tensor):
        readbacks.append(tuple(tensor.shape))
        return original_tolist(tensor)

    with patch.object(torch.Tensor, "tolist", observed_tolist):
        sampled, errors = worker._draw_batch_rows(plans, logits, None, {}, {})
    assert errors == {}
    assert readbacks == [(6, 2)]
    assert [verify_rows(sampled[i], plan.drafts) for i, plan in enumerate(plans)] == (
        expected
    )
