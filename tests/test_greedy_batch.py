"""Batched greedy verification must match the general request sampler exactly."""

import pytest
import torch
from oh_my_vllm.worker.sampler import RequestSampler, greedy_rows, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_batched_greedy_matches_request_sampling(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cases = [
        ([[3.0, 3.0, 1.0]], []),
        ([[0.0, 3.0, 1.0], [1.0, 0.0, 3.0], [4.0, 0.0, 0.0]], [1, 2]),
        ([[3.0, 0.0, 1.0], [float("nan"), 0.0, 0.0]], [1]),
        ([[0.0, 3.0, 1.0], [float("nan"), 0.0, 0.0]], [1]),
        ([[0.0, float("inf"), 1.0]], []),
        ([[-float("inf")] * 3], []),
        ([[0.0, 1.0, float("nan")]], []),
        ([[0.0, -float("inf"), 1.0]], []),
    ]
    logits = torch.tensor([row for rows, _ in cases for row in rows], device=device)
    batched = greedy_rows(logits)
    offset = 0
    for rows, drafts in cases:
        end = offset + len(rows)
        sampler = RequestSampler(SamplingParams(10, temperature=0), [], device)
        try:
            expected = sampler.sample(logits[offset:end], drafts)
        except RuntimeError:
            with pytest.raises(RuntimeError, match="reachable"):
                verify_rows(batched[offset:end], drafts)
        else:
            assert verify_rows(batched[offset:end], drafts) == expected
        offset = end


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 1},
        {"temperature": 0, "repetition_penalty": 1.1},
        {"temperature": 0, "presence_penalty": 0.1},
        {"temperature": 0, "frequency_penalty": 0.1},
    ],
)
def test_non_plain_sampling_excluded(params):
    assert not RequestSampler(SamplingParams(10, **params), [], "cpu").plain_greedy
