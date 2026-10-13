"""Distribution preservation and failure handling for sampled draft verification."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from oh_my_vllm.worker.model_runner import RuntimeConfig
from oh_my_vllm.worker.sampler import RequestSampler, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams


def sampler():
    return RequestSampler(SamplingParams(10, temperature=1, seed=784), [], "cpu")


def test_rejection_emits_residual_and_discards_suffix():
    target = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    proposal = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    with patch("torch.rand", return_value=torch.tensor([0.5, 0.5])):
        rows = sampler().draw_speculative_rows(target.log(), [0, 1], proposal)
    assert verify_rows(rows.tolist(), [0, 1]) == [1]


def test_equal_distributions_accept_all_drafts_and_sample_bonus():
    target = torch.tensor([[0.3, 0.7], [0.8, 0.2], [0.0, 1.0]])
    rows = sampler().draw_speculative_rows(target.log(), [1, 0], target[:-1])
    assert verify_rows(rows.tolist(), [1, 0]) == [1, 0, 1]


def test_actual_output_distribution_matches_target():
    target = torch.tensor([[0.22, 0.78], [0.6, 0.4]])
    proposal = torch.tensor([[0.8, 0.2]])
    verifier = sampler()
    proposals = torch.multinomial(
        proposal[0], 4000, replacement=True, generator=torch.Generator().manual_seed(42)
    ).tolist()
    zeros = 0
    for token in proposals:
        rows = verifier.draw_speculative_rows(target.log(), [token], proposal)
        zeros += verify_rows(rows.tolist(), [token])[0] == 0
    assert abs(zeros / len(proposals) - 0.22) < 0.03


def test_grammar_mask_applies_to_residual_distribution():
    target = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    proposal = torch.tensor([[0.8, 0.2]])
    mask = np.array([[2], [2]], dtype=np.int32)
    rows = sampler().draw_speculative_rows(target.log(), [0], proposal, mask)
    assert verify_rows(rows.tolist(), [0]) == [1]


def test_seed_reproduces_the_same_verification_algorithm():
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    proposal = torch.tensor([[0.9, 0.1]])
    assert torch.equal(
        sampler().draw_speculative_rows(logits, [0], proposal),
        sampler().draw_speculative_rows(logits, [0], proposal),
    )


@pytest.mark.parametrize(
    "proposal", [[[1.0, 1.0]], [[0.0, 1.0]], [[float("nan"), 0.0]]]
)
def test_invalid_reachable_proposal_is_rejected(proposal):
    rows = sampler().draw_speculative_rows(
        torch.zeros(2, 2), [0], torch.tensor(proposal)
    )
    with pytest.raises(RuntimeError, match="no valid token"):
        verify_rows(rows.tolist(), [0])


def test_probability_shape_mismatch_fails_before_sampling():
    with pytest.raises(ValueError, match="match the candidate"):
        sampler().draw_speculative_rows(torch.zeros(2, 2), [0], torch.zeros(2, 2))


def test_dspark_missing_checkpoint_fails_before_device_initialization():
    with pytest.raises(ValueError, match="draft checkpoint"):
        RuntimeConfig("target", speculative_tokens=7, speculative_mode="dspark")
    with pytest.raises(ValueError, match="count disagrees"):
        RuntimeConfig("target", speculative_tokens=7)
