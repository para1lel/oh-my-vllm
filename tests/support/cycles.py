import numpy as np
import torch
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
