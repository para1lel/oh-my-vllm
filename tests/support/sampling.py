import numpy as np
import torch
from oh_my_vllm.worker.sampler import RequestSampler, probabilities, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams


def check_seeded_drafts_mask_partial_commit(device: str) -> None:
    params = SamplingParams(
        10,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        seed=784,
    )
    prompt = [1, 2, 2, 5]
    drafts = [1, 2]
    mask = np.array([[255], [0b01111111], [255]], dtype=np.int32)
    logits = torch.tensor(
        [
            [0, 14, 0, 0, 0, 0, 0, 0],
            [0, 0, 14, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 1, 0, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    sampler = RequestSampler(params, prompt, device, vocab_size=8)
    reference_generator = torch.Generator(device=device).manual_seed(784)
    committed_history = []
    for _ in range(5):
        expected_probs = probabilities(
            logits, params, prompt, committed_history, drafts, mask
        )
        noise = torch.empty_like(expected_probs).exponential_(
            generator=reference_generator
        )
        selected = (expected_probs / noise).argmax(-1).tolist()
        expected = verify_rows([[token, 1] for token in selected], drafts)
        actual = sampler.sample(logits, drafts, mask)
        assert actual == expected
        assert len(actual) == 3
        sampler.commit(actual[:1])
        committed_history.extend(actual[:1])
        assert sampler.generated == committed_history
        assert sampler._history_counts[1].sum().item() == len(committed_history)
