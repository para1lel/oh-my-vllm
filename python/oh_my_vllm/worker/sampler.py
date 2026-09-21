"""Target sampling and exact verification of deterministic greedy drafts.

Each row is the target distribution conditioned on its preceding draft prefix.
Draw from those distributions, accept matching draft tokens and emit the first
mismatch (or the bonus token). This preserves the target distribution without a
proposal-probability correction because draft proposals are deterministic.
"""

from collections.abc import Sequence

import numpy as np
import torch

from oh_my_vllm.worker.sampling import SamplingParams


def probabilities(
    logits: torch.Tensor,
    params: SamplingParams,
    prompt: Sequence[int],
    generated: Sequence[int],
    drafts: Sequence[int] = (),
    bitmask: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Build distributions for one request's verification rows without mutation.

    Repetition considers prompt and generated history. Presence/frequency consider
    only generated history, including the speculative prefix preceding each row.
    """
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError("one target row is required per draft plus the bonus")
    scores = logits.float().clone()
    vocab = scores.shape[1]
    if vocab == 0:
        raise ValueError("sampling vocabulary cannot be empty")
    histories = (prompt, generated, drafts)
    if any(type(t) is not int or not 0 <= t < vocab for h in histories for t in h):
        raise ValueError("sampling history token is outside the vocabulary")
    prompt_counts = torch.bincount(
        torch.tensor(prompt, device=scores.device, dtype=torch.int64), minlength=vocab
    )
    counts = torch.bincount(
        torch.tensor(generated, device=scores.device, dtype=torch.int64),
        minlength=vocab,
    )
    for row in range(scores.shape[0]):
        current = scores[row]
        if params.repetition_penalty != 1:
            repeated = (prompt_counts + counts) > 0
            penalized = torch.where(
                current > 0,
                current / params.repetition_penalty,
                current * params.repetition_penalty,
            )
            current.copy_(torch.where(repeated, penalized, current))
        current.sub_(counts * params.frequency_penalty)
        current.sub_((counts > 0) * params.presence_penalty)
        if row < len(drafts):
            counts[drafts[row]] += 1
    if bitmask is not None:
        if bitmask.shape != (scores.shape[0], (vocab + 31) // 32):
            raise ValueError("grammar bitmask does not match sampling rows/vocabulary")
        mask = torch.as_tensor(bitmask, device=scores.device, dtype=torch.int32)
        ids = torch.arange(vocab, device=scores.device)
        allowed = ((mask[:, ids // 32] >> (ids % 32)) & 1).bool()
        scores.masked_fill_(~allowed, -torch.inf)
    if params.temperature == 0:
        # Invalid rows after a rejected draft are permitted, but cannot be consumed.
        valid = torch.isfinite(scores).any(-1, keepdim=True) & ~(
            torch.isnan(scores) | torch.isposinf(scores)
        ).any(-1, keepdim=True)
        out = torch.zeros_like(scores)
        return out.scatter_(1, scores.argmax(-1, keepdim=True), 1) * valid
    scores.div_(params.temperature)
    if 0 < params.top_k < vocab:
        threshold = scores.topk(params.top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if params.top_p < 1:
        sorted_scores, indices = scores.sort(dim=-1, descending=True, stable=True)
        sorted_probs = sorted_scores.softmax(-1)
        cumulative = sorted_probs.cumsum(-1)
        # Keep the first token crossing p, and always keep the highest token.
        remove = cumulative - sorted_probs >= params.top_p
        sorted_scores.masked_fill_(remove, -torch.inf)
        scores.scatter_(1, indices, sorted_scores)
    return scores.softmax(-1)


class RequestSampler:
    def __init__(
        self, params: SamplingParams, prompt: Sequence[int], device: str | torch.device
    ) -> None:
        self.params = params
        self.prompt = list(prompt)
        self.generated = []
        self.generator = torch.Generator(device=device)
        if params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(params.seed)

    def sample(
        self,
        logits: torch.Tensor,
        drafts: Sequence[int] = (),
        bitmask: np.ndarray | torch.Tensor | None = None,
    ) -> list[int]:
        probs = probabilities(
            logits, self.params, self.prompt, self.generated, drafts, bitmask
        )
        if self.params.temperature == 0:
            selected = probs.argmax(-1)
        else:
            # Exponential races also handle unreachable all-masked later rows;
            # their NaNs are rejected only if such a row becomes reachable below.
            noise = torch.empty_like(probs).exponential_(generator=self.generator)
            selected = (probs / noise).argmax(-1)
        valid = torch.isfinite(probs).all(-1) & (probs.sum(-1) > 0)
        rows = torch.stack((selected, valid.to(torch.int64)), -1).tolist()
        accepted = []
        for row, (token, is_valid) in enumerate(rows):
            if not is_valid:
                raise RuntimeError("reachable target distribution has no valid token")
            accepted.append(token)
            if row == len(drafts) or token != drafts[row]:
                break
        return accepted

    def commit(self, tokens: Sequence[int]) -> None:
        """Record only tokens retained after EOS/stop/length handling."""
        self.generated.extend(tokens)
