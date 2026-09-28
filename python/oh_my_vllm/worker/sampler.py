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
    history_counts: tuple[torch.Tensor, torch.Tensor] | None = None,
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
    histories = (prompt, generated, drafts) if history_counts is None else (drafts,)
    if any(type(t) is not int or not 0 <= t < vocab for h in histories for t in h):
        raise ValueError("sampling history token is outside the vocabulary")
    if (
        params.repetition_penalty != 1
        or params.frequency_penalty != 0
        or params.presence_penalty != 0
    ):
        if history_counts is None:
            prompt_counts = torch.bincount(
                torch.tensor(prompt, device=scores.device, dtype=torch.int64),
                minlength=vocab,
            )
            counts = torch.bincount(
                torch.tensor(generated, device=scores.device, dtype=torch.int64),
                minlength=vocab,
            )
        else:
            prompt_counts, counts = history_counts
            if any(
                count.shape != (vocab,)
                or count.device != scores.device
                or count.dtype != torch.int64
                for count in history_counts
            ):
                raise ValueError("cached sampling counts do not match logits")
            if drafts:
                counts = counts.clone()
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
    small_topk = params.top_p < 1 and 0 < params.top_k < vocab // 2
    top_values = top_indices = None
    if 0 < params.top_k < vocab:
        if small_topk:
            top_values, top_indices = scores.topk(params.top_k + 1, dim=-1)
            threshold = top_values[:, params.top_k - 1 : params.top_k]
        else:
            threshold = scores.topk(params.top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if params.top_p < 1:
        # A tied kth threshold preserves more than k tokens. Fall back to the
        # full stable sort so the original token-ID tie order stays exact.
        # The scalar check synchronizes CUDA; the common unique-cutoff path
        # still avoids sorting the full vocabulary.
        unique_cutoff = small_topk and not bool(
            (top_values[:, params.top_k] == threshold[:, 0]).any()
        )
        if unique_cutoff:
            selected = top_indices[:, : params.top_k]
            selected = selected.sort(dim=-1, stable=True).values
            values = scores.gather(1, selected)
            sorted_scores, order = values.sort(dim=-1, descending=True, stable=True)
            indices = selected.gather(1, order)
        else:
            sorted_scores, indices = scores.sort(dim=-1, descending=True, stable=True)
        sorted_probs = sorted_scores.softmax(-1)
        cumulative = sorted_probs.cumsum(-1)
        # Keep the first token crossing p, and always keep the highest token.
        remove = cumulative - sorted_probs >= params.top_p
        sorted_scores.masked_fill_(remove, -torch.inf)
        scores.scatter_(1, indices, sorted_scores)
    return scores.softmax(-1)


def greedy_rows(logits: torch.Tensor) -> list[list[int]]:
    """One batched device-to-host transfer for unpenalized, unmasked greedy rows.

    max propagates NaN and picks the first tied index. A finite maximum also
    rejects +inf and all-masked rows without allocating a vocabulary-sized PMF.
    """
    if logits.ndim != 2 or logits.numel() == 0:
        raise ValueError("greedy logits must be a nonempty matrix")
    values, tokens = logits.max(-1)
    return torch.stack((tokens, torch.isfinite(values).to(torch.int64)), -1).tolist()


def verify_rows(rows: Sequence[Sequence[int]], drafts: Sequence[int]) -> list[int]:
    if len(rows) != len(drafts) + 1:
        raise ValueError("one target row is required per draft plus the bonus")
    accepted = []
    for row, (token, valid) in enumerate(rows):
        if not valid:
            raise RuntimeError("reachable target distribution has no valid token")
        accepted.append(token)
        if row == len(drafts) or token != drafts[row]:
            break
    return accepted


class RequestSampler:
    def __init__(
        self,
        params: SamplingParams,
        prompt: Sequence[int],
        device: str | torch.device,
        vocab_size: int | None = None,
    ) -> None:
        self.params = params
        self.prompt = list(prompt)
        self.generated = []
        self._device = torch.device(device)
        self._history_counts = None
        if (
            params.repetition_penalty != 1
            or params.frequency_penalty != 0
            or params.presence_penalty != 0
        ):
            if any(type(t) is not int or t < 0 for t in self.prompt):
                raise ValueError("sampling history token is outside the vocabulary")
            width = (
                vocab_size
                if vocab_size is not None
                else max(self.prompt, default=-1) + 1
            )
            if width < 0 or (self.prompt and max(self.prompt) >= width):
                raise ValueError("sampling history token is outside the vocabulary")
            prompt_counts = (
                torch.bincount(
                    torch.tensor(self.prompt, device=device, dtype=torch.int64),
                    minlength=width,
                )
                if params.repetition_penalty != 1
                else torch.zeros(width, device=device, dtype=torch.int64)
            )
            self._history_counts = (prompt_counts, torch.zeros_like(prompt_counts))
        self.generator = torch.Generator(device=device)
        if params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(params.seed)

    @property
    def plain_greedy(self) -> bool:
        p = self.params
        return (
            p.temperature == 0
            and p.repetition_penalty == 1
            and p.frequency_penalty == 0
            and p.presence_penalty == 0
        )

    def sample(
        self,
        logits: torch.Tensor,
        drafts: Sequence[int] = (),
        bitmask: np.ndarray | torch.Tensor | None = None,
    ) -> list[int]:
        return verify_rows(self.draw_rows(logits, drafts, bitmask).tolist(), drafts)

    def draw_rows(
        self,
        logits: torch.Tensor,
        drafts: Sequence[int] = (),
        bitmask: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw on the device so a batch may copy all request rows at once."""
        needs_history = (
            self.params.repetition_penalty != 1
            or self.params.frequency_penalty != 0
            or self.params.presence_penalty != 0
        )
        if needs_history:
            self._resize_counts(logits.shape[-1])
        probs = probabilities(
            logits,
            self.params,
            self.prompt if needs_history else (),
            self.generated if needs_history else (),
            drafts,
            bitmask,
            history_counts=self._history_counts,
        )
        if self.params.temperature == 0:
            selected = probs.argmax(-1)
        else:
            # Exponential races also handle unreachable all-masked later rows;
            # their NaNs are rejected only if such a row becomes reachable below.
            noise = torch.empty_like(probs).exponential_(generator=self.generator)
            selected = (probs / noise).argmax(-1)
        valid = torch.isfinite(probs).all(-1) & (probs.sum(-1) > 0)
        return torch.stack((selected, valid.to(torch.int64)), -1)

    def commit(self, tokens: Sequence[int]) -> None:
        """Record only tokens retained after EOS/stop/length handling."""
        if self._history_counts is not None and tokens:
            if any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("sampling history token is outside the vocabulary")
            if max(tokens) >= self._history_counts[0].numel():
                self._resize_counts(max(tokens) + 1)
            counts = self._history_counts[1]
            indices = torch.tensor(tokens, device=self._device, dtype=torch.int64)
            counts.index_add_(0, indices, torch.ones_like(indices))
        self.generated.extend(tokens)

    def _resize_counts(self, width: int) -> None:
        if self._history_counts is None:
            return
        current = self._history_counts[0].numel()
        if current > width:
            raise ValueError("sampling history token is outside the vocabulary")
        if current < width:
            self._history_counts = tuple(
                torch.nn.functional.pad(count, (0, width - current))
                for count in self._history_counts
            )
