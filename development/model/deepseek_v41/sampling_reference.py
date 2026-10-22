from dataclasses import dataclass
from numbers import Integral

import numpy as np

STREAMS = {"target": 0, "proposal": 1, "accept": 2, "correction": 3, "bonus": 4}
WORD_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class Decision:
    accepted: int
    tokens: tuple[int, ...]

    @property
    def processed_rows(self):
        # The verification input has one previously emitted pending token.
        return self.accepted + 1


def integer(value, bits, name):
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or not 0 <= int(value) < 1 << bits
    ):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return int(value)


def philox(counter, key):
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox4x32 needs four counter words and two key words")
    counter = tuple(integer(value, 32, "counter") for value in counter)
    key = tuple(integer(value, 32, "key") for value in key)
    for _ in range(10):
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = (
            (second >> 32) ^ counter[1] ^ key[0],
            second & WORD_MASK,
            (first >> 32) ^ counter[3] ^ key[1],
            first & WORD_MASK,
        )
        key = ((key[0] + 0x9E3779B9) & WORD_MASK, (key[1] + 0xBB67AE85) & WORD_MASK)
    return counter


def uniform(*, seed, cycle, position, stream):
    seed = integer(seed, 64, "seed")
    position = integer(position, 64, "position")
    cycle = integer(cycle, 32, "cycle")
    if stream not in STREAMS:
        raise ValueError("unknown sampling stream")
    words = philox(
        (position & WORD_MASK, position >> 32, cycle, STREAMS[stream]),
        (seed & WORD_MASK, seed >> 32),
    )
    return ((words[0] << 21) | (words[1] >> 11)) * (2.0**-53)


def draws(values):
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result >= 1):
        raise ValueError("uniform draws must be finite and in [0, 1)")
    return result


def probabilities(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("probabilities must have rows and a nonempty vocabulary")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    totals = result.sum(axis=1, keepdims=True)
    if np.any(np.abs(totals - 1) > 1e-12):
        raise ValueError("probability rows must sum to one")
    return result / totals


def proposal_tokens(tokens, vocabulary):
    result = tuple(integer(token, 64, "proposal token") for token in tokens)
    if len(result) > 5 or any(token >= vocabulary for token in result):
        raise ValueError("at most five valid proposal tokens are allowed")
    return result


def categorical(weights, draw):
    total = weights.sum(dtype=np.float64)
    if total <= 0:
        raise ValueError("categorical distribution has zero mass")
    positive = np.flatnonzero(weights > 0)
    cumulative = np.cumsum(weights[positive], dtype=np.float64)
    index = np.searchsorted(cumulative, float(draw) * total, side="right")
    return int(positive[min(index, len(positive) - 1)])


def verify(target, proposal, tokens, accept_draws, correction_draws, bonus_draw):
    target, proposal = probabilities(target), probabilities(proposal)
    tokens = proposal_tokens(tokens, target.shape[1])
    count = len(tokens)
    if target.shape != (count + 1, target.shape[1]) or proposal.shape != (
        count,
        target.shape[1],
    ):
        raise ValueError("verification needs K proposal and K+1 target rows")
    accept_draws = draws(accept_draws)
    correction_draws, bonus_draw = draws(correction_draws), draws(bonus_draw)
    if (
        accept_draws.shape != (count,)
        or correction_draws.shape != (count,)
        or bonus_draw.shape != ()
    ):
        raise ValueError(
            "verification needs K acceptance/correction draws and one bonus"
        )
    if any(proposal[index, token] == 0 for index, token in enumerate(tokens)):
        raise ValueError("selected proposal token has zero probability")
    for index, token in enumerate(tokens):
        ratio = min(1.0, target[index, token] / proposal[index, token])
        if accept_draws[index] < ratio:
            continue
        residual = np.maximum(target[index] - proposal[index], 0)
        correction = categorical(residual, correction_draws[index])
        return Decision(index, (*tokens[:index], correction))
    bonus = categorical(target[count], bonus_draw)
    return Decision(count, (*tokens, bonus))


def verify_seeded(target, proposal, tokens, *, seed, cycle, pending_position):
    position = integer(pending_position, 64, "pending position")
    tokens = tuple(tokens)
    if len(tokens) > 5:
        raise ValueError("at most five proposal tokens are allowed")

    def draw(offset, stream):
        return uniform(
            seed=seed, cycle=cycle, position=position + offset, stream=stream
        )

    return verify(
        target,
        proposal,
        tokens,
        [draw(index + 1, "accept") for index in range(len(tokens))],
        [draw(index + 1, "correction") for index in range(len(tokens))],
        draw(len(tokens) + 1, "bonus" if tokens else "target"),
    )


def verify_greedy(target_logits, tokens):
    target = np.asarray(target_logits, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] == 0 or not np.isfinite(target).all():
        raise ValueError("target logits must be finite rows with a vocabulary")
    tokens = proposal_tokens(tokens, target.shape[1])
    if len(target) != len(tokens) + 1:
        raise ValueError("greedy verification needs K+1 target rows")
    selected = np.argmax(target, axis=1)
    for index, token in enumerate(tokens):
        if token != selected[index]:
            return Decision(index, (*tokens[:index], int(selected[index])))
    return Decision(len(tokens), (*tokens, int(selected[-1])))
