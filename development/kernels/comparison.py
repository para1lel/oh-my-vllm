"""Paired latency decisions, separate from the EngineCore 10% stability gate."""

import math
import random
import statistics


def compare(rounds, *, bootstrap_samples=10000):
    """Each round contains alternating-order (TileLang, CUDA) latency pairs.

    Bootstrap round means hierarchically, retaining between-round variation.
    The one-sided 95% lower bound must be positive; no minimum gain is imposed.
    Unresolved/noisy results are not accepted. Values are positive milliseconds.
    """
    if len(rounds) < 3 or any(len(pairs) < 20 for pairs in rounds):
        raise ValueError("at least three rounds of twenty paired samples required")
    if bootstrap_samples < 1000:
        raise ValueError("at least1000 bootstrap resamples required")
    differences = []
    round_results = []
    for pairs in rounds:
        if any(
            len(pair) != 2 or any(not math.isfinite(x) or x <= 0 for x in pair)
            for pair in pairs
        ):
            raise ValueError("latency pairs must contain positive finite numbers")
        old, new = zip(*pairs, strict=True)
        differences.append([a - b for a, b in pairs])
        round_results.append(
            dict(tilelang_ms=statistics.median(old), cuda_ms=statistics.median(new))
        )
    rng = random.Random(784)
    estimates = []
    for _ in range(bootstrap_samples):
        means = []
        for _ in differences:
            chosen = rng.choice(differences)
            means.append(statistics.mean(rng.choices(chosen, k=len(chosen))))
        estimates.append(statistics.mean(means))
    estimates.sort()
    lower = estimates[int(0.05 * bootstrap_samples)]
    return dict(
        rounds=round_results,
        paired_mean_gain_ms=statistics.mean(map(statistics.mean, differences)),
        gain_lower_95_ms=lower,
        method="hierarchical paired bootstrap, one-sided95%, seed784",
        passed=lower > 0
        and all(r["cuda_ms"] < r["tilelang_ms"] for r in round_results),
    )
