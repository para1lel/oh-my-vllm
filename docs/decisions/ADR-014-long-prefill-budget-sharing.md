# ADR-014: Long prefill budget division

Date: 2026-10-12. Status: implementation in acceptance.

## Problem

Source `22dd754` failed MTP4 input `131072`, output `4096`, batch `4` decode latency.
Its median wall time was `39.9496` seconds, with a lower bound of `12.2634` seconds.
The ratio was `3.2576`, more than the unchanged limit of `3`.
The prefill ratio and the two spread checks passed.

The first request started decode about `19.7` seconds before the last request.
Its decode interval included twelve mixed steps with other requests' long prefills.
CUDA Event measurements also show this interval. They can include host submission gaps and do not supply pure kernel time.
The original full set stays failed. Canceled collections cannot supply acceptance.

## Decision

Reserve active decode inputs before continued prefill allocation.
Divide the available token budget between two or more eligible long prefills.
Keep FIFO admission and younger-request preemption from the queue tail.
Publish each first token at computation completion without a separate output delay.

Classify a long prefill by its missing span at admission, after read-only prefix lookup.
The span must be more than the configured step budget.
`Request.prefill_start` keeps that initial reused position until the current prefill finishes.
Preemption resets it. Readmission uses a new validated prefix hit.
The classification stays effective when the uncomputed span is less than one step budget.

Do not count paused requests or requests with no available sequence slot.
Each participant must have at least one full block.
Round each quota down to a multiple of `784`.
With budget `32768`, two long requests without prefix reuse get `15680` tokens each.
Four get `7840` each, with `31360` total tokens.
Prompt tails, checkpoint boundaries, and sub-block progress keep their existing rules.

Short prompts use the ordinary budget path.
Cached prompts also use it when their initial missing span is at most the configured step budget.
Prefix estimation does not keep hit vectors across allocation.
Admission repeats the lookup before the existing two-pool protection and allocation sequence.

## Verification

CPU tests check sustained four-request progress, `784` boundaries, prompt tails, decode reservation, prefix hits, and paused sequence limits.
A regression places short prefill tails before an active decode in the running queue.
Each prefill must respect that decode's reserved input budget.

Keep all original `317` operator cases and the pinned TileLang comparison.
Add statically derived long-MTP and multiple-prefill cases for new legal shapes and implementation paths.
Use the same full-output, state, three-round, twenty-pair, and confidence-bound gates.

Test all fifteen phase workloads, six full-context cases, reasoning budgets, and the two persistent OMP modes again.
New ragged shapes, graph capture storage, numerical results, and resource growth must have new evidence.
Previous source results keep their original identities and limitations.
The numerical tolerances, DSpark weights, theoretical formulas, and acceptance limits stay unchanged.

[Architecture](../architecture.md) specifies allocation. [Requirements](../requirements.md) keep the active gates.
