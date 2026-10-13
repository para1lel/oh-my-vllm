# Current state and open work

Updated: 2026-10-13.

## Implementation

Rust controls service, scheduling, and logical KV. Python controls one B200 GPU. Their boundary uses ZMQ DEALER and msgpack.
Keep 784-token blocks, 16 FA layers, 48 GDN layers, `mamba_cache_mode="align"`, and 262144 total input and output tokens.
Ordinary decoding, MTP4, DSpark, the two APIs, tools, thinking, cancellation, constraints, and prefix reuse stay available.
DSpark weights stay unchanged.

Source `efd24d0` corrected serving control-token IDs and added the bounded generated-token cycle guard.
Source `47232e2` added persistent native goals in disposable rootless Docker sandboxes.
Source `ae070e4` accepts descriptive tool keywords through request preparation without changes to inference computation or constrained-decoding implementations.
Original tool descriptions stay in model prompts. OMP validates descriptive argument limits and returns errors to the model.
Explicit strict schemas keep their existing contracts.

The hour checks that the inference framework stays available. It also checks serving contracts, resource limits, and cleanup.
OMP goal status, task progress, database tests, tool errors, and final answers are workload observations.
Original client and collector failures keep their recorded status.
The collector now checks framework records independently and keeps its native workload audit separately.
At the deadline, it pauses continuation, sends native RPC `abort`, and drains model requests.

Public request and worker-step deadlines are 3600 and 600 seconds.
Native client first-event and idle deadlines are 900 seconds. The goal collector turn deadline is 7200 seconds.
These policy changes do not change GPU computation or decoding constraints.
The user waived more tests for timeout parameters alone.
The oneoff client now writes its required settings overlay and saves original SSE before handler shutdown completes.

## Verification

Formal GPU collections on clean `ae070e4` that stayed the same used new project caches and the pinned dependency combination.
All fifteen phase rows completed their full two warmups and five measurements with passed gates.
Six maximum-context ordinary/MTP4 cases passed at batches 1, 2, and 4 without OOM or recompute preemption.
All 487 GPU tests passed without skips. All 60 reasoning-budget cases passed.

Constraints, lifecycle, and long-context checks passed for the two speculative modes and APIs.
The oneoff OMP client stopped before model execution because its settings overlay was missing. Its original failures stay recorded.

All 517 operator cases passed numerical checks. Speed checks passed 514 cases and failed three cases.
A second B200 measurement with a new project cache reproduced all three speed failures.
The full operator matrix therefore does not pass acceptance.
Keep computation unchanged as the user instructed. Do not lower the pinned TileLang performance gate.

The previous CPU collection passed 1536 tests and 156 subtests, with 487 GPU cases deselected.
Its eight required checks and tutorial checks passed. The tutorial had ten Node tests and nine browser tests.
Complete the applicable CPU checks on committed source for the new collector and overlay fixes.

## Framework-hour evidence

Independent review accepted one 3600-second window in each mode with the user's framework-only scope.
These windows used source `ae070e4` and the recorded timeout settings.

MTP had 412 requests, 407 server completions, and five matched terminations, with the stop tail.
DSpark had 87 requests, 79 server completions, and eight matched terminations, with the stop tail.
Source, live process birth identities, and worker epoch stayed fixed. No unexpected OOM or worker restart occurred.
All eight lane workspaces, containers, copied tools, caches, and owned images were removed.

Worker RSS grew from 1.136 to 6.822 GiB in MTP and from 1.280 to 8.519 GiB in DSpark.
Maximum worker-snapshot lag was 136.230 and 138.827 seconds. Sampled resource limits and final request release passed.
These results do not prove a memory plateau, full allocation attribution, each intermediate instant, or stable operation beyond the hour.

Original journal/SDK audit failures stay separate. MTP has six server-completed requests without the same-ID native ending message.
Client stream-stall and incomplete-read causes stay unknown. The results do not claim that each request succeeded.

## Open work

Finish independent review of the other phase, budget, and service records.
Run the new collector CPU checks and full CPU suite, then finish the required checks and tutorial build.
Run the previously unexercised `after_commit` allocator-OOM stage for MTP4 and DSpark with the wider public worker-step deadline.
The other six allocator-OOM stages already passed on `ae070e4`.

Run the oneoff OMP compatibility checks through the two APIs after the overlay fix.

Three small normalization shapes fail the unchanged operator speed gate. See [audit](audit.md).
Model-task progress stays a workload observation. Further memory attribution and natural-compaction coverage stay separate limits.
Use [acceptance](acceptance.md) for source-scoped evidence. Daily repair narratives stay in Git history.

## Tutorial preview

The interactive code journey stays available through the user-requested CPU preview service.
Its host address, process identity, and maintenance exception are in ignored `LOCAL.md`.
Keep this preview active. Stop all task-owned GPU workers and temporary inference service ports after each run.
