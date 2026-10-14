# Acceptance evidence index

Each result applies only to its recorded source, environment, configuration, and workload.
Use [requirements](requirements.md) for gates and [testing](testing.md) for collection commands.
Original records supply formal verification. Portable summaries support reading and offline analysis.

## Current refactor

The accepted plan replaces external Python operator providers with repository CUDA and native libraries.
It also replaces the previous comparison gate with independent numerical contracts and framework phase latency at most 2 times the lower bound.
Prefix-hit prefill keeps its ratio exception. All phase spread gates stay active.

The final-source performance matrix and delivery checks are pending.
The [handoff](handoff.md) records completed milestones and pending checks.
Do not mark final acceptance from historical measurements or incomplete matrices.

## Selected subgroup scan checkpoint

The [source-bound summary](evidence/refactor-s8-scan/verification.json) identifies the changed scan and its original verification records.
The complete CPU suite passes 1592 cases and 153 subtests. The complete non-boundary GPU suite passes 593 cases.
Selected GPU verification passes 99 cases. Four alignment and cancellation cases pass Compute Sanitizer with zero errors.

All 539 operator numerical cases pass. Their uncommitted-source report keeps `diagnostic=true`, `passed=false`, and `selected_passed=true`.
The FP32 reduction order changes. Numerical tolerances and the theoretical model stay equal.
Final-source capacity, service, one-hour, and framework performance verification are pending.

The [clean-source repeat](evidence/refactor-f05a35d/verification.json) at `f05a35d` passes all 539 operator numerical cases.
The ordinary 32768-input/4096-output batch-one attempt collects two warmups and five measurements, with zero OOM and recompute preemption.
Prefill and decode ratios are 4.436 and 2.320. Their spread checks pass, but their two-times latency checks fail.
The failed record stays unchanged. This one row does not complete the fifteen-row framework matrix.

## Source 3324a7f

The [portable verification summary](evidence/refactor-3324a7f/verification.json) records source hashes and original report identities.
It is a derived reading artifact. Unchanged original records and independent reviews supply formal verification.

The numerical matrix passes 539 cases. The complete non-boundary GPU suite passes 589 cases.
Six ordinary/MTP4 capacity cases pass at batches 1, 2, and 4 with 258048 input and 4096 output tokens.
MTP and DSpark each pass six service clients and four allocator OOM recovery stages.
The fixture-only repair at `f5b13d7` passes the complete CPU suite: 1592 cases and 153 subtests.

Each mode has four persistent OMP goal sessions over one hour with zero OOM and restart.
Independent reviews verify sampled resource bounds, final release, normal SSE/SDK requests, and sandbox deletion.
MTP completes 389 normal streams and DSpark completes 76. Each mode has four controller deadline aborts.

The original MTP collector fails its SSE audit on a generation/abort deadline race. Its failed result stays unchanged.
The original DSpark collector reports operational completion with `passed=false`, pending independent resource review.
Separate independent reviews bind each unchanged original summary and raw manifest.

Each external driver reports unproved adopted processes after service exit, before it saves their identities.
Recorded service, worker, and sandbox identities have exited. Temporary inference listeners and GPUs are released.
The historical unknown adopted-process identities cannot be reconstructed.
Worker RSS increases during collection. The evidence does not prove a memory plateau or complete allocation attribution.

The ordinary 32768-input/4096-output batch-one phase attempt fails the two-times latency gate.
Prefill and decode ratios are 5.419 and 2.329. Their spread checks pass.
The theoretical equations are unchanged. The complete fifteen-row matrix and final delivery are pending.

## Historical scope

Pre-refactor repair delivery is recorded at `b98a83a`.
Its source-bound measurements, failures, limitations, and reviewed collector scope stay in Git history and original external records.
They do not prove acceptance of the refactored source or the new 2-times gate.
Obsolete comparison summaries were removed from the current working tree.

## Collection identity

Record source and executable hashes, dependency pins, loaded module and header hashes, compiler flags, and checkpoint identity.
Record process birth identities, worker epoch, GPU UUID, and collection times in original external records.
Portable summaries must omit host paths and secrets, and identify their original record hashes.

Keep complete failed sets and interrupted attempts with reasons. Do not select favorable samples.
Compiler activity, capture, cache eviction, interference, or source changes invalidate measured repetitions.
Profile independently from formal timing.

## Service-hour scope

Check continuous framework availability, sampled resource limits, serving contracts, and owned sandbox cleanup.
OMP goal completion, task progress, tool errors, and final answers are workload observations.
Original client failures keep their status. State configured deadlines and cancellations separately.

A one-hour result does not prove a memory plateau, complete allocation attribution, or operation after the measured interval.
Report snapshot lag and the request-drain tail. Keep memory observations in evidence, not README memory charts.
