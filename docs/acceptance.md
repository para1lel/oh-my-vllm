# Acceptance evidence index

Each result applies only to its recorded source, environment, configuration, and workload.
Use [requirements](requirements.md) for gates and [testing](testing.md) for collection commands.
Original records supply formal verification. Portable summaries support reading and offline analysis.

## Current refactor

The accepted plan replaces external Python operator providers with repository CUDA and native libraries.
It also replaces the previous comparison gate with independent numerical contracts and framework phase latency at most 2 times the lower bound.
Prefix-hit prefill keeps its ratio exception. All phase spread gates stay active.

New-source GPU, phase, context, service, and hour collections are pending.
The [handoff](handoff.md) records completed milestones and pending checks.
Do not mark final acceptance from historical measurements or incomplete matrices.

## Historical scope

Pre-refactor repair delivery is recorded at `b98a83a`.
Its source-bound measurements, failures, limitations, and reviewed collector scope stay in Git history and original external records.
They do not prove acceptance of the refactored source or the new 2-times gate.
Obsolete comparison summaries will leave the current working tree during provider migration.

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
