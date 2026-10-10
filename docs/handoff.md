# Current state and open work

Updated: 2026-10-11.

## Implementation

Rust controls HTTP service, request scheduling, and logical KV cache. Python controls one B200 GPU.
Use 784-token blocks, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
The total input and output limit is 262144 tokens.

Ordinary decoding, MTP4, and DSpark are startup choices.
Chat, Responses, streams, tools, thinking settings, constraints, cancellation, and prefix reuse stay available.
DSpark keeps fixed weights and target features `(5,19,33,47,61)`.
Its Markov proposals and cumulative confidence select zero through seven candidates, with initial threshold `0.2`.
Stochastic acceptance uses `min(1,p/q)`, normalized `max(p-q,0)` after rejection, and a target bonus after full acceptance.

CUDA Graph, multiple streams, and PDL control model execution.
All owned CUDA backend code files meet the 800-line limit. The pre-commit hook uses clang-format 23.1.3.
Header content enters build keys, load provenance, formal contracts, and profiler checks.

Prefix identities enter lookup after validated worker state commit.
Admission leases have count and byte limits for queued requests, prepared requests, encoded history, output state, and completed stream drains.
The HTTP supervisor invalidates old logical KV and loads replacement workers after failure.
Affected requests stop and report an error. Recovery keeps HTTP alive.

Compiler metadata has count and byte limits. Graph caches have count limits for resident entries and check capture headroom.
Compiler retirement starts garbage collection at safe host boundaries and keeps owners needed for resident graph replay.
Prefill admission accounts for planned storage. Graph pressure eviction releases unused pool owners.
DSpark graph recapture uses a cooldown for repeated shape changes.

[Architecture](architecture.md) specifies ownership and cache contracts.
[Service contracts](serving.md) specify budgets, worker recovery, and request timing.
[Development](development.md) gives portable environment selection. Keep host configuration in ignored `LOCAL.md`.

## Verification

The phase, capacity, overload, recovery, and long-duration collections use source `706f1a6e7ce2b0188f07794cab574cbbf60f2927`.
The final publication also corrects Markdown image checks.
It keeps the measured inference code, release binary, and selected CUDA configurations.
[Acceptance](acceptance.md) gives original hashes, source scope, failed attempts, and independent reviews.

All fifteen phase workloads passed with two full warmups and five measurements per workload.
A different review process calculated the full prefill and decode models again for all 75 measured schedules.
The [README table](../README.md#measured-performance) gives latency, lower-bound ratios, decode TPS, and theoretical percentages.
Prefix-hit prefill records its ratio. Its spread check and decode gates passed.

All six ordinary/MTP4 capacity cases produced full output at total length 262144, without OOM or recompute preemption.
DSpark keeps phase, numerical, operator-speed, and service checks.

MTP4 completed 5135 normal requests in 3675.338 seconds.
DSpark completed 3104 normal requests in 3665.792 seconds.
Each collection used four oh-my-pi agents and completed the resource and late-period memory checks.
The README has separate plots. The heavy test has no performance gate and daily commands skip it.

Two overload cases and eight allocator-OOM recovery cases passed.
The fault stages cover accepted-state commit, graph capture, sampling, and proposal generation in each speculative mode.
HTTP stayed alive. Replacement workers completed Chat and Responses with cold then reused prefixes.
The independent reviews checked empty request state, resource release, raw hashes, and owned process exit.

The full 317-case CUDA/TileLang collection uses source `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`.
Owned operator implementations and operator case, timing, and comparison files are unchanged from that collection.
The ownership utility and IR have subsequent changes. Their source scope is separate.
The current-source GPU checks cover compiler retirement, graph replay, allocator reclamation, and side-stream owner release.

The CPU suite at the measured source passed 1293 tests and 99 subtests, with 448 GPU tests deselected.
The final document checker passed 14 regressions.
Rust workspace tests passed 143 cases. The required pre-commit checks passed.
The tutorial passed ten excerpt tests and nine browser tests.

All task-owned GPU workers and temporary service listeners are closed.
Original failed or cancelled attempts keep their status. One-hour observations apply to the measured workload and shapes.

## Open work

- Extend `PY-04` to cross-shape graph eviction and recapture near the maximum supported context.
- Measure `PY-06` host overlap before a change to dependent host waits.
- Examine compiler storage with indefinitely changing shapes, using its fixed capacity contracts.
- Keep the heavy soak test explicit and do affected phase workloads again after runtime changes.
- Complete user review of the tutorial.

## Tutorial preview

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` keeps its address, process identity, and maintenance commands.
The tutorial has thirteen Chinese chapters with current project source.
