# Current state and open work

Updated: 2026-10-10.

## Implementation

Rust controls service, scheduling, and logical KV. Python controls one B200 GPU.
Use block size 784, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
The total input and output limit is 262144 tokens.

Ordinary, MTP4, and DSpark are startup choices. The worker keeps the selected mode.
Chat, Responses, streams, tools, thinking settings, constraints, cancellation, and prefix reuse stay available.

DSpark uses fixed weights and target features `(5,19,33,47,61)`.
Markov proposals and cumulative confidence select zero through seven candidates. The initial threshold is `0.2`.
Stochastic acceptance uses `min(1,p/q)`, normalized `max(p-q,0)` after rejection, and a target bonus after full acceptance.
DSpark context boundary tests and their acceptance requirement are removed.
Ordinary and MTP4 keep the six total-length 262144 boundary cases.

CUDA Graph, multiple streams, and PDL control model execution.
The native build uses ten operator headers. The GEMM build identifies its root header independently.

All backend code files meet the 800-line limit. The pre-commit hook uses clang-format 23.1.3.
Header content enters cache keys, loaded provenance, formal contracts, and profiler checks.
Build and load checks reject changed inputs.

[Architecture](architecture.md) defines ownership and cache contracts.
[Kernel development](kernels.md) gives module layout and selected configurations.
[Development](development.md) gives environment and model configuration.

## Verification

The fifteen phase rows passed with two full warmups and five measurements per workload.
The [README table](../README.md#measured-performance) gives prefill latency, lower-bound ratios, decode TPS, and theoretical percentages.
Prefix-hit prefill has no ratio gate. Its spread and decode checks passed.

All 317 formal operator cases passed numerical and speed checks in one complete collection.
A different agent calculated all round medians and confidence bounds again.
The six ordinary/MTP4 capacity cases have full output, no OOM, and zero recompute preemptions.
Their summary derives from complete raw logs. The larger failed collection keeps its original status.
MTP4 and DSpark passed oh-my-pi tool loops through Chat and Responses, with normal shutdown audits.

These collections use inference source `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`.
The final change updates the boundary collector, tests, documents, and tutorial without an inference change.
[Acceptance](acceptance.md) gives original hashes, failed attempts, review scope, and limitations.

The current CPU suite passed 760 tests and 94 subtests, with 442 GPU tests deselected.
The GPU suite passed 436 tests with maximum-context cases deselected.
Rust workspace tests, Clippy, formatting, line-width checks, Ruff, and document checks passed.
The tutorial build, ten excerpt tests, and nine browser tests passed.
All owned GPU workers exited. Temporary service ports were released.

A different agent checked the CUDA split, header identities, load races, and formatter.
The corrected Opus 5.5 audit supplied no new production selection.
Packed-scale and pipeline candidates stay external diagnostics.
[Profiling](profiling.md) records selected optimizations and measured results.

## Open work

- Test graph eviction and recapture with memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change indefinitely and measure `PY-06` host overlap.
- Complete user review of the tutorial.

## Tutorial preview

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` keeps its address, process identity, and maintenance commands.
English Markdown and full Chinese translations give current code contracts. The tutorial has thirteen full Chinese chapters.
