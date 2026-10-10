# Current state and open work

Updated: 2026-10-11.

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

All CUDA backend code files meet the 800-line limit. The pre-commit hook uses clang-format 23.1.3.
Header content enters cache keys, loaded provenance, formal contracts, and profiler checks.
Build and load checks reject changed inputs.

Prefix cache publication follows validated state commit. Request admission and completed streams share resource leases with count and byte limits.
The HTTP supervisor invalidates old logical KV and loads replacement workers without request replay.
Graph pressure eviction releases unused pool owners. Compiler and diagnostic registries have fixed limits.

Project-owned FX source and FakeTensor metadata have count and byte limits.
Compiler retirement starts Python garbage collection at safe host boundaries.
Prefill graph admission uses planned storage instead of one threshold for all routes.
[Service contracts](serving.md) specify resource budgets, recovery limits, and request timing.

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
These results apply to the code before the current serving resource changes. The new collection must verify the changed inference source.
[Acceptance](acceptance.md) gives original hashes, failed attempts, review scope, and limitations.

The previous CPU suite passed 760 tests and 94 subtests, with 442 GPU tests deselected.
The GPU suite passed 436 tests with maximum-context cases deselected.
Rust workspace tests, Clippy, formatting, line-width checks, Ruff, and document checks passed.
The tutorial build, ten excerpt tests, and nine browser tests passed.
All owned GPU workers from that collection exited. Its temporary service ports were released.

A different agent checked the CUDA split, header identities, load races, and formatter.
The corrected Opus 5.5 audit supplied no new production selection.
Packed-scale and pipeline candidates stay external diagnostics.
[Profiling](profiling.md) records selected optimizations and measured results.

### Serving resource validation

The changed-source CPU collection passed 922 tests and 99 subtests, with 445 GPU tests deselected.
Rust workspace tests passed 143 cases. Formatting, Clippy, line-width, Ruff, CUDA formatting, and document checks passed.
The updated tutorial passed ten excerpt tests and nine browser tests.

The GPU collection passed 437 tests and failed two tests that used old test contracts.
The corrected graph-pool and compiled-metadata tests then passed together.
Maximum-context cases were deselected in this collection. CUDA operator implementations did not change.

A target-model MTP4 diagnostic passed genuine cancellation and slow-client completion through Chat and Responses.
Final request state and resource leases were empty. Logical pools had only null block 0 allocated.
All owned service processes exited. This short diagnostic supplies no long-duration acceptance.

An MTP4 `after_commit` diagnostic reached CUDA allocator OOM after visible stream output.
HTTP stayed alive, the stream reported failure, and one replacement worker completed Chat and Responses requests.
The shared prefix was cold after recovery and hot on its next use. Final request state was empty and all owned processes exited.
This diagnostic covers one fault stage. Complete two-mode fault, phase, and soak acceptance is pending.

The current CPU suite passed 1133 tests and 99 subtests, with 448 GPU tests deselected.
Rust workspace tests passed 143 cases. All pre-commit checks passed.

A different agent examined protocol ends, client messages, server metrics, and the 2-second disconnect limit.
The updated tutorial passed ten excerpt tests and nine browser tests.
The source `a0109c9cdebbd142a5ae79109b54766dc3ee2464` has a complete MTP4 soak with 5733 normal requests in 3670.64 seconds.
All MTP4 availability, resource, and memory checks passed.

The DSpark soak completed 964 normal requests in 3643.13 seconds without request failure, OOM, or worker restart.
Its GPU and HTTP memory checks passed. Its worker RSS check failed with 12.32% late-period median growth.

The same source passed eight allocator-OOM recovery cases, with four stages per speculative mode.
These results apply to the code before the next compiler metadata and graph-admission corrections.
They do not supply acceptance for those corrections.
All task-owned GPU workers and temporary service ports from these collections are closed.
The full original records keep failed and stopped attempts.

The changed IR passed CUDA Graph replay after compiler retirement and a cold recapture.
Two graph-memory tests passed allocator reclamation and side-stream owner release.

An isolated DSpark batch-four diagnostic passed six captures, twelve hits, and no pressure evictions over three executions.
Saved FA/GDN bytes and generated tokens agreed across executions.
It uses the reviewed admission files over `a0109c9cdebbd142a5ae79109b54766dc3ee2464`, with the previous IR.
This diagnostic does not supply acceptance for the complete changed inference source.

## Open work

- Verify `REQ-SERVE-006`, `REQ-SERVE-007`, and `REQ-SERVE-008` with current-source resource, allocator-OOM, and long-duration service collections.
- Verify RSS stability after the compiler metadata and graph-admission corrections.
- Measure DSpark batch-four phase latency with steady prefill graph replay.
- Measure each of MTP4 and DSpark for at least 3600 seconds and 500 normal requests with four oh-my-pi agents.
- Publish each mode's request scatter plot, then measure the fifteen phase workloads again on the changed inference source.
- Keep earlier measurements limited to their recorded source scope until the new collection passes.
- Run graph eviction and recapture tests with memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show compiler storage at or below its specified capacity limits when shapes change indefinitely and measure `PY-06` host overlap.
- Complete user review of the tutorial.

## Tutorial preview

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` keeps its address, process identity, and maintenance commands.
English Markdown and full Chinese translations give current code contracts. The tutorial has thirteen full Chinese chapters.
