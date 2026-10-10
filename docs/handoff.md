# Current state and open work

Updated: 2026-10-10.

## Implementation

Rust controls service, scheduling, and logical KV. Python controls one B200 GPU.
Use block size 784, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
The total input and output limit is 262144 tokens.

Ordinary, MTP4, and DSpark are startup choices. The worker keeps the selected mode.
Chat, Responses, streams, tools, thinking settings, constraints, cancellation, and prefix reuse stay available.

DSpark uses weights that do not change and target features `(5,19,33,47,61)`.
Markov proposals and cumulative confidence select zero through seven candidates. The initial threshold is `0.2`.
Stochastic acceptance uses `min(1,p/q)`, normalized `max(p-q,0)` after rejection, and a target bonus after full acceptance.

CUDA Graph, multiple streams, and PDL control model execution.
The owned native build uses ten operator headers. These headers use the same launch and storage checks.
All backend code files meet the 800-line limit. The pre-commit hook uses clang-format 23.1.3.

Header content enters the cache key, loaded provenance, formal contract, and profiler checks.
Build and load checks reject changed inputs. The GEMM build identifies its root headers independently.

[Architecture](architecture.md) defines ownership and cache contracts.
[Kernel development](kernels.md) gives the module layout and selected configurations.
[Development](development.md) gives environment and model configuration.

## Verification

The CPU collection passed 766 tests and 94 subtests, with 445 GPU tests deselected.
The GPU collection passed 436 tests, with the nine maximum-context cases deselected for a subsequent collection.
The 50 warnings came from third-party deprecated APIs and a compile-time loop optimization suggestion.
Rust workspace tests, Clippy, formatting, line-width checks, Ruff, and document checks passed.
The tutorial build, ten excerpt tests, and nine browser tests passed.

A different agent checked the CUDA split, function tokens that stay the same, header identities, load races, and formatter.
The corrected owned-operator audit found no new production defect.
The packed-scale and pipeline candidates have diagnostic scope. Production configurations stay the same.

Complete GPU, phase, capacity, and service collection with the current source before performance acceptance.
[Acceptance](acceptance.md) identifies each record's measured source and verification limits.
Previous source results do not supply acceptance for subsequent changes.

## Open work

- Run all fifteen phase rows with two full warmups and five measurements.
- Record prefix-hit prefill ratios without a latency gate. Keep its spread and decode checks active.
- Run the complete operator matrix and all nine total-length 262144 cases.
- Run oh-my-pi tool loops through the two service APIs, with worker shutdown audits.
- Publish the measured table in the README and complete evidence review by a different agent.
- Test graph eviction and recapture with 262k memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change indefinitely and measure `PY-06` host overlap.
- Complete user review of the tutorial.

## Tutorial preview

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` keeps its address, process identity, and maintenance commands.
English Markdown and full Chinese translations give current code contracts. The tutorial has thirteen full Chinese chapters.
