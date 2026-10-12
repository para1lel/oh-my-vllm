# Current state and open work

Updated: 2026-10-12.

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

The service and requests have a numeric `thinking_token_budget`, with default `-1` for no separate limit.
Target masks force `</think>` after the content budget, then generation continues. Effort and total limits stay effective.
The service default covers OMP's Chat, Responses, and automatic compaction requests, which omit the field.
[ADR-012](decisions/ADR-012-reasoning-token-budget.md) records the engine comparison and speculative contract.

The worker step deadline has a positive configurable value, with a default of 120 seconds.
It includes first-use compilation and graph capture.

Compiler metadata has count and byte limits. Graph caches have count limits for resident entries and check capture headroom.
Compiler retirement starts garbage collection at safe host boundaries and keeps owners needed for resident graph replay.
Prefill admission accounts for planned storage. Graph pressure eviction releases unused pool owners.
DSpark graph recapture uses a cooldown for repeated shape changes.

All seven capture sites synchronize and clear cached cuBLAS workspaces before capture.
This puts new workspaces in their owning graph pools and keeps workspace budgets and numerical tolerances.
Graph-enabled startup must use an enabled native caching allocator without `max_split_size_mb`.
Unsupported explicit configurations are rejected without rewriting them. Explicit eager mode stays available.
[ADR-013](decisions/ADR-013-cuda-graph-workspace-ownership.md) records the lifetime fix and its scope.

[Architecture](architecture.md) specifies ownership and cache contracts.
[Service contracts](serving.md) specify budgets, worker recovery, and request timing.
[Development](development.md) gives portable environment selection. Keep host configuration in ignored `LOCAL.md`.

## Verification

The persistent-session collector keeps four OMP processes and histories, maximum effort, and runtime context usage.
It keeps incremental journals and checks provider-input anchors against service counts.
Formal acceptance also must have independent resource review.
The sampler keeps failed worker observations before GPU process lookup.
The collector stops the sampler before final checks and evidence hashes.

Rust tests, formatting, line width, Clippy, and CUDA formatting checks passed.
The CPU suite passed 1389 tests and 116 subtests, with 449 GPU cases deselected.
The workspace regression passed sixteen GPU tests, with one intentional allocator-OOM test deselected.
The production capture boundary passed forty-two checkpoint context cases and 210 layer comparisons, with maximum absolute error zero.

The reasoning-budget GPU suite passed sixty cases on source `ee05583` before the workspace fix.
It covered ordinary, MTP4, DSpark, the two APIs, streams, JSON, tools, stochastic sampling, and service defaults.
DSpark used confidence threshold zero for those cases. This result does not supply phase or persistent-session acceptance.

The new one-hour MTP4 and DSpark collections and revised fifteen phase workloads are pending.
The next DSpark collection failed with CUDA illegal memory access. The concurrent MTP4 and phase attempts were cancelled.

A synchronized diagnostic found the failure in a resident 29-row context graph.
A same-stream workspace lifetime reproducer failed after unrelated graph reset.
The private-pool fix passed forty-two checkpoint context cases, with 210 layer comparisons and maximum absolute error zero.
That diagnostic used synthetic target features and did not exercise the full target, scheduling, or HTTP workload.

All collectors and their owned GPU processes exited. Original failed and cancelled records stay unchanged.
Test the budget suite, fifteen phases, and two persistent-session collections again on the fixed committed source.

The user requested a stop before the budget change. The active phase attempt was cancelled and its owned processes exited.

The next collections use committed source, a service budget of `8192`, and direct-action audit prompts without an exact tool-count quota.

Previous cold-start collections failed the worker step deadline. Their original records keep their failure status.
The collector uses a 600-second step deadline and an 1800-second request deadline without separate warmup.

Twelve phase rows passed their gates on previous measured sources. They keep their original source and binary identities.
They do not supply acceptance for the reasoning-budget change. Test the fifteen workloads again on the updated source.
Typed inventory now identifies Tensor `numel` as host metadata only with a Tensor receiver and an integer result.
The old prefix failure has no such type evidence and stays rejected.

Short diagnostics and cancelled attempts keep their original status in external evidence.
[Acceptance](acceptance.md) identifies previous measurement scope. Those results do not certify the revised workload.

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
