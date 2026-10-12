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

Long-prefill budget division reserves active decode inputs before prefill allocation.
Two or more eligible long prefills get block-aligned quotas without a change to FIFO admission or tail preemption.
`prefill_start` keeps the missing-span classification through the last chunks and resets after preemption.
Four long requests without prefix reuse use `7840` tokens each per full step. Short prompts keep their ordinary path.
[ADR-014](decisions/ADR-014-long-prefill-budget-sharing.md) records the failed workload and the new allocation.

[Architecture](architecture.md) specifies ownership and cache contracts.
[Service contracts](serving.md) specify budgets, worker recovery, and request timing.
[Development](development.md) gives portable environment selection. Keep host configuration in ignored `LOCAL.md`.

## Verification

The persistent-session collector keeps four OMP processes and histories, maximum effort, and runtime context usage.
It keeps incremental journals and checks provider-input anchors against service counts.
Formal acceptance also must have independent resource review.
The sampler keeps failed worker observations before GPU process lookup.
The collector stops the sampler before final checks and evidence hashes.

The workspace fix passed the eight required checks on source `22dd754`.
Its CPU suite passed 1389 tests and 116 subtests, with 449 GPU cases deselected.
Sixteen GPU workspace regressions passed, with one intentional allocator-OOM test deselected.
Forty-two checkpoint context cases supplied 210 layer comparisons, with maximum absolute error zero.
Those context diagnostics used synthetic target features, without the full target, scheduler, or HTTP workload.

Source `22dd754` passed sixty reasoning-budget GPU cases and all six full-context ordinary/MTP4 cases without warmup.
Budget cases covered ordinary, MTP4, DSpark, the two APIs, streams, JSON, tools, stochastic sampling, and service defaults.
DSpark used confidence threshold zero for budget cases. Formal phases and service workloads use `0.2`.
Capacity cases completed input `258048` plus output `4096` at batches `1/2/4`, without OOM or recompute preemption.
Independent reviews checked original records, source identities, outputs, and owned process exit.

Thirteen phase rows completed on `22dd754`: twelve passed, and long-MTP batch `4` failed the decode ratio.
The failed row had median `39.9496` seconds, lower bound `12.2634` seconds, and ratio `3.2576`.
Its prefill and spread gates passed. The limit stays `3`.

Ordinary batch `2` was interrupted, and ordinary batch `4` did not start.
The records keep their original status. They do not supply fifteen-row acceptance.

The first persistent MTP4 attempt failed OMP's repeated-output check in ordinary response text.
The first DSpark attempt reached its `1800`-second collector turn deadline, without a GPU or tool failure.
The next attempts used a delimiter-safe prompt and a `7200`-second collector turn deadline.
They were canceled before one hour for the required scheduler change.
All collectors and owned GPU processes exited. Failed and canceled records stay unchanged.

The long-prefill candidate passed 53 scheduler tests, with sustained progress, tails, decode reservation, prefix reuse, and paused slot limits.
Eighteen focused static-case tests passed. They keep all original `317` configurations and identify `200` new cases.
The eight required checks and pre-commit checks passed. The workspace has `152` Rust tests.
The tutorial passed ten excerpt tests and nine browser tests.

Earlier CPU attempts failed initial HTTP fixture readiness before behavior assertions.
Saved logs show no ZMQ peer connection before a `15`-second timeout in one attempt.
The fixtures now use the existing `30`-second initial readiness timeout. Request, recovery, and shutdown deadlines stay unchanged.
The repeated full CPU suite passed `1398` tests and `116` subtests, with `455` GPU cases deselected.

The full operator matrix has `517` cases. New operator, phase, capacity, budget, and persistent-session GPU acceptance is pending.
Repeat the applicable checks on committed source before publication.

The next persistent collections keep four sessions per mode for at least `3600` active seconds each, without intermediate drain.
Use a service reasoning budget of `8192` and direct-action audit prompts without an exact tool-count quota.
Prompts name model delimiters without reproducing their literal spellings.
The collector turn deadline is `7200` seconds. Worker step and service request deadlines stay `600` and `1800` seconds.
First-use compilation and capture stay inside the workload, without separate warmup.

Typed inventory identifies Tensor `numel` as host metadata only with a Tensor receiver and an integer result.
The old prefix failure has no such type evidence and stays rejected.
[Acceptance](acceptance.md) identifies previous measurement scope. Previous results do not certify the revised workload.

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
