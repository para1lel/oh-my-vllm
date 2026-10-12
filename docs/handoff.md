# Current state and open work

Updated: 2026-10-13.

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

Shutdown releases resource owners, then the bridge exits the process without a final allocator flush.
Normal graph-pressure eviction keeps its allocator cleanup. The Rust shutdown deadline stays `30` seconds.

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

Fresh phase and operator attempts on `4100d1a` stopped for external GPU processes before complete rows.
The capacity attempt produced full ordinary batch `1` output, without OOM or preemption, then failed worker shutdown.
The GPU suite stopped after a GPU lookup timeout. Six boundary tests failed without saved pytest exception reports.
These attempts do not supply formal acceptance. All owned processes exited.

A small independent check showed that the inner GPU check rejects an earlier CUDA context in its parent pytest process.
The full test launcher now runs the boundary module and the other tests in two new processes.
Independent CPU collection checks kept the original node multiset, explicit selectors, duplicate counts, empty groups, and failure evidence.

Shutdown stack inspections stopped at `torch.cuda.empty_cache()` after resource-owner release.
A diagnostic that skipped only that final call completed ordinary batch `1` at input `258048` plus output `4096`.
It kept the `30`-second shutdown deadline and left no owned process alive.
The diagnostic used a saved cache and permitted external GPU work. It does not supply formal capacity or performance acceptance.

The shutdown and test-process changes passed the eight required checks and pre-commit checks.
The full CPU suite passed `1406` tests and `116` subtests, with `455` GPU cases deselected.
Collection and lifecycle checks passed `23` tests. The tutorial passed ten excerpt tests and nine browser tests.

The `3cb4cbc` phase and operator attempts stopped after a GPU query timeout, without full acceptance.
Two saved operator cases passed arithmetic checks done independently. The first capacity attempt stopped for an external GPU process.
The next capacity attempt and GPU suite were canceled before the query-scope change. All owned processes exited.
The GPU suite passed three boundary tests, then had one failure without a saved exception report.

The ownership query now uses only the selected GPU UUID. Its timeout stays `10` seconds.
Query failures still stop the collection. Process acceptance still checks exact ancestry and birth identity.
Diagnostic queries did not identify the cause of the earlier timeouts.

The GPU query change passed the eight required checks.
The full CPU suite passed `1407` tests and `119` subtests, with `455` GPU cases deselected.
Seven ownership tests passed, with three subtests. The tutorial passed ten excerpt tests and nine browser tests.
New GPU acceptance is pending on committed source.

Source `66edc40` passed twenty reasoning-budget cases each in ordinary and MTP4 modes, with raw-record checks done independently.
Its long-MTP batch `4` completed two warmups and five measurements, then failed the source contract check.
The reviewed `model_runner.py` hash was equal to the source hash before the shutdown change.
All other source hashes, providers, headers, compiler flags, and versions were equal to the runtime values.
The other collections were canceled before the contract update. All owned processes exited.

The contract update changes only that reviewed source hash.
A new CPU test checks whether the current runtime is equal to the full reviewed contract without CUDA initialization.
The test failed before the hash update. New committed-source GPU acceptance is pending.

The contract update passed the eight required checks.
The full CPU suite passed `1408` tests and `119` subtests, with `455` GPU cases deselected.
Contract and metadata checks passed `32` tests. The tutorial passed ten excerpt tests and nine browser tests.

Source `2145e13` passed all six full-context ordinary/MTP4 cases and long-MTP batch `4`, with checks done independently.
The long phase had two full warmups and five measurements, with zero preemptions and full output.
Its prefill ratio was `2.503124` and decode ratio was `1.854889`. The spread checks passed.

Its operator collection saved `176` full cases, with one small-row projection speed failure and no numerical failure.
The failed case used two rows, width `5120`, and `34816` output columns.
Its confidence bound was negative. The original records keep that failure.

Other collections were canceled before the small-row change. Persistent sessions did not reach one hour.
All owned GPU processes exited and temporary service endpoints closed before source changes.

The small-row candidate fuses residual RMS and FP8 quantization with `640` threads and eight values per thread.
It keeps BF16 rounding points, compact column-major activation scales, and the previous TRT-LLM GEMM provider.
Small inputs without the necessary vector alignment keep the previous chain.
Larger inputs keep their previous dispatch.

An external comparison passed `21` full-output and timing cases across three thread/vector configurations.
This diagnostic used a saved cache. It does not supply formal operator acceptance or identify the earlier failure cause.
Run all applicable GPU collections again on the new committed source.

The small-row candidate passed the eight required checks and pre-commit checks.
The full CPU suite passed `1408` tests and `119` subtests, with `468` GPU cases deselected.
Contract and metadata checks passed `32` tests.

Focused GPU regressions passed `69` tests, with the previous exact checks and `13` new replay/storage-offset cases.
Their runtime source files stayed unchanged during execution. All owned GPU processes exited.
The tutorial passed ten excerpt tests and nine browser tests after the chapter updates.

The next persistent collections keep four sessions per mode for at least `3600` active seconds each, without intermediate drain.
Use a service reasoning budget of `8192` and direct-action audit prompts without an exact tool-count quota.
Prompts tell agents to give file and line references without a copy of source text.
The collector turn deadline is `7200` seconds. Worker step and service request deadlines stay `600` and `1800` seconds.
First-use compilation and capture stay inside the workload, without separate warmup.

Source `d7f0b5f` passed sixty reasoning-budget cases, six full-context cases, and long-MTP batch `4`, with independent raw-record review.
The long phase had prefill ratio `2.494071` and decode ratio `1.829035`. Its spread checks passed.
Its operator collection saved `313` complete rows before cancellation. These rows do not supply the full `517`-case matrix.

Its full GPU suite passed `211` tests, failed one allocator-release test, and did not run `256` tests.
A separate diagnostic passed the original five graph-memory tests.
Another diagnostic failed the same assertion with a live tensor in a default-pool segment with allocated and free storage.
It does not prove the cause of the original suite failure.

The allocator test now uses a temporary allocation more than `total_size - allocated_size` for each existing default-pool segment.
Its original graph, output, pool, and release assertions stay unchanged. A new GPU case checks a segment with allocated and free storage.

MTP persistent collection failed OMP's repeated-output check before one hour. Its retry and the other active collections were canceled before this test change.
All owned GPU processes exited and temporary service endpoints closed. New committed-source GPU acceptance is pending.

The allocator test change passed the eight required checks and pre-commit checks.
The full CPU suite passed `1408` tests and `119` subtests, with `469` GPU cases deselected.
Six focused graph-memory GPU tests passed. The tutorial passed ten excerpt tests and nine browser tests.

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
