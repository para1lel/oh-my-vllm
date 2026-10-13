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

The previous persistent workload required four sessions per mode, with at least `3600` active seconds each and no intermediate drain.
It used a service reasoning budget of `8192` and direct-action audit prompts without an exact tool-count quota.
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

## SDK evidence and host diagnostics

Source `f019075` passed sixty budget cases, six full-context cases, eleven phase rows, and all `469` GPU tests, with independent review.
It did not complete the fifteen phase rows or the full `517`-case operator matrix.
An operator attempt saved `42` rows, with one negative confidence bound. Original failed and canceled records keep their status.

Neither persistent collection completed all four active lanes for one hour.
Six natural MTP compaction calls had SSE and service records but no independent SDK usage record.
The DSpark collector also counted a normal `turn_end` as another SDK completion.
Independent review did not accept host memory stability in either mode.
All owned collectors, GPU processes, and temporary model-service endpoints stopped before changes.

The collector now receives native OMP SDK traces through one owned loopback endpoint per fixed client.
It checks hashed original exports, full wire coverage, configuration API bindings, and normal collector closure.
Only `message_end` counts as an RPC completion. Two SDK completions with the same response ID fail verification for healthy HTTP streams.

Four pinned OMP CPU integration cases passed across the two APIs and all three automatic-compaction purposes.
Full request JSON stayed equal with native SDK telemetry disabled and enabled.
These fixtures do not supply target-inference or hour-long acceptance.

Optional glibc allocator measurements record field values and call time in different fields from RSS.
The option keeps existing diagnostic intervals and does not change allocator options or start extra resource release.
Its CPU tests passed, with the structure size and field offsets checked against the installed C header.

Rust passed `152` tests. The CPU suite passed `1466` tests and `119` subtests.
CPU mode did not select `469` GPU tests.
The eight required checks, dependency check, and release build passed.
The tutorial passed ten source-display tests and nine browser tests.

New committed-source GPU acceptance and independent resource review are necessary.

## Packaged FlashInfer artifacts

Source `9d9d105` passed ordinary batch `1` phase checks. Its other GPU collections stopped before full acceptance.
The DSpark Responses request stopped at the installed client's `300000`-millisecond progress timeout.
Service keepalive comments did not reset that timer. Original logs show cold compilation and artifact downloads during the wait.
All active and queued jobs, owned GPU processes, and temporary model-service endpoints stopped before the dependency change.

The runtime now pins `flashinfer-cubin` to the same `0.6.18.post1` version as `flashinfer-python`.
The dependency URL keeps the official wheel SHA-256.
Independent review checked the full wheel and seven completed artifacts from the failed DSpark download.
One additional header had no completed old download for comparison.

Worker identity records the selected artifact directory and check options.
The reviewed contract adds five provider source hashes and the package version.
Its other providers, versions, compiler flags, and header inputs stay equal to the previous values.

Four offline CPU artifact tests passed without CUDA initialization. The dependency check passed for `126` packages.

The two comparison manifests now bind the added dependency. Reference kernels, case IDs, and numerical tolerances stay unchanged.
The CPU SDK fixture keeps full request bodies for the two concurrent automatic summary calls.

The final CPU suite passed `1470` tests and `119` subtests, with `469` GPU tests not selected.
Rust passed `152` tests. The eight required checks passed.
The tutorial passed ten source-display tests and nine browser tests.

New committed-source GPU acceptance and independent resource review are pending.
Cold-start verification must keep the original client timeout and fresh project caches.

## Short Q/K and executable diagnostics

Source `4804ec6` passed all `469` GPU tests, sixty budget cases, and six full-context cases, with independent review.
Its phase collections did not complete all fifteen rows.
The operator collection saved `354` complete rows. All numerical checks passed, but the `144`-token Q/K speed comparison failed.
Neither persistent collection completed one hour. Independent review did not accept host memory stability.

RSS continued to increase after several cache counts reached their limits.
All owned collections and waiters stopped. Independent review checked process births, GPU release, and temporary endpoints before source changes.

The Q/K candidate uses four values per lane through `320` tokens. Larger inputs keep their eight-value paths.
The packed-layout guards, FP32 squares, epsilon, BF16 rounding, PDL, and numerical tolerances stay effective.
An external comparison supports the short path at sampled lengths through `320` tokens.

One later focused production matrix still failed the `576`-token comparison. That failed record stays unchanged.
Further verification is necessary before full acceptance.

The next focused production comparison passed all twelve Q/K cases and `48` GPU numerical, graph replay, and output-pool tests.
Independent review checked the exact bootstrap results, source and native-library identities, and process release.
This focused result does not supply the full `517`-case matrix or unified acceptance.

Three completed MTP audit items had grep output. Each work item had no source read.
The unchanged source-read gate rejects these items.
The next task prompt tells the agent to read source again before completion. Earlier records keep their original results.

Optional executable diagnostics use weak references to report owner tuple lengths, shared objects, generation identities, and other live compiled executables.
Weak reference callbacks, unsupported objects, and observation overflow have different fields. Diagnostics do not add strong executable owners.
Shallow byte counts exclude contained objects and native storage. They do not identify the cause of the earlier RSS increase.
Complete new committed-source acceptance and independent resource review before publication.

The revised CPU suite passed `1477` tests and `119` subtests. CPU mode did not select `485` GPU tests.
The ten warnings were dependency deprecation warnings.
Rust passed `152` tests. The eight required checks, release build, and dependency check passed.
The tutorial passed ten source-display tests and nine browser tests.

The reviewed runtime contract changes only the Q/K source and header, two diagnostic source files, and one new diagnostic module.
Its providers, versions, GEMM headers, and compiler flags stay equal to the previous contract.

Source `c4b45f5` passed post-commit CPU preflight with `69` tests and `3` subtests.
Its next unified collection stopped before full acceptance.
The two persistent-session tests failed Rust msgpack decoding after the first nonempty owner histogram used integer keys.
The ordinary phase stopped when process ownership observation reported `pidfd_open` error `EINVAL`.
Its record did not identify the affected PID, so the exact cause is unknown.

All seven collection roots exited before changes. Independent checks found each of the `3611` recorded owned process births missing.
Temporary ports and IPC paths closed, and GPU process queries were empty.
Original failed and canceled records keep their source and status.

Owner histogram keys now use strings for the Rust JSON-value contract.
The HTTP/ZMQ regression failed before the fix, then passed with the other seven executable diagnostics tests.
The runtime contract changes only this reviewed diagnostic source hash.

The process observer checks the original birth again after `EINVAL`.
It does not register a missing or changed birth. The same birth still causes failure.
Other `pidfd_open` errors keep their previous handling.
This check does not prove the cause of the earlier failure.
New committed-source GPU acceptance and independent continuous memory review are open.

The first revised full CPU suite passed `1482` tests and `119` subtests, with one failed lifecycle assertion.
That assertion expected an integer owner histogram key. It now expects the protocol's string key.
The failed record stays unchanged.

The next full CPU suite passed `1483` tests and `119` subtests, with `485` GPU tests not selected.
The ten warnings were dependency deprecation warnings.
Independent review checked the original results, equal source hashes before and after testing, and process release.

Rust passed `152` tests. The eight required checks and all-files hooks passed.
The tutorial passed ten source-display tests and nine browser tests.
New committed-source GPU acceptance and independent continuous memory review are open.

## Shared service sampling

Source `249bbb0` passed all `485` GPU tests, six full-context cases, and the ordinary and MTP4 phase rows at batches `1/2/4`.
Independent review checked original results, source identities, and owned process exit.
Its long-MTP batch `4` completed normally. Independent phase-model review for that row is still open.
The operator collection saved `361` of `517` rows before cancellation. These partial rows do not supply full acceptance.

Its MTP persistent test failed after `2379.1744` seconds with repeated ordinary text.
The reasoning channel had no repeated cycle. The SDK label did not identify the output channel.
The request used checkpoint temperature `1`, top_p `.95`, and top_k `20`, with presence penalty `0` and repetition penalty `1`.

The server recorded `5035` generated tokens on cancellation, before the explicit service reasoning budget `8192` and total context limit.
Original failed records keep their status and usage limitations.

The DSpark persistent test and other unfinished collections were canceled before sampling changes.
All eleven collection roots were reaped. Task-owned GPU processes exited before the first source edit.
The source stayed equal during cancellation. Original failed and canceled records stay unchanged.
Neither persistent test supplies one-hour or independent memory acceptance.

The service uses one shared presence penalty default `0.5` for production and test HTTP services in each draft mode.
Checkpoint temperature, top_p, top_k, total limits, and the separate reasoning budget default `-1` keep their previous values.
Use `--override-generation-config` for validated service changes. Explicit request values take precedence, with `0` to disable the presence penalty.

Diagnostics show service defaults. Request metrics show service and explicit request sampling fields, before Python checkpoint merge.
[ADR-015](decisions/ADR-015-shared-serving-sampling-defaults.md) records official guidance and the configuration limits.
The selected penalty is in Qwen's adjustment range, but new stochastic and persistent-session acceptance are necessary.

New persistent tests must use production sampling and reasoning-budget defaults, without different startup overrides.
Budget boundary tests and offline greedy measurements keep their specified explicit parameters.
The new integration regression failed before implementation, then passed.
Three Rust sampling tests and three HTTP/ZMQ/Python tests passed, with `13` subtests.

The first all-target Rust checks failed because the standalone normalizer had no reference to the new sampling module.
The CPU collection stopped before that module reference changed. The original failed and canceled records stay unchanged.

After the fix, Rust passed `158` tests. The eight required checks, release build, and all-files hooks passed.

The full CPU suite passed `1486` tests and `132` subtests, with `485` GPU tests not selected and ten dependency warnings.
Source and test file hashes stayed equal during execution.
The tutorial passed ten source-display tests and nine browser tests after the final request-null wording clarification.

New committed-source GPU acceptance and independent resource review are open.

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
