# Acceptance evidence index

Use [requirements](requirements.md) for gates and [testing](testing.md) for collection commands.
Each result applies to its stated source, configuration, and workload.
Do applicable collections again after inference changes.
Portable summaries support historical reading and offline analysis. Originals supply formal verification.

## Last full collection

The [source-bound record](../bench/evidence/2026-10-10-current-source-acceptance.json) applies to `c27ff1aa4f2458d295c246282f7207b9682164ae`.
Its original aggregate SHA-256 is `201e283e320f378894878bffac6b5470727ab9b8d851d08611a89a6040e7c3b5`.
The aggregate keeps JSON identities, measurements, failed attempts, and reviews by different agents.
Some raw-log reference fields are removed during portable export. The original aggregate hash binds those references.
The release binary SHA-256 is `70f473a2cdaab1f61ae31c02730620aa70a02aecfb920d19e447d423d0483576`.

### Phase latency

Each row uses two full warmups and five repetitions, with 4096 output tokens per request.
The prefix rows reuse 32144 tokens. All rows have zero recompute preemptions and the specified output counts.
Thirteen rows passed. Prefix-hit batches 1 and 2 failed prefill.
All decode phases and all wall-time spread checks passed.

Full project acceptance stays false.

| Mode | Input | Batch | Prefill ratio | Decode ratio | Result |
|---|---:|---:|---:|---:|---|
| dspark | 32768 | 1 | 2.877151 | 2.596131 | pass |
| dspark | 32768 | 2 | 2.871325 | 2.524429 | pass |
| dspark | 32768 | 4 | 2.931858 | 2.424955 | pass |
| mtp4 | 32768 | 1 | 2.878555 | 2.221825 | pass |
| mtp4 | 32768 | 2 | 2.867603 | 2.202879 | pass |
| mtp4 | 32768 | 4 | 2.945349 | 2.322192 | pass |
| ordinary | 32768 | 1 | 2.872472 | 2.297438 | pass |
| ordinary | 32768 | 2 | 2.876307 | 2.222933 | pass |
| ordinary | 32768 | 4 | 2.928289 | 2.250779 | pass |
| prefix | 32768 | 1 | 3.790295 | 2.298568 | prefill failed |
| prefix | 32768 | 2 | 3.161596 | 2.200780 | prefill failed |
| prefix | 32768 | 4 | 2.995868 | 2.112271 | pass |
| ordinary | 131072 | 1 | 2.511653 | 2.075463 | pass |
| ordinary | 131072 | 2 | 2.511573 | 2.080933 | pass |
| ordinary | 131072 | 4 | 2.523345 | 2.250702 | pass |

### Operator matrix

All 317 unique canonical cases passed numerical and speed checks across three completed shards.
A different agent recomputed 951 round medians and all per-case confidence bounds from 19020 pairs.
Each timed graph contains 100 full operations. Each sample replays that graph once.
The matrix keeps the original 230 cases and the same reference pins.

Each shard has coverage that is not full by itself and keeps its original overall `passed=false`.
The reviewed aggregate verifies full coverage and each shard's `selected_passed=true`.
The interrupted parent and empty-shard attempts stay failed. They are kept with their failed statuses.

### Context capacity

All nine ordinary/MTP4/DSpark cases passed input 258048 and output 4096 at batches 1, 2, and 4.
They have no OOM or recompute preemption. Total length is 262144.
Each case uses zero warmups and one cold repetition. Compilation and capture can occur during capacity collection.
Capacity timing and memory observations do not supply statistical phase-performance results.

| Mode | Batch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|
| ordinary | 1 | 120.625252 | 145.841797 |
| ordinary | 2 | 120.625252 | 145.166016 |
| ordinary | 4 | 121.478971 | 148.191406 |
| mtp4 | 1 | 117.010298 | 143.583984 |
| mtp4 | 2 | 117.010298 | 142.824219 |
| mtp4 | 4 | 119.753282 | 149.011719 |
| dspark | 1 | 142.096329 | 173.162109 |
| dspark | 2 | 142.578590 | 175.218750 |
| dspark | 4 | 144.550664 | 176.996094 |

Checkpoint manifests matched before and after collection. Loaded DSpark digests matched the fixed checkpoint.
Closed capacity logs include a successful post-execution module audit.
The target loader has no in-process weight-digest record.
The reviewer verified recorded digests and configuration files without rereading all weights.

Logged workers exited and temporary IPC files were removed.
The capacity artifacts have no process birth inventory or final saved NVML cleanup snapshot.
The collector checks exclusivity once per second and after exit. Interference fully between polls cannot be excluded.

### Service and Python tests

DSpark and MTP4 each passed six client groups, twelve constraints, the two API lifecycles, and observed oh-my-pi tool loops.
Each suite admitted 51 generation requests, completed 49, and cancelled 2. All requests were released.
Each has 57 HTTP records and 6 storage retrieval/deletion records.
The two agents read two source sections, forwarded the returned tool results, and continued generation with those results.

Four long-prefix requests per suite had input 131099 and output 21.
Their cached-token counts were zero for the first request and 130928 for the other requests.
Each DSpark request verified 24 drafts and accepted 16. Each MTP4 request verified 20 and accepted 15.
These are functional observations, not capacity or phase-performance results.

The saved shutdown records show owned processes exited and temporary ports were released.
Service startup checks passed. A post-execution loaded-module audit success was not observed in the service logs.
Service artifacts have no target weight tree digest. OMP answer-detail errors stay in the semantic reviews.

The full Python collection passed 1181 tests and 65 subtests. Two test assumptions failed.
Subsequent focused checks passed after their correction.
The current CPU collection passed 753 tests and 89 subtests, with 445 GPU tests deselected.
Full GPU recollection stays pending.

## Current accumulator diagnosis

Five FP8 projections at 624 through 2496 rows now use an owned paired TMEM accumulator and a direct BF16 epilogue.
The production path passed 26 boundary, graph, changed-data, PDL, stream, and storage checks.
All 18 affected full-operation cases passed their numerical and speed protocol.
Filtered dirty-source timing has diagnostic scope. Subsequent full-source verification stays pending.
The [accumulator diagnosis](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) keeps the original hashes, selected-case measurements, last profiler records, and review scope.

The observability tool records 21 counters per kernel, worker identity, and loaded CUDA records.
Representative TileFoundry shapes, profiler replays, and interleaved latency measurements have different scopes.
[Profiling](profiling.md) gives the selected implementation and rejected candidates.

## DSpark threshold and paired measurements

DSpark keeps fixed weights and shares the target embedding and vocabulary head.
Its initial cumulative confidence threshold is `0.2`. Set it to `0.0` for fixed-count diagnostics.
Output, context, and grammar limits still apply.

The [threshold-selection record](../bench/evidence/2026-10-08-dspark-confidence-selection.json) uses inputs different from the formal synthetic workload.
It has two text/tool-history requests, with input 32768 and output 512 per request.
Each setting has two full warmups and two measured repetitions.
Thresholds `0.0`, `0.05`, `0.1`, and `0.2` have TPS medians 177.287, 183.097, 184.198, and 184.323.
Their acceptance rates are 28.487%, 44.957%, 53.406%, and 62.500%. These medians apply to this small collection.

### Previous-source paired measurements

The [paired record](../bench/evidence/2026-10-09-dspark-mtp4-paired.json) uses runtime source `e83674c` on one B200.
It has input 32768 and output 4096 at batches 1, 2, and 4, with greedy sampling and cold prefixes.
The modes share one process, target weights, and physical target caches.
Each mode has two full warmups in each of three rounds.
Each round has five pairs. The mode order changes with each pair.

All three batches had the specified output counts and zero recompute preemptions.
All 90 measured intervals had zero captures or compilations.

Each table row uses fifteen measured batches for each mode.
TPS is the median of those fifteen measurements.
TPS includes registration, prefill, decode, transport, sampling, and cleanup. The measured time starts after model loading and warmups.
TTFT uses the maximum request TTFT in each batch, then the median of those values.
Acceptance divides total accepted drafts by total verified drafts.

| Batch | Mode | TPS | TTFT (s) | Acceptance | Verified / step | Accepted / step |
|---|---|---|---|---|---|---|
| 1 | MTP4 | 319.134 | 1.631171 | 90.647075% | 3.980877 | 3.608549 |
| 1 | DSpark | 150.959 | 1.598959 | 72.105644% | 1.313688 | 0.947243 |
| 2 | MTP4 | 493.461 | 3.297708 | 80.736434% | 7.413793 | 5.985632 |
| 2 | DSpark | 221.112 | 3.223185 | 63.357157% | 1.854226 | 1.174785 |
| 4 | MTP4 | 723.972 | 6.565200 | 75.358676% | 14.667266 | 11.053058 |
| 4 | DSpark | 355.888 | 6.428824 | 56.250624% | 3.394175 | 1.909245 |

The step counts include all nonempty prefill/decode scheduling batches. The numerators include all requests in each batch.
Service logs have per-request step counts. Keep these two count scopes different.
Adaptive truncation changes the verified-draft denominator. The summary keeps proposed, verified, accepted, and step counts.

The release binary SHA-256 is `4202db0d33251b705e1bedf65ec23bee979ec75377ee42ae7d9f30d0cc8c1072`.
The loaded CUDA module SHA-256 is `09729aa0e90b5b95108a0f0ed66114a0371af0559a8f6152ae545311c19645d4`.
The loaded DSpark configuration SHA-256 is `dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd`.
The loaded DSpark weights SHA-256 is `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.

The independent review passed 368 checks on raw logs, intervals, source identity, loaded files, and completed-job cleanup.
Its scope is the paired job. The boundary job failed after the paired job because another task used the GPU.
The summary keeps the original parent failure and the completed paired-job success as different fields.
The envelope's `original.sha256` identifies the export input. The `raw_artifact` fields identify the full original records.

All three paired TPS confidence lower bounds are less than zero.
This comparison has no new DSpark TPS, TTFT, spread, or confidence-bound gate.
