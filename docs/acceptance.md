# Acceptance evidence index


The [scale and stage diagnosis](../bench/evidence/2026-10-10-fp8-scale-stage-diagnosis.json) applies to subsequent dirty-source changes from `8d7d18b`.
Four affected full-operation cases passed numerical and speed checks. Thirty GPU projection checks passed.
The record keeps stage and full scale-layout comparisons, all profiler counters, and sixteen-output full-model diagnosis.
The full current-source phase, operator, capacity, and service collections stay pending.

The [CTA cluster diagnosis](../bench/evidence/2026-10-10-fp8-cluster-diagnosis.json) applies to subsequent dirty-source changes from `d4e0f1c`.
Fifteen affected full-operation cases passed numerical and speed checks. Eighteen GPU replay and layout checks passed.
The record compares thirty-five full operations with output allocation and bitwise output checks.
The record keeps Nsight counters, HIR estimates for R2 and N256, rejected candidates, and sixteen-output prefix-hit measurements.

## Phase-latency verification

The [gated projection diagnosis](../bench/evidence/2026-10-10-gated-projection-diagnosis.json) applies to dirty-source changes from `d95be77`.
Sixteen affected full-operation cases passed numerical, dispatch, and speed checks. Twenty gated GPU checks and one GPU twin check passed.
The record keeps twenty-four candidate rows, thirteen counters per launch, HIR estimates, and four sixteen-output model groups.
Prefix-hit batch 1 failed the two diagnostic gates. The full 317-case and fifteen-row collections stay pending.

The full fifteen-row collection for the current source is pending.
Use [requirements](requirements.md) for the phase gates and [testing](testing.md) for collection commands.

The [phase and regression progress record](../bench/evidence/2026-10-10-phase-progress.json) applies to source `c56ca40`.
Seven full rows used two warmups and five measurements, with 4096 output tokens per request.

Ordinary batches 1, 2, and 4, MTP4 batches 1, 2, and 4, and prefix-hit batch 1 all failed the prefill gate.
Prefill ratios were about 3.016, 3.063, 3.081, 3.007, 3.061, 3.073, and 3.846.

All seven prefill spreads were less than 10%.
The stored semantic-v3 model also failed MTP4 batch-4 decode.

The common-tail model has review by another agent. Subsequent collection must use its reviewed source contract.
The original measurements and verdicts stay the same.

The same source completed all 301 operator cases in three groups.
All numerical checks passed. Four speed checks failed: `add_norm-a8eca15b2f`, `add_norm_fp8_linear-4d4d440715`, `add_norm_fp8_linear-b0d226e89f`, and `gates-032c38d108`.

The GPU suite passed 383 tests. Nine context tests failed their external-process guard.
These failures do not give capacity results. The record keeps the full failed attempts and source scope.

The [owned-kernel and liveness diagnosis](../bench/evidence/2026-10-10-owned-tuning-liveness.json) keeps previous dirty-source experiments.
Eighteen copy/recurrent cases and twenty-one selected-output cases passed numerical and speed checks.

The [FP8 cache-traversal diagnosis](../bench/evidence/2026-10-10-fp8-cache-traversal.json) keeps a stable-source short-output experiment.
It used input 32768 and output 16, with two warmups and five measurements.

Its ordinary prefill median was `1.322744261 s`, with bound `0.44175381006472536 s` and ratio `2.994301873267811`.
Its wall spread was `0.010685699735574266`.

Filtered operator coverage and short outputs have diagnostic scope.
The full 317-case matrix and fifteen phase workloads stay active.


The [dispatch diagnosis](../bench/evidence/2026-10-10-dispatch-diagnosis.json) keeps subsequent dirty-source operator and three-case DSpark capacity measurements.
These filtered and short-output records have diagnostic scope.


Forty-eight affected operator cases passed numerical and speed checks with the full per-case protocol.
Their source was dirty. The record keeps the previous failed and interrupted attempts.

## DSpark implementation and acceptance

The optional DSpark mode keeps fixed checkpoint weights and shares the target embedding and vocabulary head.
Its cumulative confidence threshold is `0.2`. Set it to `0.0` to stop confidence truncation.
Output, context, and grammar limits still apply.

The [threshold-selection record](../bench/evidence/2026-10-08-dspark-confidence-selection.json) uses inputs different from the formal synthetic workload.
It has two text/tool-history requests, with input 32768 and output 512 per request.
Each setting has two full warmups and two measured repetitions.
Thresholds `0.0`, `0.05`, `0.1`, and `0.2` have TPS medians 177.287, 183.097, 184.198, and 184.323.
Their acceptance rates are 28.487%, 44.957%, 53.406%, and 62.500%. These medians apply to this small collection.

### Current runtime paired comparison

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

### Operators

The [operator record](../bench/evidence/2026-10-09-dspark-operators.json) passed all 230 cases with output and speed checks.
It keeps the 147 original cases and eight pinned TileLang references, with 83 new cases.
Each case has three sets of twenty pairs and one hundred graph repetitions per sample.

All 690 CUDA time medians are less than the TileLang medians for the same case and set. Each one-sided 95% time-saved bound is positive.
The smallest bound is `0.00000997330993413926 ms` for `quant-5b659364da`.
An independent review checked full graph operations, shared Q/K output pools, unchanged inputs, and sequence hashes.

### Current runtime context boundary

The [boundary record](../bench/evidence/2026-10-09-dspark-context-boundary.json) passed all nine mode/batch combinations on `198b906`.
Each request had input 258048 and output 4096, for total length 262144.
Each row had `batch_size * 4096` kept output tokens, zero recompute preemptions, and no OOM.
The table separates peak allocated memory from peak reserved memory.

| Mode | Batch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|
| ordinary | 1 | 116.300 | 118.879 |
| ordinary | 2 | 116.969 | 120.975 |
| ordinary | 4 | 118.853 | 122.525 |
| mtp4 | 1 | 112.695 | 114.611 |
| mtp4 | 2 | 113.411 | 118.797 |
| mtp4 | 4 | 115.291 | 127.305 |
| dspark | 1 | 134.184 | 138.326 |
| dspark | 2 | 134.948 | 138.854 |
| dspark | 4 | 136.871 | 150.906 |

### Service and test scope

The [current service record](../bench/evidence/2026-10-09-dspark-service.json) applies to runtime `198b906` and passed six client groups.
It passed twelve constraint cases and Chat/Responses lifecycle, oh-my-pi tool-loop, and long-context checks.
The service admitted 52 requests. Fifty completed. Two cancellation tests aborted their requests. All 52 requests were released.

Chat used 13 tools and 8 HTTP model requests. Responses used 10 tools and 6 HTTP model requests.
All model requests returned HTTP 200. Tool arguments are complete.
Tool results stay the same in the first and final model requests after each tool call.
The two clients read the full `schedule` definition at lines 236-400 and worker initialization at lines 103-125.

The Python examples are the same as the source.
The Rust examples keep all executable lines, with 1 comment line removed for Chat and 3 for Responses.
Chat gives 397 as the Rust end line, uses check commands without all required flags, and cites some documents without direct reads.
Its semantic review keeps these errors. The Responses review found no important factual error, with comment and document-citation scope limits.
Initial AGENTS context was available and correct for the two clients.

Four long-context requests each had input 131099 and output 21, with strict JSON `{"n":123,"label":"verified"}`.
The first Chat request had zero cached tokens. The other three had 130928 cached tokens.
Each request verified 24 drafts and accepted 16. All task-owned service processes stopped, and the temporary listener was released.

The independent suite review passed 440 checks. Chat and Responses have different semantic reviews.
The startup record has `vllm_importable=false`. The post-execution loaded-module audit has no observed success record.

The previous DSpark service collection on `de1591b` and full GPU collection on `46f7529` keep their historical source scope.
The current graph-budget regression passed 548 CPU tests and 70 subtests, with 323 GPU tests deselected.
Four GPU graph tests passed. The full-output fallback check passed again in 1 test.
These graph-budget tests use a small model. The current full-model boundary collection supplies the nine capacity checks above.

The full 323-test GPU selection was not run again after the graph-budget changes.
Graph-budget tests, full operator cases, full-model boundaries, and service checks test the changed paths.
