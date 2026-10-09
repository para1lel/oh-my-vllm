# Acceptance evidence index

## DSpark paired measurements and open acceptance

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
The initial twelve ordinary/MTP4 workload gates and full operator gates stay active.
The [attempt record](../bench/evidence/2026-10-09-dspark-attempt-history.json) keeps failed collections, diagnostics, original hashes, and verification limits.

### Service and capacity scope

The [previous DSpark service record](../bench/evidence/2026-10-09-dspark-service-de1591b.json) applies to runtime `de1591b`.
It passed six client groups with Chat and Responses: constraints, lifecycles, oh-my-pi tool loops, and long-context prefix reuse.
Its summaries keep source-read limits and factual errors in model answers.
The previous full GPU collection on `46f7529` passed 285 tests and nine 262144-token boundary cases.

Operator, boundary, twelve-row framework, and service acceptance for the current runtime wait for GPU availability.
Collectors reject GPU contention. Original failed attempts and completed subtask records stay available with their source identities.

The current graph-budget regression passed 548 CPU tests and 70 subtests, with 323 GPU tests deselected.
Four GPU graph tests passed. The full-output fallback check passed again in 1 test.
It uses a small test model. Full-model capacity and service checks will run again on the current runtime.

## Historical acceptance

This index identifies measurements, source scope, and limitations.
Tracked [portable evidence](../bench/evidence/README.md) is derived historical data for reference and offline analysis.
Each summary records its original hash (SHA-256) and removed fields.
Full original records stay in ignored local storage or the Git history.
Formal comparisons must use original records with matching conditions.

New servers must measure their own matching baseline.

## Full performance acceptance on 2026-09-29

Source: `619c9d98809c00081c5afb549987cde1cbd71690`, clean during collection on 2026-09-29.
Baseline source: `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
Original baseline SHA-256: `fa3729f1a2ce160235b45df75542774d628dac7af963f01d673353fb419df8fd`.
Release binary SHA-256: `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
Loaded CUDA module SHA-256: `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`.

The [operator summary](../bench/evidence/2026-09-29-ir-operators.json) has 147 passing output and pinned-TileLang cases.
Each case has three rounds of twenty pairs and one hundred graph repetitions per sample.
The mode order changes with each pair.
The smallest positive one-sided 95% gain bound is `0.0000036639670530955112 ms`.
Original external operator collector SHA-256 is `c6cdce051ea8fb50ec2e0adecde2ff0d132bd78456f829d14a3a3f49db3a5ab1`.

The [framework summary](../bench/evidence/2026-09-29-ir-framework.json) has twelve passing rows.
Each accepted row has two full warmups and five repetitions, no preemption, and a passing observed steady-state audit.
Each row uses one B200 GPU.
The table records ratios and spreads from the accepted full sets:

| Row | TPS/baseline | TTFT/baseline | TPS spread | TTFT spread |
|---|---:|---:|---:|---:|
| mtp-32768-1 | 100.67% | 94.22% | 0.12% | 0.88% |
| mtp-32768-2 | 99.53% | 95.74% | 0.20% | 0.70% |
| mtp-32768-4 | 99.27% | 95.26% | 0.24% | 0.25% |
| ordinary-131072-1 | 115.28% | 91.01% | 0.04% | 0.19% |
| ordinary-131072-2 | 113.78% | 91.65% | 0.11% | 0.37% |
| ordinary-131072-4 | 110.34% | 93.13% | 0.12% | 0.50% |
| ordinary-32768-1 | 118.15% | 92.98% | 0.10% | 0.95% |
| ordinary-32768-2 | 118.62% | 95.91% | 0.07% | 0.68% |
| ordinary-32768-4 | 115.82% | 94.21% | 0.20% | 0.40% |
| prefix-32768-1, fourth attempt | 118.86% | 83.64% | 0.16% | 5.02% |
| prefix-32768-2 | 125.63% | 84.55% | 0.28% | 3.64% |
| prefix-32768-4 | 117.53% | 76.92% | 0.17% | 4.72% |

Three previous full prefix-batch 1 attempts failed TTFT spread at 17.378%, 20.976%, and 20.177%.
The fourth full 2+5 set passed.
Six additional attempts stopped because of unrelated GPU processes.
The summary keeps rejected attempts and raw/log hashes.
The isolated prefix TTFT spike cause stays unknown.

The audit cannot exclude silent in-memory recompilation or interference shorter than process polling intervals.

This source also passed 544 GPU tests, seventy subtests, and 303 CPU tests, with six maximum-context cases.
Subsequent IR correctness changes passed 549 GPU tests, seventy subtests, and 308 CPU tests.
The full 147-case and twelve-row performance collections were not repeated after those changes.
These historical results do not show current-HEAD performance.

## Context and service evidence

| Evidence | Source and result |
|---|---|
| [Six context rows](../bench/evidence/2026-09-28-audit-evd07-context-boundary.json) | `bd8e21e5607387b081e1d4494bc7b8fdf79ac8d4`: ordinary/MTP4, batch 1/2/4, input 258048, output 4096, no OOM/preemption. |
| [Long-context HTTP](../bench/evidence/2026-09-28-audit-evd07-long-context-http.json) | `8dfc97b544d18e21f0856a2b2b4098bea90c8be5`: Chat/Responses each twice, strict JSON and MTP. |
| [Target-model MTP service and OMP](../bench/evidence/2026-09-22-ttft-agentic.json) | The two APIs completed tool-result roundtrips with MTP and successful source reads. |

The six context rows output `batch_size * 4096` tokens each.
MTP proposed/accepted totals are 5010/2842, 9075/5917, and 16219/12318.
Maximum worker reserved memory is 132441440256 bytes.
These numbers apply to the identified boundary source.

The long-context HTTP prompts contain 131099 tokens each.
Each returns `{"n":123,"label":"verified"}` and proposes twenty drafts.
Repeats have 130928 cached tokens.
The dedicated server, listener, IPC path, and worker were cleaned up.
Repeated same-shape 262144 and low-headroom probes have narrower scope than cross-shape eviction/recapture.

See [audit](audit.md).

## Key predecessor evidence

| Decision or milestone | Kept summary |
|---|---|
| Initial adapter and MTP design | [2026-09-19](../bench/evidence/2026-09-19-acceptance.json) |
| Independent runtime | [2026-09-21](../bench/evidence/2026-09-21-independent-acceptance.json) |
| Pinned TileLang | [2026-09-22](../bench/evidence/2026-09-22-tilelang-acceptance.json) |
| Refreshed twelve-row denominator | [EngineCore baseline](../bench/evidence/2026-09-22-refreshed-enginecore.json) |
| CUDA implementation | [Operators](../bench/evidence/2026-09-22-cuda-operators.json), [framework](../bench/evidence/2026-09-22-cuda-framework.json) |
| Closed-audit source `e3c42e0` | [Operators](../bench/evidence/2026-09-29-audit-final-operators.json), [framework](../bench/evidence/2026-09-29-audit-final-framework.json) |

Early measurements use different source/configuration and sometimes only three repetitions.
They apply only to their measured source and configuration.
All initial tracked records have byte-preserving local archives and hashes in [originals.json](../bench/evidence/originals.json).
The [decision records](README.md#decision-records) give records of implementation replacements.
See [current work](handoff.md) for this task's checks and open work.
