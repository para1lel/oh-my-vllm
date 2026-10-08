# Acceptance evidence index

This index identifies measurements, source scope, and limitations.
Tracked [portable evidence](../bench/evidence/README.md) is derived historical data for reference and offline analysis.
Each summary records its original hash (SHA-256) and removed fields.
Full original records stay in ignored local storage or the Git history.
Formal comparisons must use original records with matching conditions.

New servers must measure their own matching baseline.

## Latest full performance acceptance

Source: `619c9d98809c00081c5afb549987cde1cbd71690`, clean during collection on 2026-09-29.
Baseline source: `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
Original baseline SHA-256: `fa3729f1a2ce160235b45df75542774d628dac7af963f01d673353fb419df8fd`.
Release binary SHA-256: `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
Loaded CUDA module SHA-256: `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`.

The [operator summary](../bench/evidence/2026-09-29-ir-operators.json) has 147 passing output and pinned-TileLang cases.
Each case has three rounds of twenty alternating pairs and one hundred graph repetitions per sample.
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

Three earlier full prefix-batch 1 attempts failed TTFT spread at 17.378%, 20.976%, and 20.177%.
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
The [decision records](README.md#decision-records) show useful replacement relationships.
See [current work](handoff.md) for this task's checks and open work.
