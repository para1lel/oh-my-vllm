# Acceptance evidence

## Current remediation evidence — 2026-09-28

Clean `96e4ecc` passes all 29 selected affected formal operator cases on a
UUID-pinned B200: 13 `prepare_attention` and 16 `attention`. The smallest
positive one-sided 95% gain bound is 0.000796 ms. The summarized
[artifact](../bench/baseline/2026-09-28-audit-p1-operators.json) retains the
source hashes, frozen-reference hashes, protocol, selected cases, and raw
capture hash. This subset does not establish a new full 147-case result.

The full B200 pytest suite after `96e4ecc` passed 211 tests and 32 subtests,
with six skipped context-boundary placeholders. SRV-03's later CPU/protocol
repair `d33b844` passed 31 focused Python tests and 26 subtests, Rust
workspace tests, formatting, Ruff, and Clippy; the full GPU suite was not
rerun for that commit. Current 12-row throughput and TTFT remain unverified
after the hot-path changes. The 2026-09-22 results below apply to `c36d1c9`.

## Historical accepted CUDA baseline — 2026-09-22/23

**Performance.** Clean `c36d1c9`, with CUDA explicitly selected, passes all 12
frozen-vLLM comparisons.

- **Workloads:** ordinary, MTP4 and prefix at 32768 input, batch 1/2/4, plus
  ordinary-only at 131072 input, batch 1/2/4. Every output is 4096 tokens.
- **Protocol:**
  - `benchmarks/ttft.py` with 2 complete warmups and 5 measured repetitions.
  - The frozen baseline's CPU affinity.
  - A UUID-pinned B200 with process monitoring.
  - A steady-state compilation/capture audit.

| Workload | Baseline tok/s | CUDA tok/s | Throughput ratio | TTFT ratio |
|---|---:|---:|---:|---:|
| ordinary-32768-1 | 90.10 | 105.30 | 116.86% | 93.14% |
| ordinary-32768-2 | 163.50 | 193.08 | 118.09% | 95.60% |
| ordinary-32768-4 | 288.41 | 331.64 | 114.99% | 96.37% |
| mtp-32768-1 | 329.36 | 329.24 | 99.96% | 93.92% |
| mtp-32768-2 | 497.37 | 490.52 | 98.62% | 94.81% |
| mtp-32768-4 | 729.62 | 711.53 | 97.52% | 97.50% |
| prefix-32768-1 | 93.01 | 109.10 | 117.30% | 82.96% |
| prefix-32768-2 | 173.52 | 217.06 | 125.09% | 85.02% |
| prefix-32768-4 | 325.31 | 380.50 | 116.97% | 76.01% |
| ordinary-131072-1 | 72.35 | 82.68 | 114.29% | 92.95% |
| ordinary-131072-2 | 114.06 | 129.45 | 113.49% | 91.54% |
| ordinary-131072-4 | 162.85 | 178.82 | 109.80% | 93.82% |

- **Stability:** every row meets throughput ≥ 95%, TTFT ≤ 110% and
  `(max-min)/median` ≤ 10% for both metrics on both engines. Process polling
  cannot exclude arbitrarily short interference between samples.
- **Prior attempts:**
  - Prefix batch 1 failed stability twice and prefix batch 2 once. Complete
    repeats pass with the stability rule unchanged.
  - Three passing sets shared physical CPU cores with other runs. They were
    conservatively replaced by complete repeats.
  - The cause of the prefix TTFT jitter was not established; a temporary GC
    diagnostic found no collections in measured intervals and is not acceptance.
  - All attempts are preserved.
- **Baseline:** the vLLM baseline was not rerun.
- **Nature of the comparison:** this compares independent runs and framework
  versions, not a same-source paired experiment.
- **Margins:** the MTP rows have the smallest margins. MTP batch 4 is 2.52
  points above the gate.

**Operators (REQ-KERNEL-002).** All 147 formal static cases beat the frozen
TileLang implementation.

- **Rule:** three warm rounds of 20 interleaved pairs, a faster CUDA median in
  every round, and a positive one-sided 95% bootstrap bound on time saved.
- **Margins:** small gains can be nanoseconds; no minimum margin is claimed.
- **Separate from latency:** TileFoundry HIR estimates and Nsight counters are
  kept separate from acceptance latency.

**Correctness and features.**

- **Test suite:** with CUDA as the default and no backend/CUDA_HOME/TVM overrides,
  the suite passes 174 tests plus 31 subtests. There are no skips and no relaxed
  tolerances.
- **FP64 probes:** actual-model ordinary and MTP4 probes pass.
- **Real text:**
  - Ordinary and MTP4 real-text checks pass, including forced preemption and
    prefix reuse.
  - Twelve MTP4 constraint cases, the thinking levels and the lifecycle checks
    pass.
  - Four long strict-JSON calls pass.
- **Boundary runs:** six 258048-input, 4096-output runs complete without OOM or
  recompute preemption:

  | Mode | Batch | Output tok/s | Proposed / accepted drafts |
  |---|---:|---:|---|
  | ordinary | 1 | 56.25 | — |
  | ordinary | 2 | 78.58 | — |
  | ordinary | 4 | 96.65 | — |
  | MTP4 | 1 | 89.40 | 5010 / 2842 |
  | MTP4 | 2 | 111.66 | 8989 / 5939 |
  | MTP4 | 4 | 126.06 | 16272 / 12306 |

  Peak PyTorch reserved memory is 134.69 GB. This is not whole-device usage, and
  these runs are capacity diagnostics, not throughput gates.
- **oh-my-pi readback (clean `ef07b2b`):** both APIs pass the core task.
  - Chat and Responses each made nine successful reads, with 5 and 3 model
    requests respectively.
  - Proposed/accepted drafts were 3297/1943 (Chat) and 3687/2164 (Responses).
  - Some answer inaccuracies are recorded in the feature artifact. There is no
    claim of perfect grounding.
- **Cleanup:** cleanup is verified independently of the model answers.

**Provenance.**

- The 147/12 timing comes from `c36d1c9`.
- `030f60f` only changes the default selection and identity reporting. The
  kernel, model and dispatch implementations are unchanged.
- The default-selection suite and service runs used the working tree after
  `ef07b2b` (recorded with its diff hash in the feature artifact), which became
  `030f60f`; they were not rerun on the commit itself.

**Evidence files** (in `bench/baseline/`):

- `2026-09-22-cuda-operators.json`
- `2026-09-22-cuda-framework.json`, whose rows carry `baseline_sha256` values
  matching the embedded artifacts in `2026-09-22-refreshed-enginecore.json`
- `2026-09-22-cuda-features.json`

**Scope limits.**

- The 2026-09-23 [code audit](audit-2026-09-23.md) found no wrong-token defect
  on this path.
- It did record serving-availability and evidence-integrity gaps. Two in
  particular:
  - The harness sets the candidate configuration it compares, instead of
    observing it (EVD-02).
  - No automated test covers the boundary runs (EVD-07).
- Those limitations apply to the evidence above.

## Historical evidence index

Earlier stages are superseded. Their raw artifacts remain in `bench/baseline/` as
records of what was measured at the time. Do not cite them as current status, and
do not use pre-2026-09-22 baselines as the current denominator.

| Date | Stage | Result at the time | Artifacts |
|---|---|---|---|
| 2026-09-19 | Original GPUWorker adapter, 9 rows vs original EngineCore baseline (2 warmups/3 reps) | 9/9 ≥ 95% | `2026-09-19-acceptance.json`, `2026-09-19-mtp-bs2-repeat.json`, `2026-09-19-batch-text.json`, `2026-09-19-paired-investigation.json` |
| 2026-09-21 | Serving extension (real MTP4 constraints, lifecycle, oh-my-pi) | passed | `2026-09-21-serving-acceptance.json` |
| 2026-09-21 | V2 Model Runner, 9 rows vs original baseline | 9/9 ≥ 95% | `2026-09-21-v2-acceptance.json` |
| 2026-09-21 | Independent runtime (vLLM removed), 9 rows vs original baseline | 9/9 ≥ 95% | `2026-09-21-independent-{stage1,stage2,stage3,acceptance,batched-sampling,grouped-decode,proposal-graph}.json` |
| 2026-09-22 | Refreshed official vLLM baseline `e9f169d1…`, 12 rows | frozen denominator | `2026-09-22-refreshed-enginecore.json` |
| 2026-09-22 | TTFT/long-context investigation and MTP-gap diagnosis | 10/12; MTP bs4 at 94.87% | `2026-09-22-ttft-{stage1,agentic,discarded}.json`, `2026-09-22-performance-gap{,-candidates}.json`, `2026-09-22-{native-decode-fusions,prefill-*,ragged-prefill,strided-metadata}.json` |
| 2026-09-22 | TileLang migration (`da75c02`), 12 rows | 12/12 under 10% stability | `2026-09-22-tilelang-{acceptance,correctness}.json` |
| 2026-09-22 | CUDA milestone subsets (dirty source, diagnostic) | partial | `2026-09-22-cuda-*-progress.json` |

The `baseline_*`, `clean_*` and `mtp_*` files and `logs/` are early vLLM HTTP
serving captures. Their settings, inputs, timing boundaries and MTP configuration
differ from the EngineCore protocol, so they are not comparable with it. Some
older artifacts carry the erroneous "Qwen3.5" label. The model has always been
Qwen3.8-27B-FP8.

## Operational scope

Multi-GPU, CPU KV swap, multimodal input, LoRA and production deployment are out
of scope. The OpenAI-compatible HTTP service is implemented. Its acceptance
consists of the functional and agentic checks above, not an HTTP throughput gate.
EngineCore measurements do not validate HTTP-path performance.
