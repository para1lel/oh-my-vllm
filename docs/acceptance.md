# Acceptance evidence

## Semantic IR candidate — formal gates pending

The candidate working tree passes 544 B200 GPU tests (including all six
maximum-context cases), 70 subtests and 303 CPU tests. The full GPU log is
`/tmp/oh-my-vllm-ir-full-gpu-green.log`; CPU log is
`/tmp/oh-my-vllm-ir-cpu-verified.log`. Formal operator and twelve-row
framework acceptance on a clean commit are pending; the historical results
below do not establish performance for the semantic IR candidate.

## Previous clean-source acceptance — 2026-09-29

Clean source `e3c42e0` passed the [full 147-case formal operator
matrix](../bench/baseline/2026-09-29-audit-final-operators.json) on B200 UUID
`GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`. All 147 output and frozen
TileLang comparisons pass, with three rounds of 20 interleaved pairs per case,
no recorded GPU interference, a smallest positive one-sided 95% gain bound of
`0.000005722664296627035 ms`, and a largest round spread of 1.139%. The
artifact records the clean source, compiler and loaded CUDA module identity,
all case decisions, and the SHA-256 of the raw `/tmp` collector output.

The same clean source passed the [12 framework-row
gate](../bench/baseline/2026-09-29-audit-final-framework.json) against the
frozen `2026-09-22-refreshed-enginecore.json` baseline: two complete warmups
and five measured repetitions per row, candidate spread at most 10%,
throughput at least 95%, TTFT at most 110%, and no measured-window capture or
compile. The initial complete collection accepted 11 rows; `prefix-32768-1`
had 23.924% TTFT spread and was rejected despite meeting the median limits.
A separate same-source, same-GPU repeat passed with 7.825% TTFT spread.
Both attempts and their raw hashes remain in the summary. The table uses the
accepted repeat for that one row; the original 12-row owner exit was failure,
not an accepted all-pass run.

| Row | TPS / baseline | TTFT / baseline | TPS spread | TTFT spread |
|---|---:|---:|---:|---:|
| mtp-32768-1 | 99.95% | 94.12% | 0.18% | 0.59% |
| mtp-32768-2 | 99.43% | 95.94% | 0.12% | 0.29% |
| mtp-32768-4 | 99.54% | 95.69% | 0.16% | 0.30% |
| ordinary-131072-1 | 114.10% | 92.17% | 0.22% | 1.13% |
| ordinary-131072-2 | 112.86% | 92.37% | 0.20% | 0.57% |
| ordinary-131072-4 | 109.66% | 93.83% | 0.05% | 0.13% |
| ordinary-32768-1 | 116.51% | 93.07% | 0.19% | 0.68% |
| ordinary-32768-2 | 117.94% | 96.40% | 0.07% | 0.68% |
| ordinary-32768-4 | 115.26% | 94.49% | 0.11% | 0.22% |
| prefix-32768-1 (repeat) | 117.67% | 82.47% | 0.15% | 7.82% |
| prefix-32768-2 | 124.58% | 82.31% | 0.07% | 7.30% |
| prefix-32768-4 | 116.90% | 75.70% | 0.11% | 3.52% |

The full B200 pytest suite on this source passed 491 tests and 70 subtests;
its 42 warnings are third-party deprecations and one DSL compile hint.
The six real 258048+4096 boundary cases remain separately evidenced below.
The sections that follow preserve evidence and pending statements as they stood
at their earlier commits.

## Earlier remediation evidence — 2026-09-28

Clean `bd8e21e` [EVD-07 boundary evidence](../bench/baseline/2026-09-28-audit-evd07-context-boundary.json)
passes all six ordinary/MTP4, batch 1/2/4 runs at 258048 input plus 4096
output tokens on one B200 UUID. Every row generates `batch_size * 4096`
tokens with zero preemptions and no OOM; MTP4 proposed/accepted draft totals
are 5010/2842, 9075/5917, and 16219/12318. The largest worker peak
reserved memory is 132441440256 bytes. The collector recorded clean source,
release binary hash, matching start/end identities and UUID, and successful
worker cleanup. Clean `8dfc97b`
[MTP HTTP evidence](../bench/baseline/2026-09-28-audit-evd07-long-context-http.json)
adds four real strict-JSON requests: chat/completions and responses, each
twice. Every prompt uses 131099 tokens, every response validates to
`{"n":123,"label":"verified"}`, and each request proposes 20 MTP drafts.
Repeat requests have 130928 cached tokens. The dedicated server exits, its
listener and IPC path close, and the selected GPU has no compute process.
EVD-07's requested boundary and HTTP evidence are complete; current full
GPU, 147-case formal, and 12-row framework gates remain pending.

Clean `6ba9046` and `231f066` passed all 16 selected attention cases before
and after KRN-10 on B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`.
The [before](../bench/baseline/2026-09-28-audit-krn10-attention-before.json)
and [after](../bench/baseline/2026-09-28-audit-krn10-attention-after.json)
summaries retain source and loaded-module hashes, output checks, comparison
rounds, and raw collector hashes. The after subset has
`selected_passed=true`; its top-level `passed=false` means the other 131
operator cases were not selected. After-source maximum three-round
spread/median was 0.173%, and the minimum one-sided 95% gain bound over
frozen TileLang was +0.0028209709 ms. All 16 after-source CUDA three-round
medians were lower than the before-source medians on the same UUID, with a
median relative difference of 7.26%; this across-commit comparison does not
isolate each individual code change or establish framework throughput.
Three earlier after attempts were rejected when external GPU processes
appeared. The full 147-case, full GPU, and 12-row framework gates remain
pending.

Reviewed `94c3473` passes seven isolated B200 KRN-09 fault/graph cases on
UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, plus five repeated
prior-fault subprocesses. They establish non-consuming normal launch checks,
debug observation phases, and graph-capture refusal. Three same-GPU old/new
eager host-call diagnostics had near-equal medians but isolated >10% spread
outliers; their cause is unproven, so they are not performance acceptance.
The current full GPU, formal-operator, and 12-row framework gates remain
pending.

Clean `40e3e57` passes all 66 selected KRN-08 affected formal cases on B200
UUID `GPU-1b174534-ebba-826f-b452-8e7f3c05c301`: 13 each for `norm`,
`add_norm`, `gated_norm`, and `convolution`; 8 `recurrent`; and 6 `qk`.
Every row passes output verification, records exactly one fast and zero generic
CUDA host dispatches, and beats frozen TileLang across three rounds of 20
interleaved pairs. The smallest positive one-sided 95% gain bound is
0.000000213335 ms. The largest three-round spread/median is 1.695%.
The [formal subset](../bench/baseline/2026-09-28-audit-krn08-operators.json)
has clean source identity and `selected_passed=true`; `passed=false` means it
does not cover all 147 cases. The owned GPU process exited. Current full GPU
and 12-row framework gates remain pending.

Clean `96e4ecc` passes all 29 selected affected formal operator cases on a
UUID-pinned B200: 13 `prepare_attention` and 16 `attention`. The smallest
positive one-sided 95% gain bound is 0.000796 ms. The summarized
[artifact](../bench/baseline/2026-09-28-audit-p1-operators.json) retains the
source hashes, frozen-reference hashes, protocol, selected cases, and raw
capture hash. This subset does not establish a new full 147-case result.

Clean `312c54b` passes all 60 selected `quant`, `silu_quant`, `gates`, and
`recurrent` cases after both KRN-04 entry points were corrected. Each case has
three rounds of 20 alternating pairs, 100 graph repetitions per sample,
passing output verification, and a positive one-sided 95% bootstrap gain
bound. The smallest gain bound is 0.00000699734 ms; the largest three-round
spread/median on either backend is 0.846%. The
[formal subset](../bench/baseline/2026-09-28-audit-mnt02-krn04-operators.json)
records B200 UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`, clean source
identity, and live nvcc/module hashes. Its `selected_passed=true` applies to
60/60 cases; `passed=false` reflects incomplete 147-case coverage. The first
attempt at clean `6bd2119` stopped at a historical KRN-04 false rejection
and is not acceptance evidence. A later clean `312c54b` attempt passed 59/60:
the 1248-token `gates` row had a negative bootstrap gain bound, with elevated
latency on both backends and no detected compute-process interference. Its cause
is unproven, so that attempt is rejected. A complete repeat passed 60/60 on
the same UUID, and an independent
[13/13 gates confirmation](../bench/baseline/2026-09-28-audit-gates-confirm-operators.json)
passed on UUID `GPU-80cebbaf-a106-2605-b902-f5f9a08645ea`. An initial
attempt to start that confirmation was stopped before measurement when an
external GPU process appeared. Full current GPU and 12-row framework gates
remain pending.

Clean `0394727` passes all 8 selected recurrent cases after the MNT-03 state
alignment check was expressed in vector bytes. The
[affected formal artifact](../bench/baseline/2026-09-28-audit-mnt03-recurrent-operators.json)
records B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, output
verification, 3 × 20 alternating pairs, and positive one-sided 95% bounds;
the smallest lower bound is 0.000064624 ms. The source is clean and
`selected_passed=true`; `passed=false` only reflects incomplete 147-case
coverage. The owned GPU process exited.

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
- At that time it recorded serving-availability and evidence-integrity gaps.
  The harness inferred FA/GDN capacity rather than observing it (EVD-02), and
  no automated test covered the boundary runs (EVD-07). These limit the
  historical `c36d1c9` evidence; later repairs and the clean EVD-07 six-row
  artifact are recorded in the current remediation section above.

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
