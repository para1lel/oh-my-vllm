# Handoff — 2026-09-22

## Active task: TTFT and long context

User approved a new 12-row throughput/TTFT baseline on latest official vLLM main,
plus 262144-total-token ordinary/MTP4 boundary checks at batch 1/2/4. See current
requirements. New acceptance is pending; the results below describe the prior task.
Frozen upstream SHA: e9f169d16b9408bb9ae44f75072b91a5521d733c.
The prior local vLLM telemetry commit was preserved in
/data0/shared/dongwu.chen/vllm-before-ttft-20260922.bundle before updating main.
Baseline environment adaptation is complete. The exact cu130 wheel metadata
was initially unavailable, so a source build began. Official wheel publication
completed at 02:13 UTC; rechecking metadata succeeded and the owned source build
was stopped before compilation. Installed the official frozen-commit wheel via upstream editable mode: version
0.29.1rc1.dev505+ge9f169d16.precompiled, Torch2.13.0+cu130. pip check and
vllm._custom_ops imports pass. See /tmp/oh-my-vllm-baseline-install-official.log
and /tmp/oh-my-vllm-ttft-final/baseline-environment.json for extension hashes.

Implemented, not yet accepted: per-request TTFT, optional independent GDN pool,
formal measurement/provenance checks, and corrected Qwen3.8 naming. 77 Rust tests
pass. Final complete GPU pytest passes 100 tests and 12 subtests without skips,
after rebuilding the debug binary for the new model ID. Formatting, clippy and
all pre-commit checks pass.
131072-input batch4 short-output diagnostic completed without OOM/preemption.
The first full 262144 boundary attempt exposed signed-int32 decode page-offset
overflow; widened before multiplication and added FP64 high-page tests (9 pass).
All six ordinary/MTP4 batch 1/2/4 boundary checks finished: input 258048 +
output 4096, zero preemptions. Peak reserved memory is 134951731200 bytes ordinary
and 140033130496 bytes MTP4. These cold diagnostic runs are not formal performance
measurements. Actual-model ordinary/MTP FP64 probes retain original tolerances.
Twelve MTP constraint cases, service lifecycle and both APIs with 131099 input
tokens pass; repeat requests reuse 130928 cached tokens. All owned boundary/probe/
service workers exited and port 18015 is released. See
bench/baseline/2026-09-22-ttft-stage1.json for raw boundary results and log hashes.
Both oh-my-pi APIs reran with Qwen3.8 model ID and MTP4: Chat/Responses made
10/8 successful reads, including both required source files, with zero tool errors.
Elapsed times were 28.02/22.16 seconds. Their GB/GiB unit error and other answer
caveats are retained in bench/baseline/2026-09-22-ttft-agentic.json. Server/worker
exited and port 18016 is released; unrelated GPU processes remain untouched.
Prefill profiling found inefficient per-row FP8 quantization. A 16-row tile
reduces observed quantization GPU time from 295 to 51 ms in a diagnostic model
run. Arithmetic is unchanged; small/decode shapes retain the old kernel. Complete
suite before final threshold tuning: 103 tests + 12 subtests; final boundary/FP8
tests: 14 pass. See 2026-09-22-prefill-quantization.json for limitations and hashes.
Post-optimization ordinary/MTP actual-model FP64 checks also pass. Four initial
baseline attempts were aborted because an external TP4 job entered their GPUs;
see 2026-09-22-ttft-discarded.json. Those results are excluded. The collector now
streams to disk so interference/timeout retains child output. Eleven baseline rows have passed collection audits; remaining
rows are collecting/retrying in /tmp/oh-my-vllm-ttft-final. Prefix4 attempt103
was rejected for TTFT spread 13.94% despite stable throughput. User confirmed to
keep waiting/retrying when external GPU interference occurs. No complete 12-row
comparison exists yet. Candidate MTP attempt0 was canceled during first warmup
because copying FlashInfer caches caused expensive recompilation; no measured
result from that attempt is used. Future candidates use the existing independent
FlashInfer cache and auditable per-run Triton cache.

Long-prefill attention now compacts active KV for independent ragged TRT-LLM
attention, retaining persistent 784-token pages and small-query FA2. Full suite
110 tests +12 subtests and both actual-model FP64 paths pass. Ordinary/MTP4 bs4
maximum-context reruns pass without preemption, with unchanged peak reserved
memory. See 2026-09-22-ragged-prefill.json. Final numerical checks, including
mixed long-decode isolation, pass (3 tests). Finish baseline
collection, performance fixes and reviews before claiming completion.

Prefill convolution/SiLU tiling and dual-SM large-projection GEMM reduce a
diagnostic GPU profile from 1.816 to 1.721 seconds. Full GPU suite passes
113 tests +12 subtests; actual-model MTP target/draft FP64 checks pass.
See 2026-09-22-prefill-tiles.json. First candidate rows exposed shared Triton
cache audit contamination, so subsequent runs must use separate Triton roots.
The 131072-input batch1 candidate on 0ad2bb4 passed both gates; other first
rows are diagnostic only. MTP throughput remains below threshold. Isolated
baseline step-count diagnostic found 898 steps versus our 899, pointing to
per-step cost rather than acceptance rate. Alternate private GEMM tactics
triggered illegal access in a temporary prototype and are not used in runtime.

Packed GDN/convolution/RMS inputs now retain their token/head strides; target
and draft metadata transfers are grouped. Review requested a single int32 starts
conversion per batch to avoid repeated FlashInfer casts; implemented. GPU suite
120 tests +12 subtests, actual MTP FP64, all 12 service constraints, lifecycle
and 131099-input prefix checks pass. Service18017 exited/released. Diagnostics
improved MTP bs1/2 to305.86/450.65 tok/s, still below their new baseline gates.
See 2026-09-22-strided-metadata.json. Prefix2 baseline continues to retry after
variance failures and repeated unrelated GPU entrants; user authorized retries.

## Previous task: independent runtime

The independent runtime is implemented and all nine performance rows pass the
original frozen EngineCore baseline at 95% or better. Both final oh-my-pi
workflows completed with MTP4 against the reconciled documentation.

Rust owns HTTP serving, scheduling and logical KV. Python loads and runs Qwen
using project-owned code, PyTorch/Triton/FlashInfer and XGrammar. There is no
installed/imported/linked vLLM, old source/environment dependency or legacy runner.
Use scripts/with-env.sh and scripts/with-gpu.sh. Target model, single B200,
block 784 and the conda oh-my-vllm environment remain unchanged. Dependencies are
pinned in requirements/runtime.txt; extension scope is documented in ADR-006.

## Previous task verification

- GPU pytest 92 tests and 12 subtests pass, with no skips; Rust 75 tests pass.
- fmt, hard Rust line width, clippy, ruff and all pre-commit checks pass.
- Actual-model ordinary/MTP FP64 checks retain the original tolerances; grouped
  target/draft attention and graph buffer lifetime/cache restoration also pass.
- Distinct-city text tests pass through prefix reuse, staggered arrivals and
  recompute preemption. Twelve real MTP constraint cases and service lifecycle
  (mixed batch, continuation/delete, disconnect and release) pass.
- Six ordinary/prefix rows use clean ca5a200. Three MTP rows use clean 5c0848e;
  only MTP execution changed between them. Existing graph definitions are
  AST-identical and other non-MTP source files are unchanged. Per-row identities,
  raw repetitions and hashes are in the final independent acceptance artifact.
- Source/runtime/file-access audits found no old vLLM dependency. Kernel caches
  belong to this project. No original EngineCore baseline was rerun or replaced.

See acceptance.md and bench/baseline/2026-09-21-independent-acceptance.json for
numbers and evidence. Intermediate artifacts retain failed performance rows,
external-GPU-contaminated attempts, two CPU-overlap exclusions and an isolated
prototype buffer-lifetime failure fixed before production. Diagnostics are never
substituted for formal repetitions.

## Previous task agentic verification and cleanup

Chat / Responses completed 7 / 18 successful read calls, including both required
source files, with zero tool errors and real follow-up requests. End-to-end times
were 26.27 / 36.68 seconds. Both answers identify the project and completion state;
minor Responses wording/count inaccuracies are explicitly retained in acceptance.md
and the JSON evidence. Do not claim perfect model-answer grounding.

No work remained in the previous independent-runtime scope. All owned
GPU programs exited; nvidia-smi has no compute processes and ports 18013/18014 are
released. No service was requested to remain running.

Develop and commit only on main, stage intentional files, keep hooks enabled and
retain the required attribution. Never leave task-owned GPU programs running.
