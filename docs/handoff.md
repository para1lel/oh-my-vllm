# Handoff — 2026-09-22

## Active task: TTFT and long context

User approved a new 12-row throughput/TTFT baseline on latest official vLLM main,
plus 262144-total-token ordinary/MTP4 boundary checks at batch 1/2/4. See current
requirements. New acceptance is pending; the results below describe the prior task.
Frozen upstream SHA: e9f169d16b9408bb9ae44f75072b91a5521d733c.
The prior local vLLM telemetry commit was preserved in
/data0/shared/dongwu.chen/vllm-before-ttft-20260922.bundle before updating main.
Baseline environment adaptation is in progress: exact cu130 precompiled wheel is
unavailable, so pip is building the frozen source with host CUDA13.1. Logs:
/tmp/oh-my-vllm-baseline-source-build.log. Do not use stale installed metadata as
proof of the new compiled runtime until installation and import checks finish.

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
No formal baseline or 12-row performance result exists yet. Finish baseline
collection, performance fixes and reviews before claiming completion.

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
