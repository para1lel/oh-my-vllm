# Handoff — 2026-09-21

## Current state

The independent runtime is implemented and all nine performance rows pass the
original frozen EngineCore baseline at 95% or better. Both final oh-my-pi
workflows completed with MTP4 against the reconciled documentation.

Rust owns HTTP serving, scheduling and logical KV. Python loads and runs Qwen
using project-owned code, PyTorch/Triton/FlashInfer and XGrammar. There is no
installed/imported/linked vLLM, old source/environment dependency or legacy runner.
Use scripts/with-env.sh and scripts/with-gpu.sh. Target model, single B200,
block 784 and the conda oh-my-vllm environment remain unchanged. Dependencies are
pinned in requirements/runtime.txt; extension scope is documented in ADR-006.

## Verified

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

## Agentic verification and cleanup

Chat / Responses completed 7 / 18 successful read calls, including both required
source files, with zero tool errors and real follow-up requests. End-to-end times
were 26.27 / 36.68 seconds. Both answers identify the project and completion state;
minor Responses wording/count inaccuracies are explicitly retained in acceptance.md
and the JSON evidence. Do not claim perfect model-answer grounding.

No implementation or regression work remains within the agreed scope. All owned
GPU programs exited; nvidia-smi has no compute processes and ports 18013/18014 are
released. No service was requested to remain running.

Develop and commit only on main, stage intentional files, keep hooks enabled and
retain the required attribution. Never leave task-owned GPU programs running.
