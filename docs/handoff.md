# Handoff — oh-my-vllm

Updated 2026-09-19. Baseline: 94b827f. Pre-existing user change: `.gitignore`.
The previously documented two unstaged worker fixes are already in HEAD.

## Current work

Stage 1: documentation, conda entrypoints, idle B200 selection, timestamped logs.
55 Rust tests, clippy with all features/-D warnings, Python format/check,
two CPU runtime tests and real Python -m bridge logging test passed.
Independent review passed after fixing logger entrypoint, async tracing and
documentation findings. GPU smoke is waiting
for an idle B200; all eight cards are occupied by an unrelated process. No GPU
accuracy or performance check has been run in this stage.

## Accepted delivery plan

1. Diagnostic/environment foundation; review and commit.
2. Actual-path GQA/GDN FP64 tests and coherent text; all ordinary workloads.
3. Complete MTP, prefix caching, continuous batching, recompute preemption and
   combination tests, with milestone reviews and commits.
4. Fair measurements for bs=1/2/4, 32768 input / 4096 output, ordinary/MTP/prefix
   modes: median output throughput >=95% of matching vLLM, minimum three runs.

Rust owns scheduling/KV; Python uses GPUWorker in the existing vllm conda env.
Use the oh-my-vllm conda env for framework work, no uv. Single GPU tests wait
for an idle B200 and pin its UUID. Never interrupt unrelated GPU processes.

## Unverified / risks

No end-to-end forward or performance acceptance has passed yet. Existing
standalone GQA test does not establish coverage of the actual inference path.
The benchmark has unequal inputs, sampling and warmup cache state and must be
repaired before its results can be used. Internal GPUWorker API adaptation,
cache allocation and request lifecycle remain to be verified.
