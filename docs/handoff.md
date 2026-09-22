# Handoff — 2026-09-22

## Current CUDA migration status

Use this section for current state. Older milestone counts in
`cuda-development.md` and progress artifacts are historical, not current results.

The native CUDA implementation at clean `c36d1c9` passes all **147** statically
derived operator configurations against the frozen TileLang reference. Each case
passes three independent warm rounds of twenty alternating-order pairs and a
positive one-sided95% paired-bootstrap gain bound. The previous173-case inventory
predates production full-attention preparation fusion and is no longer the active
matrix. Frozen TileLang has a single implementation in `kernels/tilelang_reference`.

Full CUDA correctness passes **173 tests plus28 subtests**, without relaxed
references or tolerances. Actual-model ordinary/MTP4 eager FP64 probes pass.
Ordinary/MTP real-text isolation, prefix reuse and forced recompute preemption pass.
Six input258048/output4096, ordinary/MTP batch1/2/4 boundary runs finish with exact
output counts and zero OOM/preemption.

The12-row framework matrix uses ordinary/MTP/prefix input32768 at batch1/2/4,
plus ordinary-only input131072 at batch1/2/4; output is4096 throughout. It compares
only against frozen vLLM `e9f169d16b9408bb9ae44f75072b91a5521d733c`. No vLLM runtime,
source/environment dependency or new baseline run was introduced. All12 rows pass
throughput>=95%, TTFT<=110% and spread/median<=10%.
Ratios and complete failed/excluded attempts are in the current acceptance artifact.

MTP4 service constraints, lifecycle and long strict-JSON prefix reuse pass. Initial
Chat/Responses oh-my-pi runs have verified real tool calls, follow-up requests and
accepted drafts, but their final answers confuse historical/current counts; the
Responses answer also incorrectly says prefill sorts last. Actual batch planning
sorts prefill first. Updated-document readback remains required before final closeout.

TileLang remains the default pending that readback and the final default-selection
change. Runtime/kernel implementation is frozen during formal collection. See
`acceptance.md` for current evidence and `cuda-development.md` for CUDA/PTX choices,
TileFoundry estimates versus measured Nsight observations, and historical tuning.

## Remaining work

- Verify updated-document Chat/Responses readback with MTP4 and actual tool results.
- Switch the default to CUDA and make runtime identity report the same selected
  backend. Preserve explicit `OH_MY_VLLM_KERNEL_BACKEND=tilelang` comparison.
- Run final relevant checks, independent review, commit and push on main.
- Stop every task-owned GPU program and verify its GPU/port resources are released.

All collection processes have exited; nvidia-smi has no compute processes and
port18030 is released. No service was requested to remain running. All-file
Rust fmt/width/clippy/tests and Ruff checks pass.

## Operating rules

Use `scripts/with-env.sh` and idle-UUID selection via `scripts/with-gpu.sh`.
Stay on main, stage only intentional files, keep hooks enabled, maintain Chinese
Markdown companions and retain the required commit attribution. Do not leave any
GPU service or child worker running after its task finishes. Temporary operator
experiments and raw GPU traces stay outside the repository. Formal summarized
operator/framework acceptance evidence belongs in `bench/baseline`.

Historical work is retained in Git and the dated artifacts. The independent
runtime and TileLang migration were separately accepted; those results do not
replace current CUDA acceptance.
