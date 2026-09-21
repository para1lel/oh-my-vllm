# Implementation plan

## Active migration — independent runtime (ADR-006)

1. Resolve recent compatible stable dependencies in conda `oh-my-vllm`; validate
   independent FP8, GQA, GDN and convolution/state operators against CPU FP64.
   Exercise ragged MTP snapshots, prefix isolation and the 784-token boundary.
2. Validate the concrete Qwen model and greedy draft/target sampling, then replace
   the Worker adapter while retaining Rust scheduling, cache ownership and ZMQ.
3. Integrate MTP4, hybrid cache checkpoints, preemption/resumption, structured
   decoding and service lifecycle. Add CUDA graph execution and measure hot paths.
4. Remove all runtime/build/test dependencies on vLLM, its checkout, old conda
   environment and artifacts. Verify clean independent caches and dependencies.
5. Run existing correctness, both real oh-my-pi tasks with MTP, and the nine-row
   performance matrix against only the original frozen EngineCore baseline.
   Reconcile documents, obtain independent review, commit and push `main`.

The environment/operator and independent Worker integration are validated.
MTP4, prefix/preemption, graphs and real constraint/lifecycle cases pass.
Runtime-default removal, final oh-my-pi tasks and performance acceptance continue.
Intermediate use of the old adapter is
explicitly allowed; it is not evidence of final independence. Future model,
NVIDIA backend, single-node multi-GPU and local DSpark extension points are
documented without placeholder interfaces or additional acceptance scope.

## Completed and saved

1. Environment/documentation foundation, idle GPU selection and timestamped logs
   (4444ab1).
2. Actual installed GPUWorker adapter, correct physical hybrid mapping, coherent
   Chinese and actual-path GQA/GDN FP64 checks (ca65aba, ADR002).
3. Actual MTP drafts/state lifecycle, prefix/arrival/preemption combinations,
   MTP cross-block FP64 checks and matched benchmark lifecycle (b2e3a11, ADR003).
4. Controller thread experiment and benchmark interference/identity safeguards
   (914de33).

These milestones passed independent review and required checks. Verification
counts and individual run outcomes are in handoff.md, rather than duplicated here.

## Acceptance

Real prefix-hit text and documentation reconciliation were saved in 12d227d.
The full ordinary/MTP/prefix performance matrix passed at batch sizes 1/2/4;
[acceptance.md](acceptance.md) links raw evidence and configuration. Additional
multi-request text tests exercise staggered arrivals, recompute preemption and
MTP/prefix together, with separate initial-admission cache-hit accounting.
Cancellation now propagates finished notifications through the Worker lifecycle.

Final variance follow-up, independent review and required checks are recorded in
handoff.md. Future changes need the same review/check/commit discipline.

## Deferred scope

CPU KV swap, multiple GPUs, gRPC serving, LoRA, multimodal execution and
production deployment remain outside the current task. CI is not implemented;
local pre-commit checks are required. See requirements.md for scope authority.

## Serving extension — implemented and accepted with real MTP4

Both real oh-my-pi API tasks passed after the host CUDA/NVML outage recovered.
Real JSON/strict-tool tests, thinking levels and lifecycle checks passed. The GPU
runs exposed Qwen XML wrapping-newline handling and repeated tokenizer vocabulary
lookup overhead; both were fixed and retested. See acceptance.md for current
serving evidence, which remains separate from the older EngineCore matrix.

The supported subset and deliberate exclusions remain in serving.md. No further
work is required for the agreed serving task; normal future changes retain the
review/check discipline and model/hardware/scheduler invariants.
