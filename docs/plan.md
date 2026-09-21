# Implementation plan

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
