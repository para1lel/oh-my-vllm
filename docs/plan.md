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

## Serving extension — implemented, GPU acceptance blocked

The confirmed design is implemented in Rust HTTP/protocol/state modules and the
Python model/grammar adapter. CPU and OMP protocol tests use the real Rust server,
tokenizer and XGrammar with scripted model output. They do not validate GPU MTP.

Remaining: restore functional CUDA/NVML on this host; launch an idle UUID-pinned
B200 with MTP4; run both actual OMP repository-reading tasks, strict JSON/tool
cases, ordinary serving EOS/cancellation and log-based performance checks. Repeat
actual-path state/attention probes if GPU investigation changes those paths.
Record evidence and resolve runtime findings before declaring serving accepted.
See serving.md, testing.md and the latest handoff entry.
