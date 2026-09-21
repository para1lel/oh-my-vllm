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

## Serving extension — discussion documented, implementation not started

The 2026-09-21 discussion accepted both OpenAI APIs, configurable thinking,
in-memory Responses history and Rust-first entry points. The user requested only
documentation for this task. See [serving.md](serving.md) for open decisions.

Future implementation sequence (proposal): specify remaining compatibility and
thinking mappings; add a persistent engine lifecycle and correct termination;
implement both HTTP protocols and response storage; validate both using real
oh-my-pi repository-reading tasks. No service or agentic acceptance run exists yet.
