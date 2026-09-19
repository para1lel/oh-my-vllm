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

## In progress

- Complete real-text prefix-hit validation and keep all current documentation
  aligned with AGENTS.md and production code.
- Measure and repair remaining performance gaps. Target batch1/2/4,
  input32768/output4096, ordinary/MTP/prefix, at least95% of matched vLLM.
- Investigate RPC wakeup overhead and CPU affinity with matching baseline settings.
- Rerun cases with material variance or external GPU interference; retain actual
  configuration/source/binary identity with acceptance artifacts.

Whole-project performance acceptance has not passed. Partial passing rows do not
complete this phase. Each further milestone needs review, fixes, checks and commit.

## Deferred scope

CPU KV swap, multiple GPUs, HTTP/gRPC serving, LoRA, multimodal execution and
production deployment remain outside the current task. CI is not implemented;
local pre-commit checks are required. See requirements.md for scope authority.
