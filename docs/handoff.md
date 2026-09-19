# Handoff — 2026-09-19

## Scope and authoritative decisions

AGENTS.md governs the model, block784, Rust scheduler/KV ownership and conda
entrypoints. Single-GPU tests wait for any idle B200 and pin its UUID; unrelated
GPU processes must never be interrupted. No push/PR authorized. Every milestone
requires independent subagent review and a commit with the required attribution.
User authorized committing all project changes, including the original .gitignore
modification. Local .vscode/settings.json selects conda oh-my-vllm for Python
analysis and adds python/ to resolution; .vscode/ is ignored, as requested.

## Saved milestones

- 4444ab1: documentation/environment alignment, idle GPU selection, correlated
  timestamped Rust/Python logging. Independent review and required checks passed.
- ca65aba: installed GPUWorker API adaptation, actual output lifecycle, physical
  hybrid cache alias fix (ADR-002), actual-path FP64 probe and coherent Chinese.
  Independent review and required checks passed (58 Rust tests at that stage).

## Current milestone: MTP and feature/benchmark lifecycle

MTP now schedules and consumes real drafts, retains all accepted tokens, rolls
back only scheduled rejections, and reserves/migrates private recurrent states.
BF16 SSM storage preserves block784 in MTP mode and is matched in vLLM (ADR-003).
Bench keeps one worker, resets caches, supports controlled prefix seeding,
request arrivals and reduced scheduler capacity for recompute preemption.
Matched driver uses identical token IDs/sampling/warmup/cache policy, three
measurements, exact cached-token checks and draft counters on both engines.
It records initial source/binary identity and owns engine process-group cleanup.

Review first found missing timeout descendant cleanup, silent zero-draft mode,
incomplete cache-hit equivalence, misleading dirty-version attribution and
missing MTP allocator regressions. Fixes were independently re-reviewed with no
blocking findings. Additional multi-block speculative slot migration test added.
62 Rust tests and all-feature clippy passed; ruff Python checks, runtime tools and
benchmark process-group CPU tests passed. Adapter (3) and bridge logging (1) checks also passed.

Actual GPU evidence (logs outside repository):

- /tmp/oh-my-vllm-text-d.log: ordinary coherent 64-token Chinese output.
- /tmp/oh-my-vllm-probe-d.log and -probe-boundary.log: actual GQA/GDN
  prefill/decode/state FP64 checks, including cross784 continuation, passed.
- /tmp/oh-my-vllm-mtp-text-a.log: MTP4 coherent64tokens in27steps.
- /tmp/oh-my-vllm-probe-mtp-a.log and -probe-mtp-boundary.log: actual target
  fused MTP verification/all draft states, GQA and GDN prefill passed FP64 bounds;
  cross-boundary32-token Chinese output is coherent, all8 categories covered.
- /tmp/oh-my-vllm-prefix-a.log: prefix + arrivals, output256, hits3136.
- /tmp/oh-my-vllm-preempt-a.log: ordinary reduced pool, output2048,
  3 recompute preemptions, hits3920 on resumed requests.
- /tmp/oh-my-vllm-combined.log: MTP4 + prefix + arrivals, output256,
  hits3136,43steps,300proposed/181accepted drafts.

- /tmp/oh-my-vllm-mtp-preempt-b.log: MTP4 reduced logical pool18, output2048,
  3preemptions, hits5488,2024proposed/1545accepted drafts.
- /tmp/oh-my-vllm-mtp-pair-short.json: updated driver short MTP paired run
  passed with real nonzero draft metrics on both engines and zero cold cache hits;
  this 2048->64 case is not a target workload acceptance result.

## Performance status and remaining work

Full acceptance has NOT passed. Preliminary ordinary paired32768->4096 medians
relative to vLLM: bs1=94.10%, bs2=94.77%, bs4=95.87%. Raw logs/results are
/tmp/oh-my-vllm-ordinary-bs{1,2,4}.{log,json}. These were measured before the
identity/metrics review fixes on a dirty working tree; do not label their
numbers as clean ca65aba results or final acceptance artifacts. The initial
bs4 standalone313.3tok/s had no matched warmup and is not acceptance evidence.

Remaining: validate updated baseline prefix-hit fields end-to-end;
profile and repair ordinary bs1/2
shortfall; run all ordinary/MTP/prefix batch1/2/4 matched acceptance cases with
reproducible artifacts and investigate material variance. GPU kernel timing is
not covered by host-duration logs. Update this handoff with measured outcomes.
