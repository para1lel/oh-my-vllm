# Handoff — 2026-09-19

## Scope and authoritative decisions

AGENTS.md governs the model, block784, Rust scheduler/KV ownership and conda
entrypoints. Single-GPU tests wait for any idle B200 and pin its UUID; unrelated
GPU processes must never be interrupted. The user authorized creating and syncing
the private GitHub repository below; no PR was requested. Every milestone
requires independent subagent review and a commit with the required attribution.
User authorized committing all project changes, including the original .gitignore
modification. Local .vscode/settings.json selects conda oh-my-vllm for Python
analysis and adds python/ to resolution; .vscode/ is ignored, as requested.

## Current status: acceptance complete

All nine target performance rows passed on the clean 12d227d binary: ordinary
batch1/2/4 = 101.66%/101.29%/101.46%; MTP = 102.64%/103.78%/100.77%; prefix =
101.68%/101.55%/97.13%. Each has two warmups and three measurements, matched GPU
and single-core CPU affinity, and monitored GPU ownership. Exact protocol,
commands, identities and all raw repetitions are committed under bench/baseline/
2026-09-19-acceptance.json. See acceptance.md for table and limitations.

MTP batch2 showed material spread, so an additional three-warmup/five-measurement
pair was run and retained separately in 2026-09-19-mtp-bs2-repeat.json. It passed
at98.85%, with within-engine spread0.67%/0.13% (vLLM/framework). This confirms the
95% gate; do not claim the initial3.78% advantage as a stable speedup. The repeat
used the same binary; unrelated dirty edits are recorded, and Python runtime
sources were unchanged until all measurements completed.

Final code review found cancellation retained Python/native Worker state and
that aggregate cache-hit counters could falsely establish seeded prefix reuse
in a preemption smoke. Both were fixed: abort queues/forwards finished-only
notifications; initial_prefix_hit_tokens counts first actual admission only.
Run supports ordered prompt-file batches and staggered arrivals, with input
validation before GPU startup. A new smoke checks two distinct Chinese topics
through actual recompute preemption and inspects both beginning and tail.

The repeated ordinary text test observed2 preemptions, initial hits0; MTP plus
prefix observed4 preemptions, initial hits3136, accepted drafts1143. Both produced
1024 tokens per request and coherent Beijing/Tokyo text without switching topics.
Full generated text, counters and dirty binary identity are saved in
2026-09-19-batch-text.json. These text/cancellation/CLI changes follow the measured
12d227d executable; the full nine-row matrix was not rerun on the later binary.
Independent review found no material performance risk from those changes.

Verification:64 Rust tests passed, all-target/all-feature clippy passed, Python
format/check passed; runtime tools2, benchmark tools4, adapter4 and bridge2 CPU
tests passed. The new legacy-abort regression failed before the fix and passes
now. Independent code and performance/data reviews passed after resolving their
findings. The earlier actual-path GQA/GDN FP64 and ordinary/MTP/prefix text checks
remain applicable; no inference kernels or numerical path changed in this stage.

No remaining blocker within the agreed scope. The local VSCode environment is
verified and ignored; .gitignore was included in ca65aba.
Logs/traces stay outside git; reproducible numeric and text evidence is retained.
The sections below are chronological history, including superseded blockers and
incomplete-performance status. Use this section and acceptance.md for current status.

## GitHub repository synchronization

At the user's request, created the private repository
https://github.com/para1lel/oh-my-vllm using the existing gh CLI credentials.
The authenticated API identity is para1lel (the local gh account label is zynier).
Set origin to https://github.com/para1lel/oh-my-vllm.git and pushed main with its
complete commit history through fefba8a; main now tracks origin/main. The local
repository has no other branches or tags. GitHub reports visibility PRIVATE.
Ignored local settings, build artifacts and GPU traces were not added to git.

This synchronization changes no runtime code. Cargo workspace tests (64),
all-feature clippy and Python format/check were rerun and passed. No GPU tests
were rerun for this repository/documentation operation. This handoff update is
included in the final synchronization commit.

## Historical milestones

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

## GPU availability interruption (12:09 UTC)

Feature milestone committed as b2e3a11 after final independent review; all hooks
passed. Updated prefix paired smoke failed during framework initialization:
an unrelated job occupied the selected GPU between the two engines, leaving
156.1GiB free versus the worker's requested164.09GiB. This is not a cache
correctness failure or a valid throughput measurement. All eight B200s had
unrelated compute processes at that observation. No unrelated process was terminated.

The waiting scripts acquired idle cards again around12:10UTC:
- Prefix long batch1/2/4 pair: /tmp/oh-my-vllm-prefix-long.log and eventual .json.
- MTP long batch1/2/4 pair: /tmp/oh-my-vllm-mtp-long.log and eventual .json.
- Ordinary execution cProfile/DEBUG diagnostic: /tmp/oh-my-vllm-profile-b.log,
  eventual /tmp/oh-my-vllm-execute-profile.pstats (never commit profiling data).

The first temporary profiling wrapper rejected the interpreter's -m arguments
before connecting. Its owned Rust driver was terminated and the wrapper fixed;
no valid profile came from /tmp/oh-my-vllm-profile.log. Init retry currently waits
until its timeout if a child dies before connecting; improve early child-exit
reporting when revisiting client lifecycle. Pending GPU commands may finish after
this note; inspect their logs before restarting any run.

## IPC experiment and monitored measurements

The cProfile/DEBUG run completed: Rust scheduling median1us, RPC median11606us,
Python execution median10930us. The current-thread Tokio experiment completed
actual32768->256 inference; corresponding medians were11513us and10937us. These
are diagnostics on separate GPUs, not acceptance throughput evidence. A CPU echo
experiment with10ms simulated execution reduced128-step elapsed from1.355s to
1.305s when controller and responder shared a CPU. A real paired ordinary bs1
experiment now uses the same inherited CPU8 affinity for both engines.

Benchmark changes now reject numeric GPU IDs, detect external same-GPU clients
by process group during execution, support --binary, and execute a hash-checked
private binary copy. Independent review passed after closing GPU-ID bypass and
binary replacement windows. Four CPU benchmark tests,62 Rust tests, clippy and
ruff passed; current-thread real diagnostic output completed correctly.

The unmonitored MTP run encountered an external GPU user after startup; its
bs1 result is invalid and its next owned engine was stopped. Do not use
/tmp/oh-my-vllm-mtp-long.json for acceptance. Replacement monitoring-enabled
/tmp/oh-my-vllm-mtp-monitored.json has bs1 ratio94.18% (old multi-thread binary),
so it fails the95% gate. bs2 completed at96.2063%; bs4 is pending. Current-thread
ordinary bs1/2 pairing is /tmp/oh-my-vllm-current-thread-pair.{log,json};
bs1 median ratio95.0964% is a preliminary pass with a slow first repetition,
so repeat before final acceptance.
CPU8 ordinary bs1 pairing is /tmp/oh-my-vllm-pinned-ordinary-bs1.{log,json}.
Both variants live in /tmp/oh-my-vllm-current-thread-target/release/ to avoid
replacing target/release while another measurement uses it. Record and compare
actual binary hashes, not just source HEAD. Overall acceptance remains incomplete.


## Documentation and real prefix-text completion

The current README, design/protocol, development/profiling commands, plan/index,
requirements and contributing guide were audited against code and AGENTS.md.
Removed obsolete next_token_id/always-empty drafts, old Worker API, single Mamba
slot, standalone-only accuracy claims, outdated validation status and unwrapped
GPU/environment commands. Historical HTTP-serving baseline captures are explicitly
separated from matched EngineCore investigation data in bench/baseline/README.md.
Completed investigation pairs are preserved in the adjacent2026-09-19 JSON; they
are not the full acceptance matrix and missing early CPU masks are null.

Run --prefix-hit seeds the same prompt, then requires a real cache hit before
returning text. smoke-text supports this and an isolated --binary. Both ordinary
and MTP prefix text runs passed with64 output tokens and a coherent initial Chinese
answer (/tmp/oh-my-vllm-prefix-text.log and -mtp-prefix-text.log). Fixed-length
smoke intentionally ignores EOS, so output after the first answer may continue
with subsequent chat-role tokens; do not interpret this as an EOS-stopping API.
The client now detects a worker that exits before connecting; its new CPU test
passes in0.10s instead of waiting the initialization timeout.63 Rust tests,
all-feature clippy, Python format/check and both actual prefix text checks passed.
Independent code/document review found no blocking code issues; listed document
cleanup suggestions were addressed.

All three CPU-affinity candidate pairs were interrupted by external GPU clients
and produced no valid comparison result. The new monitor correctly rejected them.
The current-thread ordinary bs2 and prefix later batches were also interrupted.
The user was asked asynchronously about a stable single-card measurement window;
no reply yet. At approximately12:39UTC the short tests obtained cards and completed.
Continue to wait for any idle B200 and never interrupt other users' processes.
