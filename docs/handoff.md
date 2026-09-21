# Handoff — 2026-09-21

## Four-proposal CUDA Graph — final MTP rerun pending

The ca5a200 matrix passed all six ordinary/prefix rows and MTP bs1. MTP bs2
measured94.31% of the frozen baseline, so acceptance remains incomplete. Its
three repetitions agree closely. The obsolete MTP bs4 attempt was stopped after
two external-GPU-contention discards because another MTP optimization was ready.

The proposer now captures initial greedy selection and three draft/head steps in
one graph, transferring all four tokens together. All three cache writes must fit
Rust's allocated pages; near-boundary batches retain the old stepwise trimming.
Capture/warmup restores overwritten FA rows, and row indices remain graph-owned.
An isolated prototype exposed a dangling row-index buffer; a new allocator-pressure
and dynamic-table regression covers the fix. The corrected diagnostic measured
409.517tok/s for bs2, with the same steps and accepted draft count as the stepwise
candidate. The target/sampler/ordinary/prefix paths are unchanged by this patch.
Retain the six valid ca5a200 ordinary/prefix rows and rerun all three MTP rows on
the new clean commit, explicitly recording each implementation identity. GPU pytest
passes92 tests and12 subtests, with no skips. Rust75 tests, fmt/width/clippy, ruff
and all pre-commit checks pass.

## Batched sampling and metadata staging — final matrix pending

Seven f9c283d rows passed: ordinary bs1/2/4, MTP bs1/4 and prefix bs1/2.
MTP bs2 and prefix bs4 were excluded because diagnostic processes accidentally
shared their pinned CPU cores during measurement. Those rows are not accepted
as formal results; raw attempts and exclusion reasons are retained in the
batched-sampling artifact. Diagnostic jobs now use separate CPU cores20/21.

Plain greedy requests now reduce all target rows and transfer their selections
once per batch, preserving first-rejection and invalid-reachable-row semantics.
Masks, penalties and stochastic sampling retain the general sampler. MTP draft
warmup/verification also shares KV reads, with request count in its graph key.
Private host metadata uses nonblocking copies on the consuming CUDA stream;
pageable staging can still synchronize internally. No claim of guaranteed DMA
overlap is made. Full-row graph logits avoid redundant indexing copies.
Candidate diagnostics measured399.111tok/s for MTP bs2 and338.548tok/s for
prefix bs4 (one warmup/measurement, not final acceptance). GPU pytest passes91
tests and12 subtests, with no skips; Rust75 tests, fmt/width/clippy, ruff and
all pre-commit checks pass. The next action is a clean-commit nine-row matrix.

## Grouped verification optimization — final acceptance pending

Clean commit9c8e27b's MTP bs1 measured247.345tok/s (91.895% of original
baseline), failing95%; prefix bs1 passed110.240%. Other attempts were discarded
when the monitor detected external GPU clients. The baseline was not rerun.
Target verification now shares KV reads across up to5 queries from one request,
with separate per-row causal masks and dynamic ragged starts in CUDA Graphs.
An uncommitted diagnostic measured263.494tok/s (one warmup/measurement); this
is not final acceptance. Raw preoptimization rows and the diagnostic are in
bench/baseline/2026-09-21-independent-grouped-decode.json.

Default independent actual-model ordinary/MTP FP64 probes passed, as did
ordinary and MTP prefix/preemption text regressions,12 constrained MTP service
cases, lifecycle/cancellation and both real oh-my-pi API workflows. Both agent
runs made9 successful read tool calls; the Responses answer misstated the earlier
ordinary diagnostic as below threshold. It was correctly marked non-final, but
that factual slip must not be represented as perfect answer accuracy. Rerun the
agent tasks against final documentation. The service and port18013 were released.

## Current execution path — independent Worker only

The legacy GPUWorker/V2 adapter and adapter-specific tests are removed. The owned
Worker is now worker/model_runner.py; probe_worker.py instruments this implementation.
Rust and scripts/with-env.sh default to conda oh-my-vllm with independent kernel
caches. No opt-in flag or old implementation remains. Benchmark code can only run
the framework and read the original frozen baseline, checked by fixed SHA256.
The parsed baseline and recorded hash use the same immutable byte snapshot.

All54 CPU unittests ran in the target environment (52 pass,2 CUDA tests skipped
there). Rust75 tests, release build, fmt/width/clippy, ruff and pip check pass.
A default-path real MTP arithmetic run generated25 and passed module/library
shutdown audit. Its strace file-access audit found zero accesses to either the
vLLM source checkout or old conda environment; traces stay outside the repository.
Full GPU pytest passes:80 tests and12 subtests, with no skips. All task-owned
GPU processes exited. Next: commit the independently reviewed removal, then
run the monitored clean-commit nine-row matrix and both real oh-my-pi tasks. Do not
edit Python code during measurements. No final migration acceptance claim yet.

## Current migration — independent GPU runtime in progress

The user approved ADR-006 after the design interview. Rust/Python ownership stays
unchanged. Final builds/tests/inference must not depend on vLLM, its source, old
environment or compiled cache artifacts. Intermediate stages may retain vLLM.
Only the original frozen EngineCore baseline is an acceptance reference.

### Stage 1 (completed before Worker integration)

First milestone: establish stable dependencies in conda oh-my-vllm and validate
independent FP8, GQA and GDN/MTP state operators against CPU references. The new
operator modules are not yet the production execution path. The 80-package stable
candidate closure is installed in conda oh-my-vllm, including the editable project;
pip check passes and vLLM is not importable. All 11 GPU operator tests pass in that
environment with independent FlashInfer/Triton caches; 17 service/sampling CPU tests
also pass there. The complete transitional CPU suite has 48 passing tests in the
old environment. Rust workspace tests, fmt/width and clippy pass.

Numerical tests exposed two FlashInfer boundary mismatches: prefill's advertised
q/k normalization flag is unused, requiring explicit normalization; CUTLASS's
K-major activation scale requires contiguous K groups despite its column-major
parameter description. Both are fixed and covered by numerical tests. Quantizer
FP32 rounding is independently checked before FP64 GEMM reference comparison.
The sampler review caught NaN/+inf greedy rows; reachable bad rows now fail,
while unreachable rows after draft rejection remain harmless. Static re-review
passes. The independent model probe generated 32 coherent Chinese tokens with
finite hidden states. Twelve Chat/Responses × off/medium × JSON/object/tool real
MTP4 service cases pass through the transitional V2 adapter; this validates the
service utility migration, not independent MTP. All task-owned GPU processes
exited and port18012 is released. Evidence is recorded in
bench/baseline/2026-09-21-independent-stage1.json. Production MTP, graph integration,
actual-model FP64 and final acceptance remain pending.
Existing V2 service still defaults to the old environment until replacement. Do
not claim full runtime independence or performance acceptance yet.

## Independent Worker integration milestone

The opt-in independent Worker now executes Rust schedules, owns concrete model
computation and preserves Rust's logical allocations. Host planning checks accepted
history, physical state aliases and 784-token checkpoint copies. Shifted MTP cache
rows preserve prefix immutability, restore boundary hidden features and shorten
proposals when Rust has not yet allocated the next page. Target/draft graphs restore
all touched state after warmup/capture; dedicated CUDA tests cover in-place state
and recurrent reuse of a graph's output.

Ordinary/MTP text, MTP prefix+five-preemption batches, twelve real constrained API
cases and full service lifecycle pass. A real arithmetic failure exposed
nondeterministic FlashInfer CUTLASS SM100 FP8 results for 17..32 rows; the final
small-matrix path uses independently validated TRT-LLM kernels and their distinct
activation-scale layout. Seventeen operator, three paged-decode, three pointwise,
and two graph GPU tests pass. Final ordinary/MTP actual-input FP64 probes pass
unchanged tolerances. Rust75 tests, fmt/width/clippy and ruff pass. Transitional CPU
suite:58 pass,2 GPU tests skipped there (run separately in the target environment).
The full CPU suite still needs the old environment only for legacy adapter tests.

A one-warmup/one-measurement diagnostic ordinary32K->4096 run reached94.499tok/s.
This is not the monitored clean-commit acceptance matrix. Final nine-row performance,
both independent oh-my-pi tasks, removal of the old default/environment/legacy tests
and final documentation reconciliation remain pending. Evidence is in
bench/baseline/2026-09-21-independent-stage2.json. All task-owned GPU processes exited
and port18012 is released. Continue implementation; do not treat this milestone as
completion of ADR-006.

## Previous migration — V2 complete and verified

Implementation `d85c23e` requires V2 exclusively; no old-runner fallback or vLLM
source edits remain. Rust retains scheduling/KV and accepted history. Resumption
re-admits requests with full accepted history; ordinary decode stays incremental.
Local draft transfer supplies real MTP IDs. Startup cache clearing and fresh-page
zeroing prevent stale hybrid state without clearing shared prefixes/live MTP slots.

All nine ordinary/MTP/prefix × bs1/2/4, 32768→4096 rows pass at 97.22%–106.62%
of the frozen 2026-09-19 EngineCore baseline. No baseline was rerun. Each row has
two warmups and three measurements from a clean implementation commit. 6 attempts
with external GPU contention were discarded; unrelated processes were untouched.
Prefix variance follow-ups retain five measurements each without replacing the
original matrix: bs2 reaches 100.02% with 0.09% spread; bs4 reaches 99.94% with
0.07% spread.
Actual vLLM source HEAD is 039b2ad67da6d64f7c1835c4738c7fb545ad37fc, while package
metadata remains g6376c601e. This is historical, not same-source paired evidence.
See [acceptance](acceptance.md) and
[recorded results](../bench/baseline/2026-09-21-v2-acceptance.json).

Actual FP64 ordinary/MTP probes pass unchanged tolerances with zero nonfinite
values. Coherent multi-request text survives ordinary/MTP preemptions, prefix hits
and staggered arrivals. Twelve real constraint cases preserve MTP. Two lifecycle
rounds prove same-batch mixing and Worker release after disconnect. Both real OMP
APIs read implementation files and complete tool follow-ups with no tool errors.
The acceptance prompt now explicitly requires those source reads. Evidence keeps
preliminary runs, the Chat citation typo and the earlier failures/fixes separately.

Checks pass: 75 Rust tests, 39 Python unittest tests, rustfmt/hard width, clippy,
ruff and pre-commit. Independent implementation and final evidence reviews pass.
All task-owned probes, benchmark workers and HTTP servers stopped; GPU process
inventory and port18002 checked after the final run. No running-service exception.
No migration acceptance work remains.

## Latest maintenance — enforce Rust formatting inside macros

The reported error-response line in serving/mod.rs was 234 characters, yet
cargo fmt --check passed because rustfmt does not reliably reflow JSON macro
bodies and max_width is not a hard validation rule. Added stable rustfmt.toml
and a separate hard 100-character check covering all tracked/unignored new Rust
files, including macros, comments and strings. Pre-commit now rejects formatting
drift with cargo fmt --check; it also runs the width check when configuration or
hook definitions change. Both checks read the repository's formatting policy.

Expanded JSON macros and logging expressions, split long literals with concat!,
and applied the stricter multiline-if/let-else settings. This is a formatting
change; protocol data and test strings remain unchanged. Added five checker
regressions. The actual new pre-commit hook rejected a temporary copy of the
original 234-character line, then passed on the reformatted repository.
Review identified a Unicode line-separator bypass in splitlines(); physical-line
splitting and a regression now prevent it. Checks passed: 73 Rust tests, 16
serving Python/HTTP tests, five style regressions, clippy, rustfmt and ruff.
No GPU process was started for this maintenance task.

## Latest cleanup — release task-owned GPU programs promptly

The user identified that the acceptance service had been left running without
an explicit request to retain it. Sent SIGINT to the verified task-owned Rust
server (PID 821668); it shut down its Python worker (PID 821857). Both processes
are gone, port 8000 is no longer listening, and nvidia-smi no longer lists the
worker's 87,516 MiB allocation. No unrelated process was stopped.

AGENTS.md and CONTRIBUTING.md now require prompt cleanup after tests/runs,
including completion, failure, cancellation and handoff. Keeping a GPU service
running requires an explicit user request and a documented handoff exception.
The serving implementation and recorded acceptance results are unchanged.

## Latest workflow decision — develop only on main

The user requires all development and commits on `main`, with no new local or
remote branches. AGENTS.md and CONTRIBUTING.md now record this policy. The serving
commits through a7f47fd were fast-forwarded onto local main without rewriting or
discarding work and published to origin/main. The codex/openai-compatible-serving
branch was then deleted locally and on GitHub; only main remains in both places.
Independent review passed; 73 Rust tests, all-target/all-feature clippy and Python
format/check passed. The policy documentation is committed directly on main.
That workflow-only change did not alter the service or its acceptance.

## Latest implementation — real MTP4 serving acceptance complete

The host CUDA/NVML fault recovered: seven visible B200s enumerate normally, and
an idle UUID-pinned card passed cuInit, allocation and matrix multiplication.
The real serving run used GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad, MTP4,
Qwen3.5-27B-FP8 and block784. Rust still owns HTTP, scheduling and KV.

Both real oh-my-pi tasks completed with successful README/architecture/source
reads, tool-result follow-ups and grounded final answers. Real tests passed all
12 Chat/Responses × off/medium × JSON object/schema/strict-tool combinations,
all five thinking levels, stored Responses continuation/retrieve/delete, mixed
constraint/plain batching, and disconnect recovery. Every constraint case
proposed and accepted actual MTP drafts. See acceptance.md and
bench/baseline/2026-09-21-serving-acceptance.json for commands and evidence.

Real runs exposed two defects missed by the scripted CPU worker. Qwen wraps
ordinary XML parameters in newlines; the parser now removes one wrapper on each
side, matching the installed vLLM parser, while preserving schema enum/const
values exactly. Tokenizer length recomputed the full vocabulary on every output
token (~29 ms); vLLM's cached tokenizer removes that cost. Added DEBUG-only
grammar/model-sampling/output phase timings to identify such host overhead.

Warm short constraint cases reported 202.8–280.1 output tok/s, versus ~30 before
the tokenizer fix. Final agentic answers reported 208.1/219.3 tok/s; these are
serving observations, not a matched EngineCore performance comparison. One Chat
thinking-loop retry recovered and is preserved in evidence. The initial answers
faithfully cite then-current docs that still described the recovered GPU outage;
status documentation was reconciled afterward.
Both APIs were subsequently rerun against the updated docs, with zero tool errors
or retries and unchanged repository diff; doc_refresh_agentic stores those answers.
Named-tool and no-tool choices also passed on both real APIs.

CPU regression coverage remains 73 Rust tests, 16 serving Python/HTTP cases and
12 existing Python tests. Independent review approved the newline and tokenizer
fixes. No kernel/scheduler/KV change was made in this follow-up; the historical
EngineCore matrix and FP64 probes were not rerun because this change is confined
to HTTP output handling and diagnostic timing. The agreed serving task has no
remaining GPU blocker or required implementation work.

The acceptance service previously ran at http://127.0.0.1:8000/v1 (Rust PID
821668, worker PID 821857) and has now been stopped as recorded above. Its log
remains at /tmp/oh-my-vllm-serving-live4.log and used DEBUG worker phase logging.
Normal startup defaults to INFO as documented in serving.md.

## Earlier implementation milestone — GPU acceptance was blocked

The user subsequently authorized implementation and confirmed both APIs, default
medium thinking (high maps to xhigh), JSON/strict tools with MTP, and observational
performance checks without a new serving throughput gate. Rust now owns Axum HTTP,
request lifecycle, protocol events and bounded Responses storage. Python handles
local templates, incremental detokenization and XGrammar masks across speculative
prefixes; Rust scheduler/KV ownership and block784 are unchanged.

Implemented Chat/Responses streaming and nonstreaming, models, tool history,
response retrieve/delete/continuation, cancellation and deadlines. See serving.md
for the supported subset, explicit schema restrictions and commands. Added an
isolated real oh-my-pi runner for the agreed read-only repository introduction.

CPU evidence: Rust unit tests and actual Rust HTTP/ZMQ integration tests pass;
actual local tokenizer/XGrammar tests cover speculative rollback, invalid drafts,
EOS/bonus rows and thinking boundaries. Installed oh-my-pi 18.2.6 completed tool
read/follow-up exchanges via both APIs at medium and off against the scripted CPU
worker. These verify client wiring only, not model quality or GPU MTP acceptance.
Client logs are in /tmp/oh-my-vllm-omp-wire3{,-responses}/ and
/tmp/oh-my-vllm-omp-off2-{chat,responses}/ on this host.

Final checks: 73 Rust tests, all-target/all-feature clippy, rustfmt, Python
format/check, 16 serving CPU/HTTP tests and 12 existing adapter/bridge/runtime/
benchmark-tool tests passed. Independent reviews fixed stream cancellation,
history inheritance, schema ambiguity and XML boundary issues; subsequent review
passed. The EOS-draft regression verifies rollback including a terminal token.
No actual GPU FP64, MTP correctness or performance regression suite could run on
this host during this task; CPU tests explicitly select VLLM_TARGET_DEVICE=cpu.

Real GPU attempt: an MTP4 service failed before model initialization with NVML
Unknown Error (/tmp/oh-my-vllm-serving-mtp.log). Minimal torch CUDA allocation also
failed on two idle B200 UUIDs selected with scripts/with-gpu.sh:
GPU-1b174534-ebba-826f-b452-8e7f3c05c301 and
GPU-b0da5a7e-8dcf-e7dd-9836-a29a45d01b45. NVML cannot obtain the device handle for
0000:4B:00.0. No unrelated GPU process was stopped or device reset attempted.
The final release build passed; another MTP4 startup failed in the same NVML
initialization path (/tmp/oh-my-vllm-serving-final-mtp.log). Owned test servers
and clients were stopped after verification.

Remaining: restore host CUDA/NVML health, start the real MTP4 service, run both
agentic-acceptance.py API tasks, inspect actual tool results/final answers, test
JSON/strict tools with real speculative acceptance and review latency/throughput/
proposal/acceptance logs. No real serving performance number or GPU acceptance
is claimed. Earlier EngineCore performance results below are historical evidence,
not measurements of this serving change.

## Previous task — serving discussion documented (2026-09-21)

The user requested documentation only and explicitly stopped implementation.
Accepted future scope: both OpenAI Chat Completions and Responses, configurable
thinking, Rust-first outer entry points, full-history and stored Responses with
bounded/expiring memory and no restart persistence. Both service and oh-my-pi run
on this machine. Each API must independently pass a real oh-my-pi task that reads
this repository and returns a file-grounded project introduction without edits.

See serving.md, REQ-SERVE-001..003 and ADR-004. Requirements, architecture, plan,
index and contribution scope were reconciled. Existing EngineCore acceptance
below applies to the previous implementation, not the future HTTP service.
No runtime or client configuration was changed, and no service/GPU/agentic test
was launched. Native thinking levels are off/low/medium/xhigh; external mappings,
defaults and the other open choices are explicitly listed in serving.md.

Verification: independent documentation review passed with no blocking findings;
64 Rust tests, all-target/all-feature clippy, Python format/check and git diff
whitespace checks passed. Python formatting left all six files unchanged. GPU,
HTTP and agentic acceptance tests were not run: this task changes documentation
only and serving is not implemented. Serving implementation remains future work,
not a blocker for this documentation-only task.

## Previous engine implementation handoff (2026-09-19)

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
