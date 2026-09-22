# Acceptance evidence

## TTFT and 262144 context — in progress, 2026-09-22

All six ordinary/MTP4 batch 1/2/4 maximum-context checks complete 258048 input +
4096 output tokens without preemption. Separate FA/GDN capacities use 1400/128
slots. A high-page int32 address overflow was found and fixed; FP64 regression
tests cover both decode kernels and cache append. Long strict-JSON requests and
prefix reuse pass through both APIs with MTP4 and the corrected Qwen3.8 model ID.
[Stage evidence](../bench/baseline/2026-09-22-ttft-stage1.json) records raw diagnostic
results. They are not formal throughput/TTFT repetitions. The refreshed 12-row
baseline is now frozen, and all candidate rows have completed on clean ee2fb63.
Ten pass; MTP batch4 throughput is 692.190 tok/s versus the 693.140 gate
(94.8698% of baseline), while prefix batch1 fails TTFT stability despite passing
both ratio gates. Overall acceptance remains pending. See
[comparison and causes](performance-gap-2026-09-22.md) and
[complete candidate evidence](../bench/baseline/2026-09-22-performance-gap-candidates.json).
The following acceptance belongs to the prior task.

## Independent runtime — 2026-09-21

All nine 32768→4096 rows pass 95% against the original frozen EngineCore baseline.
No native baseline was rerun. Each row used two warmups and three measurements,
block 784, the original CPU affinity, and an exclusively monitored UUID-pinned B200.

| Mode | Batch | Original baseline tok/s | Independent tok/s | Ratio |
|---|---:|---:|---:|---:|
| ordinary | 1 | 92.21 | 102.51 | 111.17% |
| ordinary | 2 | 167.87 | 170.36 | 101.48% |
| ordinary | 4 | 250.96 | 278.80 | 111.09% |
| mtp | 1 | 269.16 | 285.73 | 106.16% |
| mtp | 2 | 418.53 | 403.86 | 96.49% |
| mtp | 4 | 446.61 | 546.98 | 122.47% |
| prefix | 1 | 95.16 | 109.19 | 114.73% |
| prefix | 2 | 179.44 | 190.07 | 105.93% |
| prefix | 4 | 333.32 | 336.16 | 100.85% |

[Full independent evidence](../bench/baseline/2026-09-21-independent-acceptance.json)
records raw repetitions, hashes, runtime identity, correctness and discarded runs.
The six ordinary/prefix rows used clean commit ca5a200; the three final MTP rows
used clean commit 5c0848e. The latter changes only MTP execution and adds a new
ProposalGraph class: existing graph definitions are AST-identical, and other
non-MTP production modules are unchanged. Each row retains its actual identity.
This is a historical comparison, not a same-source/same-GPU paired experiment.
Prior failed measurements and contaminated diagnostics remain in the intermediate
artifacts; none replace the frozen baseline or contribute to the final medians.
GPU-process polling cannot exclude arbitrarily short interference between samples.

The final GPU suite passes 92 tests and 12 subtests, without skips. Rust 75 tests,
fmt, hard 100-column Rust checks, clippy, ruff and all hooks pass. Original actual-
model GQA/GDN FP64 tolerances remain unchanged (output atol/rtol 0.03, state NRMSE
1% and relative-peak error 2%). Actual grouped target and draft attention also pass.
Ordinary and MTP text tests cover prefix reuse, staggered arrivals and recompute;
real service tests pass 12 API/reasoning/constraint combinations and cancellation,
stored continuation/deletion and mixed-batch cleanup. All use conda oh-my-vllm.

Final oh-my-pi readback completed on Chat and Responses with MTP4 and medium
reasoning: 7 and 18 successful read calls, zero tool errors, 3 and 6 model turns,
and 26.27 / 36.68 seconds end to end. Both read the required scheduler and Worker
source and correctly described the goal, architecture and independent runtime.
Responses says three ordinary/prefix rows rather than six (three per mode), uses
brackets to denote an optional shell flag, and overstates full-file reading; these
minor answer defects are retained in the evidence, not silently corrected. Its
nine-row all-pass conclusion is correct. The earlier diagnostic-number
misinterpretation is superseded, not claimed as perfectly grounded.

Every request has nonzero MTP proposal/acceptance counts. Cold first-request TTFT
is 6363 ms, explained by preparation (1102 ms) and prefill (4066 + 1194 ms).
The first subsequent decode takes 1536 ms, consistent with graph warmup/capture
overhead but without a separate capture timer. Later TTFT is 652–1917 ms.
Final-answer output rates are 158.78 / 197.25 tok/s. No hang or unexplained sustained
slowdown was observed; these mixed cold/warm workflows have no strict throughput
gate. Owned server/worker processes exited, nvidia-smi shows no compute processes,
and ports 18013/18014 are released.

## Historical V2 Model Runner acceptance — 2026-09-21

The predecessor implementation used V2 exclusively, with Rust scheduling/KV ownership and no
vLLM source changes. All nine 32768→4096 rows pass the >=95% gate against the frozen
2026-09-19 EngineCore measurements. Only the framework was rerun: two warmups and
three measured repetitions per row, MTP4 where applicable, block 784, original
per-row CPU affinity, and one UUID-pinned B200 per run.

| Mode | Batch | Historical EngineCore tok/s | V2 framework tok/s | Ratio |
|---|---:|---:|---:|---:|
| ordinary | 1 | 92.21 | 92.16 | 99.94% |
| ordinary | 2 | 167.87 | 166.96 | 99.46% |
| ordinary | 4 | 250.96 | 252.26 | 100.52% |
| mtp | 1 | 269.16 | 286.98 | 106.62% |
| mtp | 2 | 418.53 | 434.80 | 103.89% |
| mtp | 4 | 446.61 | 451.12 | 101.01% |
| prefix | 1 | 95.16 | 95.60 | 100.46% |
| prefix | 2 | 179.44 | 179.29 | 99.92% |
| prefix | 4 | 333.32 | 324.07 | 97.22% |

[Full V2 evidence](../bench/baseline/2026-09-21-v2-acceptance.json) records every
repetition, commands, frozen-baseline SHA256, binary/worker source hashes, GPU/CPU
identity and correctness results. Measurements used clean implementation commit
`d85c23e`. Actual editable vLLM HEAD is `039b2ad67da6d64f7c1835c4738c7fb545ad37fc`;
installed package metadata still reports the historical `g6376c601e` build label.
This is a historical comparison, not a same-source or same-GPU paired benchmark,
and does not establish that V2 alone caused any speedup. The original baseline
and historical 2026-09-19 MTP-bs2 supplemental repeat are unchanged; that historical
repeat was not substituted for the main baseline.

Independent queues ran on distinct GPUs with process monitoring every second.
6 attempts were discarded in full after detecting outside GPU processes; only
subsequent uncontended runs contribute numbers. Prefix bs2/bs4 originally showed
3.46%/4.47% within-run spread, so framework-only follow-ups used two warmups and
five measurements each: bs2 reaches 100.02% of baseline with 0.09% spread; bs4
reaches 99.94% with 0.07% spread. The artifact preserves both rounds;
the original matrix is not replaced. Even the slowest follow-up measurement
over the fastest historical baseline reaches at least 99.69%.
Polling cannot exclude activity between samples. Every measured request generated 4096 tokens with no preemption.

Actual scheduled GQA/GDN FP64 probes pass ordinary and MTP paths across block 784:
51 and 40 checks respectively, zero nonfinite values. Maximum state NRMSE is
0.2704% and maximum relative error is 0.3916%, below unchanged 1%/2% limits;
output atol/rtol remain 0.03. Two-request text tests preserve city isolation and
1024-token budgets through two ordinary and three MTP preemptions. The latter
combines prefix reuse, staggered arrivals and 1181 accepted drafts.

All 12 Chat/Responses × off/medium × JSON object/schema/strict-tool cases pass
with actual proposed and accepted MTP drafts. Lifecycle checks pass twice and
prove mixed plain/constrained request IDs entered the same batch, and disconnect
cancellation reached a successful Worker release RPC before follow-up success.
All five thinking levels and stored Responses retrieve/continue/delete pass.

Both real oh-my-pi tasks read README/architecture and implementation files
`crates/scheduler/src/lib.rs` and `python/oh_my_vllm/worker/model_runner.py`.
Chat used four assistant turns/nine tools, Responses four turns/eleven tools,
with zero tool errors and positive MTP counters on all eight server requests.
Earlier documentation-only runs are retained separately as preliminary evidence.
Answers describe files at read time, before this final status update; Chat has a
nonblocking `docs/accepting.md` citation typo preserved in the artifact.

Warm short constraint cases report 183.49–283.70 output tok/s and 42–65 ms TTFT,
with zero queue wait in sequential runs. The first request has a roughly 1.1 s
initialization spike. These are serving observations, not a matched benchmark or
new HTTP performance gate. No persistent stall or worker error was observed in
the final runs. All task-owned GPU workers/services stopped and port18002 released.

Checks pass: 75 Rust tests, 39 Python unittest tests, rustfmt, hard 100-character
Rust limit, clippy with warnings denied, ruff and pre-commit hooks. Independent
reviews cover V2 adaptation, cache zeroing, regression protocol and evidence.

The sections below preserve earlier serving and matched EngineCore evidence;
they describe their original binaries, not additional V2 runs.

## Serving acceptance — 2026-09-21

Real Qwen3.8-27B-FP8 serving passed on B200 UUID
GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad with four MTP drafts and block 784.
[Recorded requests, results and metrics](../bench/baseline/2026-09-21-serving-acceptance.json)
include the release-binary identity, commands, actual tool paths and final answers.

- Chat and Responses each completed the real read-only oh-my-pi repository task,
  including README, architecture and source reads, tool-result follow-ups and
  a file-grounded final answer. Both used medium thinking and MTP4.
- All 12 API × off/medium × JSON object/JSON Schema/strict-tool cases completed
  valid output with real proposed and accepted MTP drafts, without fallback.
- All five thinking levels, stored continuation/retrieve/delete, mixed structured
  and ordinary batches, disconnect cleanup and a subsequent request passed.
  Explicit named-tool and no-tool choices passed on both APIs as well.
- After first-request initialization, the 11 short constraint cases reported
  202.8–280.1 output tok/s including request preparation and prefill. Final agentic
  answers reported 208.1 tok/s (Chat, 2,005 output tokens) and 219.3 tok/s
  (Responses, 4,490 tokens). These are workload observations, not a new throughput
  gate or a matched vLLM comparison. Queues were empty in these sequential runs.

The runs found and fixed two defects: Qwen XML wrapping newlines had entered file
paths; uncached tokenizer length queried the vocabulary on every output token
(~29 ms each). Same-GPU constrained runs after caching reduced output processing
to ~0.1 ms per step. DEBUG phase timings are host durations, not kernel timings.
One initial Chat thinking-loop retry recovered; it remains in the saved evidence.
The final answers cite docs as they existed at read time, including the earlier
GPU outage; those status docs were reconciled after acceptance. Both API tasks
were then repeated with updated docs: no tool errors or retries, updated grounded
answers, and no repository diff changes. The artifact retains both rounds.
GPU ownership was
checked with snapshots; this does not exclude brief outside activity.

The EngineCore matrix below predates serving and does not validate the HTTP paths.

## Matched performance

All nine required rows passed the >=95% throughput gate on single B200 GPUs.
Input/output lengths are 32768/4096 per request. Each pair uses identical input
IDs, GPU UUID, inherited single-core CPU affinity, physical KV capacity (1024),
block size (784), greedy sampling and fixed output counts. MTP uses four drafts
and BF16 SSM on both sides. Async scheduling is disabled.

| Mode | Batch | vLLM tok/s | Framework tok/s | Ratio |
|---|---:|---:|---:|---:|
| Ordinary | 1 | 92.21 | 93.74 | 101.66% |
| Ordinary | 2 | 167.87 | 170.03 | 101.29% |
| Ordinary | 4 | 250.96 | 254.64 | 101.46% |
| MTP | 1 | 269.16 | 276.27 | 102.64% |
| MTP | 2 | 418.53 | 434.33 | 103.78% |
| MTP | 4 | 446.61 | 450.04 | 100.77% |
| Prefix | 1 | 95.16 | 96.76 | 101.68% |
| Prefix | 2 | 179.44 | 182.22 | 101.55% |
| Prefix | 4 | 333.32 | 323.75 | 97.13% |

These are medians of three measurements after two warmups per engine.
[Raw data and exact commands](../bench/baseline/2026-09-19-acceptance.json)
preserve clean source commit `12d227d`, executable SHA256, GPU/CPU identity,
vLLM version, configuration and every repetition. Later cancellation and CLI
changes are not attributed to this measured executable.

The baseline uses `LLM.generate` to drive native EngineCore with detokenize=False.
Both timers include request submission through completion; Rust includes
registration, scheduling, transport and final cleanup. Loading, compilation,
warmup and prefix seeding are outside the timer. Each prefix request reports
exactly 32144 cached tokens in both engines. Every MTP measurement proposes and
accepts real drafts. Cold rows report zero prefix hits.

Cases ran concurrently on different idle GPUs, with one CPU core assigned per
pair. These results apply to that recorded configuration, not arbitrary CPU
placement. GPU ownership was monitored during each engine run; no external
same-GPU process was observed. Polling cannot exclude arbitrarily short activity
between samples. Even comparing each row's slowest framework repetition with
its fastest vLLM repetition exceeds 95% (minimum 97.10%).

MTP batch 2 had approximately 2.6% within-engine timing spread, so an additional
[pair with three warmups and five measurements](../bench/baseline/2026-09-19-mtp-bs2-repeat.json)
passed at 98.85% (vLLM 427.32 tok/s, framework 422.42 tok/s). Within-engine
spread fell to 0.67% for vLLM and 0.13% for the framework. The original three
measurements remain intact. This follow-up supports the >=95% gate, not a stable
3.78% speedup claim. It uses the same measured executable with unrelated dirty
source edits recorded; Python worker sources were frozen until it completed.

## Correctness and feature coverage

Actual-path probes instrument real GQA and GDN calls during inference and compare
their rounded inputs/outputs with independent CPU FP64 references. Coverage
includes prefill, decode, crossing 784-token boundaries and all speculative
state prefixes. BF16 outputs use atol=rtol=0.03; recurrent states require NRMSE
<=1% and maximum absolute error <=2% of reference peak. These are eager,
single-request kernel checks, not full-model FP8 token-equality claims.
See [testing](testing.md) and the historical probe paths in [handoff](handoff.md).

Ordinary, MTP and prefix-hit real Chinese text checks passed. The additional
[two-request text evidence](../bench/baseline/2026-09-19-batch-text.json) uses
distinct Beijing/Tokyo prompts, staggered arrivals, 2048 input tokens and 1024
output tokens each. Both beginnings and tails remain coherent and on topic.

- Ordinary: 2 recompute preemptions, zero initial prefix hits.
- MTP plus seeded prefix: 4 preemptions, 3136 initial prefix-hit tokens,
  1143 accepted drafts.

Initial prefix hits exclude hits on re-entry after preemption. The aggregate
hit counter also reports those recompute hits. Fixed output counts deliberately
ignore EOS, matching the performance workload. Text checks test request isolation
and recovery; they do not establish exact output equality with vLLM.

Rust tests cover cache allocation/release, prefix ownership, arrivals, preemption,
speculative state migration/rollback and cancellation. Cancellation is allowed
between completed steps and must flush finished IDs to the worker. CPU adapter
and bridge checks cover cleanup of Python and native Worker request state.

## Operational scope

Timestamped multilevel logs correlate Rust/Python runs and steps; debug host
timers help locate scheduler, IPC and worker overhead. They are not CUDA timings.
GPU test scripts wait for any idle B200, and baseline/worker use conda vllm while
framework tools use conda oh-my-vllm. VSCode's ignored local settings select the
framework environment for Python analysis.

Multi-GPU execution, CPU KV swap, multimodal inputs, LoRA and production
deployment remain outside the agreed scope. OpenAI-compatible HTTP serving is
implemented (see serving.md), with its own acceptance above; it is not covered by
this historical performance evidence. These measurements remain EngineCore-level.
