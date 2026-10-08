# Requirements

These requirements define the active scope.
See [acceptance](acceptance.md) for measured source identities and [audit](audit.md) for open qualifications.

## REQ-GOAL-001: Inference goal

Supply Rust inference for Qwen3.8-27B-FP8 on one B200 GPU.
Keep all existing offline and HTTP features.
Each performance row must meet `REQ-PERF-001` and `REQ-PERF-002`.

## REQ-ARCH-001: Ownership

Rust controls service, request queues, token budgets, logical KV allocation, prefix reuse, and preemption.
Python controls model loading, GPU forward computation, kernels, and manual CUDA Graphs.
Use one ZMQ DEALER channel with msgpack.
Keep tensors on the Python side.

## REQ-MODEL-001: Target

Use Qwen3.8-27B-FP8 with 48 GDN layers and 16 FA layers.
Use block size 784 on all production paths.
Use FA group 0 and GDN group 1 with `mamba_cache_mode="align"`.
Set the checkpoint location through `OH_MY_VLLM_MODEL` or `--model`.

## REQ-PERF-001: Throughput

| Mode | Input tokens | Output tokens | Batch sizes |
|---|---:|---:|---|
| ordinary, MTP4, prefix-hit | 32768 | 4096 | 1, 2, 4 |
| ordinary | 131072 | 4096 | 1, 2, 4 |

These combinations make twelve rows.
Each row must have throughput of at least 95% of the pinned EngineCore baseline.
Use identical tokens, sampling, output counts, and comparable effective cache capacities.
Record source, environment, and reported cache identities.

## REQ-PERF-002: TTFT and measurement

Each row must have TTFT of at most 110% of the baseline.
Start the monotonic clock at the same pretokenized batch submission, before request registration.
Stop each request's clock when the caller receives its first kept output token.
Include engine queue time, scheduling, transport, and sampling.
Exclude HTTP, tokenization, loading, and warmup.

Use the median of per-repetition maximum request TTFTs as the row statistic.
Report GPU prefill time and cold latency independently.

Use at least two full warmups and five measured repetitions.
Reset prefix reuse for ordinary and MTP measurements.
For prefix rows, seed exactly 32144 reusable tokens per request.
The two engines must have `(max-min)/median <= 0.10` for throughput and TTFT.
Investigate excess spread and repeat the full set.

Keep excluded attempts and reasons. Do not select individual repetitions.
Compilation, new graph capture, interference, or changing source invalidates a measured attempt.
Fix failures. Change thresholds only with user authorization.

The pinned baseline uses official vLLM source `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
Original baseline SHA-256 is `fa3729f1a2ce160235b45df75542774d628dac7af963f01d673353fb419df8fd`.
Use full original data for formal comparison.
On a new server, measure a new baseline with the same protocol and matching conditions.
Historical portable summaries are for reference and offline analysis.

The next performance-policy goal derives thresholds from roofline analysis instead of vLLM measurements.
The existing thresholds stay active until a new requirement replaces them.

## REQ-PERF-003: DSpark and native MTP4

Compare optional DSpark with native MTP4 from the same source and binary after implementation.
Use 32768 input tokens, 4096 kept output tokens, and batches 1, 2, and 4.
Use the same synthetic token IDs, greedy sampling, and fixed output counts. Ignore EOS.
Include registration, prefill, scheduling, transport, sampling, and cleanup in EngineCore throughput.

One comparison worker keeps the target weights and physical target caches for each batch.
Load the two draft models before warmup. Reset prefix reuse before each attempt.
For each batch, use three rounds.
Each round has two full warmups per mode and five measured pairs with alternating mode order.

Report these statistics for each batch:

- Each mode's throughput median and spread in each round.
- Each round's DSpark median difference from MTP4.
- The hierarchical paired-bootstrap one-sided 95% lower bound on throughput gain.

Report TTFT and its spread.
Do not add a DSpark speed, spread, or confidence-bound acceptance gate for this comparison.
The twelve existing vLLM throughput and TTFT gates stay active.
Keep each failed or interrupted attempt and its full raw records.
Compilation or graph capture during measured work invalidates an attempt.

Bind records to source files, binary, Python environment, loaded CUDA module, runtime configuration, and checkpoint bytes.
Compare loaded DSpark configuration and weight hashes with the requested checkpoint. They must agree.
Record capacities, peak allocated/reserved memory, compile/capture audit, and scheduled/accepted draft counts.
Acceptance rate uses scheduled draft tokens as its denominator. Returned proposals for future steps are a different counter.

Current-source DSpark performance acceptance results are not available.
See [testing](testing.md#dspark-comparison) for the collection procedure.

## REQ-CONTEXT-001: Context and memory

Support input plus output up to 262144 tokens.
Complete the full 258048-input, 4096-output cases in ordinary and MTP4 modes at batches 1, 2, and 4.
These runs must have no OOM or recompute preemption.
Keep the 32768-token step budget and 784-token blocks.
Record performance and peak GPU memory. Add no extra ratio gate.

Include long-context prefix restoration and constrained decoding.
Apply the same boundary cases to DSpark at batches 1, 2, and 4.
Keep the six ordinary/MTP4 cases as different gates.

## Functional requirements

| ID | Contract |
|---|---|
| REQ-FUNC-001 | Chunk prefill with at most 32768 tokens in one step. Materialize aligned GDN checkpoints. |
| REQ-FUNC-002 | Schedule active decode work before new prefill work in each step. |
| REQ-FUNC-003 | Use chain-hash prefix reuse, LRU eviction, and coordinated FA/GDN groups. |
| REQ-FUNC-004 | Preempt later-admitted running requests first. Keep accepted history and clear unverified drafts for recompute. |
| REQ-FUNC-005 | Supply MTP4 with explicit kept tokens and new draft IDs. Keep BF16 GDN state and block 784. |
| REQ-FUNC-006 | Supply optional DSpark with the local BF16 checkpoint, at most seven drafts, target verification, and block 784. |

DSpark uses zero-based target layer outputs `(5,19,33,47,61)` and five draft GQA layers.
Keep native MTP4 available. Select one draft mode for the worker, not per request.

The initial DSpark cumulative confidence threshold is `0.2`. The proposal count can be zero through seven.
A first confidence product less than the threshold gives one target-token step.
Set `--dspark-confidence-threshold 0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.

For stochastic sampling, use conditional proposal probabilities and target acceptance `min(1,p(x)/q(x))`.
On rejection, sample from normalized `max(p-q,0)`. Sample the bonus token from the target distribution.
Apply user sampling parameters and grammar masks to each conditional distribution.

Keep BF16 GDN state and the existing half-precision rounding constraints in speculative modes.
Only accepted input rows can commit target state or DSpark context.

## REQ-ACC-001: Numerical accuracy

Compare model GQA and GDN inference paths with independent references with CPU FP64 computation.
Include prefill, decode, chunk boundaries, and recurrent state.
Use rounded inference inputs.
BF16 output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of the reference peak.

Keep path-specific tolerances and existing tests.
Show coherent text, MTP, and prefix hits.
Full-model token equality is not a gate.

## REQ-RUNNER-002: Independent runtime

Use project implementations and third-party libraries.
Project builds, tests, and inference must not install, import, or link vLLM.
They must not depend on its checkout, old environment, or compiled caches.
The former adapters are removed. [ADR-002](decisions/ADR-002-gpuworker-adapter.md) and [ADR-005](decisions/ADR-005-v2-model-runner.md) record their replacement.

`REQ-RUNNER-001` is retired and superseded by this requirement.
Select compatible stable dependencies in dependency order and pin versions with satisfactory compatibility tests.
Keep ported-code provenance and licenses.
Document future architectures, multiple GPUs, and other NVIDIA GPUs briefly.

## REQ-OBS-001: Diagnostics

Supply UTC timestamps, monotonic durations, run/request/step correlation, and log levels that the user can set.
Avoid per-step I/O at the default log level.
Debug logs show scheduler, cache, transport, and worker durations.
Identify host time independently from CUDA kernel time.
Make profiling an explicit choice.

## Service requirements

| ID | Contract |
|---|---|
| REQ-SERVE-001 | Supply text Chat Completions and Responses, streams, tools, history, usage, model discovery, retrieval, and deletion. |
| REQ-SERVE-002 | Supply off/low/medium/high/xhigh thinking. Default medium, high maps to xhigh, none maps to off. |
| REQ-SERVE-003 | Complete oh-my-pi tasks through each API with MTP4 and DSpark. Read files without error and use their returned results in subsequent model input. |
| REQ-SERVE-004 | Apply JSON/Schema/strict-tool masks before sampling, with each mode's drafts and bonus tokens. Reject unsupported schemas. |
| REQ-SERVE-005 | Examine queue time, preparation, TTFT, output rate, cache behavior, and draft counters without an extra HTTP throughput gate. |

[Service contracts](serving.md) define supported parameters, XML restrictions, storage limits, and errors.
Scripted workers alone cannot show target-model acceptance.
For each draft mode, test the two APIs, constraints, cancellation, mixed batches, and long-context prefix reuse.
OMP evidence must show two source reads, forwarded results, continued generation, and an answer that uses those results.
Compare response IDs with server logs for the active mode and scheduled drafts.

## REQ-KERNEL-001: Pinned TileLang comparison

The owned Triton-to-TileLang migration is full.
Keep the accepted TileLang implementation pinned as the comparison for `REQ-KERNEL-002`.
The migration does not include third-party internals.
Use the pinned TileFoundry fork as a development tool.
Discuss substantial fork changes before implementation.

Keep temporary tuning in an external directory.
The former Triton backend supplies no acceptance threshold.

## REQ-KERNEL-002: CUDA and PTX

Use owned CUDA C++ kernels with optional inline PTX for B200.
CUTLASS is permitted.
CUDA is the default. TileLang is an explicit comparison choice with no silent fallback.
Apply accuracy, framework, context, and model service gates independently.

Statically derive the largest legal invocation of each implementation path in each of the twelve workloads.
Deduplicate identical configurations.
Use different phases where implementation paths differ.
Do not independently maximize incompatible dimensions or use dynamic tracing for formal case selection.
Each case must be faster than pinned TileLang. There is no minimum percentage gain.

Use at least three rounds of twenty interleaved pairs.
Each CUDA round median must be faster.
The one-sided 95% confidence bound on paired time saved must be positive.
Equivalent fused chains include all required copies and reductions.

Keep formal cases, the harness, and summarized evidence in the repository.
Keep raw traces and temporary tuning in an external directory.
Profile independently from acceptance timing.
Report unavailable counters. Keep the gates active.
Use [kernel procedures](kernels.md).

DSpark adds independently derived supplemental operator cases.
Keep the old case set and pinned TileLang source hash unchanged.
Keep the new TileLang reference in a different file from that pinned implementation.
Apply the same accuracy, three-round, twenty-pair, and confidence-bound gates to each new case.

## REQ-IR-001: Semantic IR

Control the semantics and provider selection of project CUDA and key FlashInfer operations in the Qwen target and MTP paths.
Ordinary PyTorch operations, embedding, reshapes, and `F.linear` are not in this operator inventory.
Each operation must have a PyTorch reference, shape/dtype contract, mutation schema, and statically selected providers.
Provider selection uses phase, shape, dtype, layout, and device metadata.
Provider failure is an error. The reference is an explicit debug choice.

Compile prefill, target decode, MTP draft, and four-step proposal with `torch.compile(fullgraph=True)`.
Compile DSpark target features, context injection, backbone, Markov step, and greedy proposal with the same fullgraph contract.
Keep semantic nodes until provider lowering.
Keep host planning, allocation, scheduling, and ZMQ not in these units.
Manual CUDA Graphs contain compiled units. Disable compiler-managed graphs.

Keep ordered persistent writes, capture restoration, and memory guards.
Any activation donation must have proof for the named temporary. Persistent caches cannot be donated.
Graph rewrites must have equivalence tests and an exception-aware call-site inventory.

Acceptance includes all existing accuracy tests, ordinary/MTP4 context cases, formal operator cases, and twelve performance rows.
Add DSpark boundary, service, supplemental operator, and paired-performance cases.
A target-model test must exercise all four compiled units and graph replay with changed metadata.
There is no new compile-speedup percentage gate.

## REQ-LEARN-001: Interactive tutorial

Maintain a Twine-engine tutorial in `code-journey/` with thirteen full Chinese chapters.
Include core inference logic, decisions, and implementation through project source.
Use reading choices for interests and internal prerequisites.
Give a description of displayed fields, parameters, and variables. Assume basic Rust and Python syntax.

Use LXGW WenKai, Fira Code Nerd Font, KaTeX, and 17/16px desktop/mobile prose.
Keep theme selection and reading/experiment progress.
Use halfwidth Chinese punctuation and the requested spaces. Apply Humanizer-zh.
Supply the user-requested preview on port 18084.

Keep its server address and process details in `LOCAL.md`.

Use conditional unordered lists for each page transition.
Only an explicit completion choice completes a chapter.
Review/map visits keep reading progress. Old short passages migrate to visits.
Extract current source with attached documentation, attributes, and decorators.

Remove the leading indentation that is the same on all lines that contain text.
Keep relative indentation, line numbers, and full-file hashes.
Use identical displayed and copied excerpts, readable highlighting, full-source pages, and keyboard scrolling.
Reject missing source coverage, anchors, and record-field explanations at build time.

Test thirteen routes, prerequisites, choices, completion, migration, restart, refresh, and backward history.
Test five Rust CPU traces: 784, 785, 1568, 1569, and 32768 tokens.
Test fonts, formulas, fields, console health, desktop/mobile layout, and the two themes.
Test excerpt documentation, dedentation, clipboard paths, source hashes, line anchors, URLs, and code contrast of at least 4.5:1.
Examine screenshots against the requested reading style.

Synthetic CPU feedback shows scheduling. GPU evidence has its own acceptance records.

## REQ-DOC-001: Documents and portability

Write project-owned English Markdown to ASD-STE100 Issue 9 rules and dictionary.
Maintain full Chinese translations, a technical glossary, automated checks, and semantic review by another agent.
Keep useful current information and key historical decisions.
Keep host details in ignored `LOCAL.md` and original evidence in ignored local storage.
Use environment variables and CLI parameters consistently across Rust, Python, scripts, and tests.

Clean the current tracked content. Keep Git history.

## REQ-OUT-SCOPE-001: Future scope

Current scope excludes multimodal inputs, CPU KV swap, multiple GPUs, tensor parallelism of more than one, gRPC, LoRA, and production packaging.
Document extension points without empty interfaces or untested claims.
