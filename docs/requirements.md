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

## REQ-EXEC-001: GPU execution

Use CUDA Graph, multiple CUDA streams, and Programmatic Dependent Launch (PDL) for model computation.
Use semantic operations and tensor/state access to define graph dependencies.
Independent branches can run in parallel.
PDL consumers must wait before they read producer output.

Keep graph capture restoration, cache ownership, and bounded graph storage.
Tune execution with full-operation and framework measurements.
Record the configuration, numerical checks, memory, and latency of each selected optimization.
Use separate measurements to identify graph, stream, and PDL effects.
The phase-latency and operator gates stay active.

## REQ-PERF-001: Phase latency

| Mode | Input tokens | Output tokens | Batch sizes |
|---|---:|---:|---|
| ordinary, MTP4, prefix-hit, DSpark | 32768 | 4096 | 1, 2, 4 |
| ordinary | 131072 | 4096 | 1, 2, 4 |

These combinations make fifteen rows.
Use synthetic token IDs, greedy sampling, and fixed output counts with ignored EOS.
Reset prefix reuse before each repetition.
For prefix-hit rows, seed 32144 reusable tokens per request.
Keep each full workload resident with zero recompute preemptions.

Record submission, first kept token, and last kept token for each request.
Prefill is submission to the first token. Decode is the first token to the last token.
For each repetition, use the maximum request prefill interval and maximum request decode interval.
Use each request's own boundaries when phases overlap in a batch.

Include registration, queue time, scheduling, Python, transport, and sampling.
Exclude loading, tokenization, HTTP, and warmup.


Record full batch time, cleanup time, TPS, acceptance rate, and GPU diagnostics independently.

Except for prefix-hit prefill, the wall-time median must be at most three times the theoretical lower-bound median.
The wall-time spread must be at most 10%.
Record the prefix-hit prefill latency ratio, wall time, and theoretical bound without a latency gate.
Its 10% spread check stays active. Prefix-hit decode keeps the three-times limit.

## REQ-PERF-002: Theoretical model and measurement

Use a fixed semantic DAG with data, token, layer, and persistent-state dependencies.
Independent branches can run in parallel.
CUDA stream order supplies no extra dependency.
Record effective queries, KV lengths, state access, drafts, verification, and rollback for each executed step.
Canonical rules must stay the same when kernels split or fuse.

The theoretical lower bound is the maximum of these limits:

- The semantic DAG critical path.
- Necessary HBM traffic divided by the official single-GPU bandwidth.
- Necessary work on each shared execution resource divided by its official peak rate.

Include parameter reads, scales, model computation, GPU sampling, KV/state updates, drafts, and rejected verification rows.
Use the precision and execution resource of each operation.
Count all seven DSpark backbone rows when the algorithm executes seven rows.
Keep padding, repeated transfers, and unnecessary computation in overhead diagnostics.
They must not increase the theoretical bound.

Use single-GPU, dense B200 peaks: FP8 Tensor 4.5 PFLOPS, BF16 Tensor 2.25 PFLOPS, FP32 75 TFLOPS, and FP64 37 TFLOPS.
Use HBM bandwidth 7.7 TB/s.
Use published instruction rates for other known execution resources.
Record hardware identity, peak sources, operation rules, and model version.
Measured rates and empirical efficiency factors cannot increase the bound.

Use ideal cache reuse and fused intermediate storage on chip.
Use finite storage capacities to calculate minimum necessary HBM traffic.
Unpublished on-chip bandwidth is infinite in this relaxation.
Identify all GPU operations. An unidentified operation blocks acceptance.

Use two full warmups and five measured repetitions per row.
Calculate a new bound from each repetition's executed schedule.
Apply these two independent checks to prefill and decode:

```text
median(wall_time) <= 3 * median(theoretical_lower_bound)
(max(wall_time) - min(wall_time)) / median(wall_time) <= 0.10
```

For prefix-hit prefill, record the first check's result without a failure gate.
Apply the second check to each phase.

Keep complete failed sets and interrupted attempts with their reasons.
Investigate a failure. Run the affected workload again with two full warmups and five measurements.
Do not select samples.

Compilation, graph capture, interference, or source changes invalidate measured work.
Connect each attempt to source, binary, environment, CUDA module, configuration, and checkpoint identities.
Portable summaries cannot replace original records for formal verification.

## REQ-CONTEXT-001: Context and memory

Support input plus output up to 262144 tokens.
Complete the full 258048-input, 4096-output cases in ordinary and MTP4 modes at batches 1, 2, and 4.
These runs must have no OOM or recompute preemption.
Keep the 32768-token step budget and 784-token blocks.
Record performance and peak GPU memory. Add no extra ratio gate.

Include long-context prefix restoration and constrained decoding.
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

Record scheduled, accepted, and returned draft counts for each mode.
Acceptance rate is total accepted drafts divided by total scheduled drafts.
Returned next-step proposals have a different diagnostic counter.

Compare loaded DSpark configuration and weight hashes with the requested checkpoint. They must agree.

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

## REQ-KERNEL-002: CUDA and PTX

Use owned CUDA C++ kernels with optional inline PTX for B200.
CUTLASS is permitted.
CUDA is the default. TileLang is an explicit comparison choice with no silent fallback.
Apply accuracy, framework, context, and model service gates independently.

Split the owned CUDA backend by function. Each `.cu`, `.cuh`, and `.py` file must have at most 800 lines.
Use pinned clang-format 23.1.3 and the pre-commit hook. Keep include dependencies in order.
Connect all source and headers to build, load, and formal evidence identities.

Statically derive the largest legal invocation of each implementation path in each of the fifteen workloads.
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
Record unavailable counters. Keep the gates active.
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

Acceptance includes all existing accuracy tests, six ordinary/MTP4 context cases, formal operator cases, and fifteen phase-latency rows.
Keep DSpark service and supplemental operator cases active.
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
