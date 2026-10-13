# Acceptance evidence index

Use [requirements](requirements.md) for gates and [testing](testing.md) for collection commands.
Each result applies to its stated source, configuration, and workload.
Do applicable collections again after inference changes.
Originals supply formal verification. Portable summaries give results for reading and offline analysis.

## Current collection

Acceptance checks the inference framework and its active numerical, performance, context, and serving contracts.
OMP goal status, task progress, tool errors, and final answers are workload observations.
The hour checks continuous framework availability, sampled resource limits, and owned sandbox cleanup.
Original client and collector failures keep their recorded status.

Formal GPU collections on clean `ae070e4` that stayed the same used new project caches and pinned dependencies.

| Check | Measured result |
|---|---|
| GPU suite | 487 passed, no skips |
| Phase latency | Fifteen rows passed, two warmups and five measurements per row |
| Maximum context | Six ordinary/MTP4 cases passed at batches 1, 2, and 4 |
| Reasoning budget | Sixty cases passed |
| Service protocols | Constraints, lifecycle, and long context passed in MTP4 and DSpark with the two APIs |
| Operators | 517 numerical passes, 514 speed passes, three speed failures |
| Framework hour | MTP4 and DSpark each passed the independently reviewed 3600-second window |
| Allocator recovery | Six stages passed, two first cold stages stopped before injection at the old worker-step deadline |
| Oneoff OMP | Original attempts failed before model requests because the settings overlay was missing |

A second B200 measurement with a new project cache reproduced the three operator speed failures.
The user deferred optimization. The full operator matrix does not pass acceptance.
See [audit](audit.md#krn-07-small-normalization-shapes).

MTP had 412 requests, 407 server completions, and five matched terminations, with the final stop tail.
DSpark had 87 requests, 79 server completions, and eight matched terminations, with the final stop tail.
The two windows had fixed source, process birth identities, and epoch, without unexpected OOM or worker restart.
All eight agent sandboxes, generated files, copied tools, caches, and owned images were removed.

Worker RSS increased during the hour. MTP grew from 1.136 to 6.822 GiB. DSpark grew from 1.280 to 8.519 GiB.
Worker snapshots lagged collection by up to 136.230 and 138.827 seconds.
Resource checks do not prove a memory plateau, full allocation attribution, or stability beyond the measured hour.
Natural compaction coverage and original journal/SDK errors are separate workload observations.

The subsequent changes adjust timeout policy and collector evidence scope without changes to GPU computation or constrained decoding.
The user waived more tests for timeout parameters alone.
New CPU checks, missing recovery stages, and corrected oneoff compatibility checks are in final verification.
The [handoff](handoff.md) gives results and limits. Earlier measurements stay scoped to their recorded source.

## Partial collection on `f019075`

Independent review accepted sixty reasoning-budget cases, six full-context cases, eleven phase rows, and all `469` GPU tests.
The other phase rows and full `517`-case operator matrix did not finish.
One operator attempt saved `42` rows with one negative confidence bound. Other attempts stopped with `158` and `16` rows.
These records do not supply full operator acceptance.

Neither persistent collection completed all four active lanes for one hour.
The MTP collector lacked independent SDK usage for six natural compaction calls.
The DSpark collector counted a normal `turn_end` as another SDK completion. Host memory stability has no accepted result.
All owned collectors, GPU processes, and temporary model-service endpoints stopped before collector and diagnostic changes.

## Historical collection on `706f1a6`

The [serving resource record](../bench/evidence/2026-10-11-serving-stability.json) uses source `706f1a6e7ce2b0188f07794cab574cbbf60f2927`.
The release binary SHA-256 is `ceb58b4e6b248fc657b567fe82a9e2211d88d565a73cceaceffc7a1b944fd890`.
The record keeps original hashes, measurements, failed attempts, and independent reviews.
The final publication also corrects Markdown image checks.
It keeps the measured inference code, binary, and selected CUDA configurations.

### Phase latency

All fifteen workloads passed their active prefill and decode checks.
Each workload has two full warmups and five full measurements, with 4096 output tokens per request.
Prefix-hit rows reuse 32144 tokens. Other rows use cold prefixes.
All measured requests have full output and zero recompute preemptions.

Each row comes from one full, continuous workload collection.
The rows use the same source, binary, environment, capacity, and formal contract.
The independent review calculated the full prefill and decode models again for all 75 measured schedules.
It checked source identities, loaded modules, output, trace arithmetic, statistics, and owned worker exit.
The two phase models include all required operations.

The [README table](../README.md#measured-performance) gives prefill seconds, lower-bound ratios, decode TPS, and theoretical percentages.
Prefix-hit prefill records its ratio. Its spread and decode gates passed.
Decode TPS counts 4095 tokens per request because the first token belongs to prefill.

### Operator matrix

The full 317-case CUDA/TileLang collection uses source `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`.
All numerical and speed checks passed. The matrix keeps the original 230 cases and reference pins.
At the `706f1a6` publication, owned operator implementations and operator case, timing, and comparison files matched that collection.
That unchanged source proof covers `python/oh_my_vllm/kernels/`, `development/kernels/`, and `benchmarks/kernels.py`.

The new long-prefill case selection has subsequent changes. Do not extend the historical full-harness proof to those changes.

The ownership utility and IR have subsequent changes. The old operator result keeps its original source scope.
The [operator record](../bench/evidence/2026-10-10-current-source-acceptance.json) keeps its source, original hashes, and independent review.

The review calculated 951 round medians and confidence bounds again from 19020 pairs.
Each confidence calculation used 10000 bootstrap samples. The results agreed with the original record.
Each timed graph contains 100 full operations. Each sample replays that graph once.
The review checked full outputs, state writes, dispatch, loaded libraries, headers, templates, and pinned comparison sources.

Five current-source GPU regressions passed in 19.39 seconds.
They check compiler metadata retirement, existing graph replay, cold recapture, allocator reclamation, and side-stream owner release.
They also check graph pressure recapture. Allocated memory decreased to its initial value.
The independent review checked all 39 owned process exits and the full test logs.

### Context capacity

The six ordinary/MTP4 cases passed input 258048 and output 4096 at batches 1, 2, and 4.
They have no OOM or recompute preemption. Total length is 262144.
Each case uses zero warmups and one cold repetition. Compilation and capture can occur during capacity collection.
Capacity timing and memory observations have this cold-run scope.

| Mode | Batch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|
| ordinary | 1 | 120.625252 | 144.326172 |
| ordinary | 2 | 120.625252 | 146.617188 |
| ordinary | 4 | 121.478909 | 145.777344 |
| mtp4 | 1 | 117.010298 | 144.105469 |
| mtp4 | 2 | 117.010298 | 144.169922 |
| mtp4 | 4 | 119.754255 | 146.652344 |

The six-row collector and its supervisor completed successfully.
Two independent reviews checked full raw logs, source identities, full output, measured memory, worker exit, and removed IPC endpoints.
DSpark keeps phase, numerical, operator-speed, and service acceptance.

### Long-duration agentic service

MTP4 and DSpark each completed at least 3600 seconds and 500 normal requests.
Each collection used four oh-my-pi agents, with at most four model calls at the same time.
Chat and Responses executed the required source reads and supplied their results in subsequent model input.
Warmup, cancellation, slow clients, overload, and injected faults have separate records.

| Mode | Normal requests | Agent tasks | Normal duration (s) | Result |
|---|---:|---:|---:|---|
| MTP4 | 5135 | 2430 | 3675.338 | pass |
| DSpark | 3104 | 1461 | 3665.792 | pass |

Normal requests had no unexpected OOM, worker restart, or endpoint failure.
Drained request maps and admission leases were empty. Each logical pool kept only null block 0 allocated.
Graph counters stayed at or below their count limits. Compiler metadata stayed at or below its count and byte limits.
Host RSS and GPU allocated, reserved, and device-used memory passed the last-half median checks.

These medians use drained snapshots in the last temporal half.
For those ordered samples, compare medians from the first and last halves.
The full record keeps all five memory metrics and their specified allowances.

| Mode | Initial late worker RSS median (GiB) | Final late worker RSS median (GiB) | Worker RSS growth (MiB) | GPU reserved growth (MiB) |
|---|---:|---:|---:|---:|
| MTP4 | 5.375099 | 5.546391 | 175.402344 | 48.000000 |
| DSpark | 6.697308 | 6.697308 | 0.000000 | 830.000000 |

The full raw review calculated each request point again from the server's processing start and monotonic processing interval.
It parsed each saved SSE body again and compared terminal records, SDK messages, tool results, and server metrics.
Output-limit answers keep their original `length` reason and separate counts.
Their protocol, client, request budget, and completed server output counts must agree.

The [README plots](../README.md#long-duration-agentic-service) use one point per normally completed model request.
The horizontal axis is framework processing start. The vertical axis is output tokens divided by processing wall time.
Queue time is separate. This heavy test has no performance gate, and daily commands skip it.

Tracked SVG files use whitespace formatting. The record keeps original and published figure hashes separately.

### Overload and allocator OOM

MTP4 and DSpark passed the bounded-body overload test.
It held 64 partial HTTP bodies. The next request received a retryable 503 response without an admission lease.
After clients closed, request resources were empty and Chat and Responses returned the answer `25` without a worker restart.

All eight allocator-OOM recovery cases passed.
Each mode has faults after accepted-state commit, graph capture, sampling, and proposal generation.
Each affected stream reported one error without a successful completion.
The same HTTP process stayed alive and supplied 503 during recovery.
One replacement worker served Chat and Responses, with cold then reused prefixes.
Independent review checked original hashes, request state, resource limits, cache invalidation, and all owned process exits.

### CPU checks and verification limits

The frozen-source CPU suite passed 1293 tests and 99 subtests, with 448 GPU tests deselected.
The final document checker passed 14 regressions, with image syntax and invalid-path checks.
Rust workspace tests passed 143 cases. Formatting, Clippy, line-width, Ruff, CUDA formatting, and document checks passed.
The tutorial build, ten excerpt tests, and nine browser tests passed.

Each long-duration result applies to its full one-hour workload and observed shapes.
The [audit](audit.md) keeps maximum-context cross-shape graph recapture and host-overlap work open.
Generated answer meaning and source citations have a separate diagnostic review.
Executed reads and exact tool-result forwarding are required checks.

The target loader has no in-process weight-digest record. Original collectors bind the checkpoint files separately.
Phase GPU ownership checks occur once per second and after exit. Interference fully between polls cannot be excluded.
The long-duration supervisor records other GPU processes and permits light foreign use.
All owned GPU workers and temporary service listeners exited after collection.

## Failed and canceled collections

Original failed or canceled collections keep their original status and hashes in the current resource record.
Their samples do not enter accepted phase rows or normal request plots.

One MTP4 attempt stopped because its captured agent task lacked the two required source reads.
The captured SSE and SDK message contain text through the output limit without tool calls.
The unchanged full workload passed on a fresh attempt.
Requests with the same body and cold diagnostic requests returned the required tool calls.
The cold diagnostic did not execute those calls.

The original attempt has no token trace before SSE conversion.
The agent requests use the default sampling temperature `1.0` without a fixed seed.
These observations apply to the captured outputs and stated verification scope.

A stopped recovery collection completed one fault stage before GPU reassignment.
The accepted collection runs all eight stages again. An interrupted overload collection keeps its foreign-process monitor failure.
Original records keep earlier memory and collector failures. The current record identifies the corresponding runtime and collector corrections.

## Selected CUDA implementation

The native module uses ten operator headers. All owned CUDA backend code files meet the 800-line limit.
Header content enters build keys, load provenance, formal contracts, and profiler checks.
The clang-format hook uses version 23.1.3.

Five FP8 projections at 624 through 2496 rows use an owned paired TMEM accumulator and a direct BF16 epilogue.
The 317-case operator collection at `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb` verifies the selected implementation.
The [accumulator diagnosis](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) keeps its earlier measurements and profiler records.
[Profiling](profiling.md) gives selected configurations, rejected candidates, and measured optimization results.
The [owned CUDA audit](../bench/evidence/2026-10-10-owned-cuda-audit.json) records source checks and external tuning candidates.
Those candidates supplied no new production selection during cleanup.

## DSpark threshold selection

DSpark keeps fixed weights and shares the target embedding and vocabulary head.
Its initial cumulative confidence threshold is `0.2`. Set it to `0.0` for fixed-count diagnostics.
Output, context, and grammar limits still apply.

The [threshold-selection record](../bench/evidence/2026-10-08-dspark-confidence-selection.json) uses inputs different from the formal synthetic workload.
It has two text/tool-history requests, with input 32768 and output 512 per request.
Each setting has two full warmups and two measured repetitions.
Thresholds `0.0`, `0.05`, `0.1`, and `0.2` have TPS medians 177.287, 183.097, 184.198, and 184.323.
Their acceptance rates are 28.487%, 44.957%, 53.406%, and 62.500%. These medians apply to this small collection.
