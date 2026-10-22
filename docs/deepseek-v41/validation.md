# DeepSeek V4.1 validation plan

## Status and scope

This is the acceptance proposal for the [architecture](design.md) and [phase plan](plan.md).
It does not report completed tests.
Document-only changes use applicable document and whitespace hooks.
They do not trigger GPU measurements or changes to old performance values.
Implementation changes use applicable CPU, numerical, state, memory, and performance checks.

## HTTP matrix

Input and output lengths apply to each request, not to the full batch.
All configurations use four B200 GPUs and total context capacity 1048576.

| Mode | Parallel strategy | Batch sizes | Input tokens | Output tokens | Cases |
|---|---|---|---|---|---:|
| DSpark, maximum five drafts | TP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| DSpark, maximum five drafts | TEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| DSpark, maximum five drafts | DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 12 |
| Ordinary decode | TP4 | 4 | 400000 | 64, 8192 | 2 |

The proposed total is 38 HTTP cases.
The user specified the batch/length products but did not assign HTTP parallel strategies.
The full DSpark product across strategies and the TP4 ordinary baseline are planning assumptions for approval.

Do not silently treat them as approved scope.
If HTTP uses the AgentX batch-to-strategy mapping, the total becomes 14 HTTP cases.
Do not add ordinary-decode performance cases beyond the two approved cases.
Small correctness checks are separate from performance cases.

Use uncached prefixes and exact token counts after model-specific prompt encoding.
Use the current tools workload semantics with a DeepSeek codec and a versioned sampling profile.
Freeze temperature, top-k, top-p, penalties, seeds, reasoning mode, and tool schema before measurement.

Use the same request distribution for ordinary and DSpark comparisons.
Keep sampling and grammar active. Do not substitute greedy unconstrained generation.
Load weights before timing.
Finish initialization before timing.

Include HTTP submission, preparation, queueing, prefill, decode, sampling, communication, and response parsing in the full timer.
Report initialization and timeout settings separately.
Check that the configured request limit admits the specified batch.
Record aggregate and per-request timings, not only time to first token.

## AgentX matrix

Use DSpark with maximum five drafts only.

| Session-tree concurrency | Strategy | Cases |
|---:|---|---:|
| 4 | TP4 | 1 |
| 16 | TEP4 | 1 |
| 64 | DEP4 | 1 |

There are three AgentX cases.
AgentX concurrency counts session trees, not necessarily simultaneous HTTP requests.
Record the configured and observed HTTP concurrency, with a service limit of 64 active requests.

Record queueing when tree branches submit additional requests.
Do not reduce the requested tree concurrency to avoid queueing.
Use the existing pinned harness and dataset first.

Record dataset identity, tokenizer, seed, trajectory rules, warmup, and drain settings.
Use at least 900 seconds of profiling per case and the existing 120-second drain default.
Record overflow, failed, cancelled, incomplete, and successful request counts separately.
Use full response IDs to join streaming usage and service traces.

The current dataset has 256k traces. It does not prove 1M capacity.
Use separate capacity checks for that target.

## Function checklist

Keep these functions for Qwen.
Implement their DeepSeek equivalents:

| Area | Acceptance items |
|---|---|
| HTTP | Model listing, Chat JSON/SSE, Responses JSON/typed SSE, diagnostics |
| Stored responses | Retrieval, deletion, continuation, storage limits, and `store=false` |
| Offline | `run`, `bench`, token/prompt files, arrival control, prefix reuse, fixed output lengths |
| Sampling | Temperature, top-k, top-p, seed, frequency/presence/repetition penalties, cycle guard |
| Stops | EOS, `ignore_eos`, stop strings, minimum output, maximum output, context limit |
| Reasoning | Effort selection, thinking budget, incremental parsing, usage accounting |
| Tools | Auto/none/required/named choice, parallel-call control, results in history, streamed arguments |
| Structured output | JSON object/schema, token masks, independent speculative grammar state |
| Media | Ordered text/images/video, limits, frame timestamps, history, prefix isolation |
| Cache | Full and partial prefixes, copy-on-write, preemption, host offload, restoration, eviction |
| Lifecycle | Deadlines, disconnect cancellation, backpressure, worker failure/replacement, graceful shutdown |

Keep current restrictions unless separately approved.
There is one completion per request, no logprobs, no background execution, and no automatic truncation.
Enabled tools and structured-output mode cannot be combined in the current contract.
Model-specific rendering may differ. Public API semantics must not change silently.
Specify how named effort levels map to DeepSeek's numeric effort without changing Qwen behavior.

## Numerical and state checks

Keep the frozen Qwen fixtures and limits.
Use independent DeepSeek references derived from original quantized weight values, not inference outputs.
Freeze checkpoint, reference implementation, fixtures, full vocabulary vectors, and precision limits before acceptance.

Use layer-streamed FP64 reference computation for short full-model cases.
Use stored full endpoint vectors for restoration, media, and selected long-context cases.
Normal acceptance loads fixtures instead of generating new expected outputs.
Check target prefill, one decode step, draft outputs, verification, and restored continuation separately.
Compare TP4, TEP4, and DEP4 for the same semantic requests and approved numerical limits.

Exercise compressed-block boundaries, shared KV ownership, partial prefix branching, and each draft rejection position.
Check transfers that complete after cancellation and failure during rank communication.
Use Compute Sanitizer for changed native memory and synchronization paths.
Reuse existing service, cache, scheduler, constraint, execution, and transfer tests where possible.
Before a new test, state the failure it detects and why existing checks do not detect it.

## Capacity checks

The total context limit is 1048576, with input and output counted together.
Check a 1048512-token prompt with 64 output tokens at the limit.
Check rejection above the limit without silent truncation.

In Phase 5, exercise 64 such active requests with ordinary decode in each strategy after smaller cases pass.
In Phase 6, repeat this capacity check with DSpark in each strategy.

This is a capacity check, not an additional HTTP throughput matrix.
Record per-rank allocations and peak usage, pinned host bytes, queueing, and measured overlapping request activity externally.
Keep all 64 requests admitted while bounded chunks share execution.
Do not count 64 queued requests with only four admitted requests as capacity acceptance.
Include graph pools, expert buffers, incomplete compressor state, rollback storage, and host transfer leases in the budget.

## Results and delivery

Put approved DeepSeek HTTP and AgentX measurements in the separate [performance document](performance.md) and its full Chinese translation.
Do not put these measurements in README.
Keep raw logs, traces, responses, profiler reports, and unrelated test results external to the repository.

Split long result documents into linked bilingual parts before an English file exceeds 2000 words.
Keep result provenance with each table: source, executable, native modules, checkpoint, tokenizer, rank topology, and configuration identities.
Report latency, throughput, draft acceptance, failure counts, and memory with their units and definitions.
Do not reuse Qwen's `http-reference-v4` efficiency calculation for DeepSeek.
Any new efficiency model must count native formats, communication, replicated work, and host transfers with a reviewed version.
Measured throughput has no invented minimum acceptance threshold.

Shared runtime changes still use the applicable Qwen regression scope and its 24 HTTP cases before implementation delivery.
Keep existing README results unchanged as historical measurements.
Put new nightly Qwen regression measurements in separate bilingual result documents as well.

Get independent code and document review before each delivery.
Correct findings.
Keep hooks active.
Commit selected files.
Push `nightly`.
