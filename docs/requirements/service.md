# Service contracts

## Service requirements

| ID | Contract |
|---|---|
| REQ-SERVE-001 | Supply text Chat Completions and Responses, streams, tools, history, usage, model discovery, retrieval, and deletion. |
| REQ-SERVE-002 | Supply off/low/medium/high/xhigh thinking. Default medium, high maps to xhigh, none maps to off. |
| REQ-SERVE-010 | Apply explicit reasoning token budgets before target sampling in ordinary, MTP4, and DSpark modes. Keep effort separate from numeric and total limits. |
| REQ-SERVE-003 | Use oh-my-pi through each API with MTP4 and DSpark. Verify model requests, responses, and returned tool results in subsequent input. |
| REQ-SERVE-004 | Apply JSON/Schema/strict-tool masks before sampling, with each mode's drafts and bonus tokens. Reject unsupported schemas. |
| REQ-SERVE-005 | Examine queue time, preparation, TTFT, output rate, cache behavior, and draft counters without an extra HTTP throughput gate. |
| REQ-SERVE-011 | Use shared production and test service sampling defaults across ordinary, MTP4, DSpark, the two APIs, and automatic compaction. Explicit values take precedence. |
| REQ-SERVE-009 | Apply a bounded generated-token cycle guard in all service modes. Keep grammar continuations, speculative prefix correctness, and explicit disable available. |

[Service contracts](../serving.md) define supported parameters, XML restrictions, storage limits, and errors.
Scripted workers alone cannot show target-model acceptance.

Thinking grammar and budget accounting must use the same closing token identity.
Enabled tool constraints must start at the opener recognized by the output parser.

Keep original tool descriptions and parameter schemas in the model prompt.
For tools without explicit `strict=true`, accept descriptive keywords through a separate copy of the supported XML schema.
OMP checks descriptive argument limits and returns errors to the model. Keep explicit strict schemas and inference computation unchanged.

For each draft mode, run tests for the two APIs, constraints, cancellation, mixed batches, and long-context prefix reuse.
OMP evidence keeps model requests, responses, tool calls, returned results, and subsequent requests.
Task progress, tool errors, goal status, and final answers are workload observations. They do not decide framework acceptance.
Compare response IDs with server logs for the active mode and scheduled drafts.

## REQ-SERVE-006: Resource limits and cache integrity

Keep request admission, queues, history copies, output buffers, and completed stream drains at or below explicit count and byte limits.
Check and take capacity before history copies or request preparation.
The request deadline starts at admission and includes queue time.
Reject temporary overload with a retryable error. Reject a request that exceeds the supported capacity.

Report failures explicitly. This includes failures after stream output starts.

Publish prefix identities only after a validated GPU result commits the corresponding state.
Keep referenced blocks safe during LRU eviction. Clear uncommitted identities after failure or cancellation.
Logical KV eviction reuses the fixed GPU arena. It does not release that arena's HBM storage.
Keep graph entries, compiler metadata, response records, and shutdown references at or below their specified resource limits.

## REQ-SERVE-007: Worker recovery

Keep the HTTP process alive after a GPU OOM or worker failure.
If device state cannot be reused safely, fail the affected requests and rebuild the worker.
Clear all logical cache identities for that worker before new admission.
Report temporary unavailability during recovery. Keep recovery attempts and shutdown waits at their specified limits.
Do not replay requests that already supplied stream output.

Run tests for OOM, partial state changes, shared prefixes, and successful service after recovery.

## REQ-SERVE-008: Long-duration agentic service

Run MTP4 and DSpark independently for at least one hour each. Four fixed oh-my-pi sessions supply each mode's request workload.
The hour verifies inference service availability, serving contracts, resource limits, and cleanup.
Keep each process, session ID, and conversation history through follow-up tasks and automatic compaction.
Use two Chat sessions and two Responses sessions. Limit concurrent model calls to four, with compaction calls.

Report measured request counts. No minimum request count applies.

Start the clock after service readiness and client initialization. Include first-request compilation and graph capture.
Use maximum reasoning effort for tasks and compaction. OMP `max` maps to service `xhigh`.
Supply `262144` as the model context window. Include reasoning tokens in the output budget.

OMP uses runtime provider usage and its conversation estimate to decide automatic compaction.
Independent token checks verify the input and output capacity. They do not select compaction times.
Report natural compaction coverage and subsequent requests in the same session. Summary content and task results are workload observations.

Each session translates the same validated English book in a separate sandbox, without a copy of inference source.
Supply the external corpus directory, which holds the volumes and their original manifest.
Use OMP native goals and automatic continuation. Keep a prepared sequence of volume, passage and verification goals.
Submit the next goal after completion, pause, or stopped continuation. These changes do not fail acceptance.

Keep the same process, session ID, history, and workspace throughout the hour.
Then pause continuation, abort active agent work through native RPC, and drain model requests.
Record the stop action and report the tail separately.

Delete sandbox containers, modified files, installed tools, and caches after completion, failure, or cancellation.
Keep original test logs and source hashes in a separate evidence directory.
Check resource limits throughout collection and released request state after the final drain.

Keep full host RSS and GPU allocated/reserved memory curves. Examine growth against context length, active requests, and cache occupancy.
Independent memory review must accept these observations before publication as a passed soak test.
The framework must have no unexpected OOM, worker restart, or failed model execution.
Report client cancellations and configured request deadlines separately. They do not imply successful request completion.

Run cancellation, slow-client, overload, and fault tests separately.

Record other GPU processes. Continue collection if request and resource checks pass.
Phase latency tests must use a GPU with no other processes.

Publish one README scatter plot per mode. Include task requests and measured compaction requests, with their purpose identified.
The horizontal axis is framework processing start, when request preparation starts.
The vertical axis is output tokens divided by processing-start-to-generation-completion wall time.

Include reasoning tokens. Report queue time, cancelled requests, and failed requests separately.
Do not use API color legends. This test has no performance gate. Daily test commands skip it.
Measure affected phase workloads again after inference-path changes.
