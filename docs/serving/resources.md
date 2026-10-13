# Service resource contracts

## Storage and resource limits

Without an explicit value, Responses uses `store=true`. OMP usually sends `store=false`.
Stored responses and resolved history snapshots expire one hour after completion.
Defaults are 1000 records and 256 MiB of serialized response/history bytes.
Object/allocation overhead is additional.

Use `--response-ttl-seconds`, `--response-capacity`, and `--response-max-bytes` for changes.
Evict oldest records at capacity. Oversized records fail.
Unknown, deleted, or expired IDs return 404. An HTTP server restart invalidates stored IDs.
Worker replacement keeps stored histories but clears logical KV.

Concurrent continuations copy snapshots.
Ancestor deletion keeps already-started requests and stored children.

Reserve request capacity before the HTTP body is read or an ancestor history is copied.
Stored-response retrieval uses the same capacity reservation before its response copy.
The reservation stays with the engine, HTTP body, and completed stream drain until all owners release it.
Completed streams with slow readers thus keep a live request slot.

The encoded request state budget is 512 MiB by default. Use `--request-state-max-bytes` to change it.
This budget uses conservative multiples of serialized input, history, and output sizes.
JSON tree storage, allocator overhead, and Python objects use additional memory. Measure RSS independently of this budget.

| Resource | Default limit |
|---|---:|
| HTTP body | 8 MiB |
| Queued reservations / live reservations | 64 / 64 |
| Admission channel / active requests | 64 / 64 |
| Encoded request state reservations | 512 MiB |
| Scheduled sequences | 32 |
| Stream channel | 256 events |
| Pending stream queue | 1024 events or 8 MiB serialized bytes |
| Backpressure grace | 30 seconds |
| Request timeout | 3600 seconds |
| Prepare RPC deadline | 120 seconds |
| Worker step deadline | 600 seconds |
| Shared shutdown grace | 35 seconds |
| Worker replacement attempts | 5 |
| Worker memory snapshot interval | 30 seconds |

A full channel pauses its request and starts a monotonic backpressure deadline.
Pending events keep order, with final events.
Resume generation after the pending queue empties and at least 128 channel slots become free.
Sparse reads do not reset the deadline.
Disconnect marks cancellation immediately. The next engine check applies it.

The absolute request deadline starts before body reading. It includes queue wait, preparation, and generation.
Expired queued requests cannot start preparation.
Overload returns 503. A request that exceeds the logical KV capacity returns 413.

Set `--worker-step-timeout-seconds` for the maximum wait for one GPU step.
The positive value includes first-use compilation and graph capture.
A step timeout fails active requests and starts worker recovery.
Keep this deadline finite. Request and shutdown deadlines stay separate.

Deadline checks occur between operations. RPCs or blocked sends can make cancellation occur later.
Background CPU preparation itself does not block this check.
Paused requests keep KV when capacity is available. Priority preemption can still evict them.

Request-local validation or generation failures affect that request alone.
A GPU transaction that fails before completion prevents further use of the worker state.
CUDA OOM, device faults, worker exit, and invalid execution replies fail affected requests and start worker recovery.
The HTTP listener stays alive. New generation requests receive 503 during recovery.

The service stops the old worker and its owned process group before it loads a replacement.
It clears the scheduler and all logical cache identities for the old worker.
The replacement must report the same capacities. Its new epoch lets new admission start.

The default recovery limit is five launch attempts. Wait intervals are 1, 2, 4, 8, and 16 seconds.
For a larger `--worker-restart-attempts` value, each wait interval is at most 30 seconds.
After all recovery attempts fail, HTTP stays available and generation stays unavailable.
Streams report failure explicitly. The service does not automatically replay output or tool calls.

Ctrl-C and SIGTERM close streams and request worker shutdown.
HTTP drain and worker cleanup share the `--shutdown-grace-seconds` deadline, which the user can set.
Stalled readers disconnect at deadline. Kernels have no individual preemption.

An uncatchable Rust SIGKILL causes the parent-death mechanism to send SIGKILL to the direct GPU worker.
For whole-group cleanup after that signal, use an external process supervisor.
Normal shutdown and worker recovery have owned-group cleanup.

`GET /diagnostics` supplies Rust request, reservation, storage, and KV counts with the latest Python resource snapshot.
Use `--diagnostics-interval-seconds` to change the default 30-second snapshot interval.
The worker reports each request-state map, graph cache, compiler cache, host RSS, and GPU allocated/reserved bytes.
Per-request `SERVING_REQUEST_METRIC` logs use UTC identities and monotonic queue/processing durations.
Processing starts before Python preparation and ends when generation completes. HTTP drain has a different lifetime.

## Verification

Use `scripts/serving-acceptance.py`, `scripts/serving-lifecycle.py`, and `scripts/agentic-acceptance.py`.
Use OMP through `--api chat` and `--api responses` with the target checkpoint in each draft mode.
Use `--omp` or the executable on `PATH`.
Keep original tool calls, results, errors, and subsequent requests in client JSONL and forwarded HTTP records.

The recording proxy saves original SSE bytes, response hashes, IDs, and forwarded model requests.
Check SSE completion and usage against server metrics for the active mode.
Source-read outcomes and final answers are workload observations. The script reports its original task audit separately.
Keep the repository unchanged during the agent task.

Examine measured queue/preparation time, TTFT, rate, steps, prefix reuse, and verified/accepted drafts.
`verified_draft_tokens` counts scheduled candidates, with rejected candidates. Future proposals are a different counter.
Completion logs keep `proposed_draft_tokens` as a legacy alias for the scheduled-candidate count.
Use the active mode in `BENCH_CONFIG` and the explicit `--speculative-mode` acceptance option.

Do constraints, mixed-batch/cancellation, and long-context checks again for native MTP4 and DSpark with the two APIs.
There is no additional HTTP throughput gate.
Script exit status or a scripted worker alone does not show target-model acceptance.
See [testing](../testing.md) and [acceptance](../acceptance.md).
