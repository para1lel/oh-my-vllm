# HTTP service contracts

Rust controls Axum HTTP, admission, protocol adaptation, response storage, cancellation, scheduling, and logical KV.
Python controls tokenization, templates, XGrammar, detokenization, and GPU model execution.
The client executes function tools.

## Start

Set the environment and model as specified in [development](development.md).
Use a unique IPC path for each service:

```bash
scripts/with-env.sh cargo build --release
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-speculative-tokens 4 serve
```

Defaults are `127.0.0.1:8000` and model ID `qwen3.8-27b-fp8`.
The local service has no authentication.
Use `serve --listen` and `--served-model-name` to change these values.
Keep server addresses in `LOCAL.md`.

For DSpark, set `OH_MY_VLLM_DRAFT_MODEL` and select the mode explicitly:

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-dspark-serve.ipc --speculative-mode dspark serve
```

One worker serves one draft mode. HTTP requests cannot change that mode.
Native MTP4 stays available through the legacy four-token option or `--speculative-mode mtp`.
DSpark has at most seven drafts, with the same target checkpoint, block 784, APIs, and request constraints.

The initial `--dspark-confidence-threshold` setting is `0.2`. It limits cumulative confidence from the first candidate.
It can return zero through seven candidates. Zero uses one target token for that step.
Use `0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.

Use the [acceptance index](acceptance.md) for source identities and measured scope.

## API surface

| Method and route | Contract |
|---|---|
| `GET /v1/models` | Model discovery. |
| `POST /v1/chat/completions` | JSON or SSE, text, function tools, tool history, usage, and finish reasons. |
| `POST /v1/responses` | JSON items or typed SSE, text, tools, full history, and `previous_response_id`. |
| `GET /v1/responses/{id}` | Stored response retrieval. |
| `DELETE /v1/responses/{id}` | Stored response deletion. |
| `GET /diagnostics` | Service health, resource reservations, cache counts, and the latest worker memory snapshot. |

Inputs are text-only with one completion per request.
Unknown fields and unsupported features, with logprobs, background mode, and automatic truncation, produce errors.
Supported OMP provider payloads and event consumers have tests.
The API subset follows [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and [Responses](https://developers.openai.com/api/reference/resources/responses/methods/create) conventions.

## Sampling, tools, and stopping

Sampling supplies temperature, top_p, top_k, seed, and frequency/presence/repetition penalties.
Checkpoint defaults are temperature `1`, top_p `.95`, and top_k `20`.
Offline Run/Bench use greedy sampling, fixed output length, and ignore EOS.

Tool choice can be auto, none, required, or a named function.
Multiple calls are permitted unless `parallel_tool_calls=false`.
The service generates tool calls. OMP executes them and returns results.

The default output budget is 8192 tokens, with reasoning and text.
The default CLI context is 65536 tokens.
Prompt plus output budget more than the context limit is an error.
EOS ends generation.
Up to four nonempty stop strings can cross token and UTF-8 boundaries in ordinary text.

Stops cannot interrupt enabled tools or structured output.
Length exhaustion returns `length` or `incomplete`.
The service withholds unfinished tool calls from execution.

## Thinking and replay

Chat uses `reasoning_effort`. Responses uses `reasoning.effort`.
The default is medium.
Native off/low/medium/xhigh control the template. High aliases xhigh and none aliases off.

Unknown efforts fail.
These prompt controls do not specify fixed token budgets or monotonic quality/latency.

The two APIs accept top-level `thinking_token_budget`.
Use a nonnegative integer to limit generated reasoning content tokens. Use `-1` for no separate limit.
The service option `--thinking-token-budget` supplies the default when a request omits this field.
Its default is `-1`, as in vLLM's default behavior.

A request value takes precedence. Effort does not select the numeric budget.
After the budget, the target mask lets the sampler select only `</think>`, then answer or tool generation continues.
A zero budget forces this marker immediately when thinking is enabled. Off does not produce a reasoning block.

The closing marker uses one output token. Existing reasoning usage includes this token, so it can be `budget + 1`.
The total output and context limits stay effective. They can stop generation before the closing marker or answer.
The budget does not limit answer tokens or detect repeated answer text.
Responses output and service diagnostics report the effective budget.

OMP's installed OpenAI adapters omit this field. Use the service option for the two APIs and automatic compaction.
Do not infer a numeric budget from OMP `max` or service `xhigh`.
[Reasoning budget decision](decisions/ADR-012-reasoning-token-budget.md) gives the source comparisons and speculative contract.

Chat separates `reasoning_content` from text and tool arguments.
Responses supplies readable reasoning events/items from the model's emitted reasoning.
It does not generate another summary.
Supported `chat_template_kwargs` are `preserve_thinking`, `enable_thinking`, and `reasoning_effort`.
Direct `enable_thinking=false` selects off. Other template options fail.

History tool arguments arrive as JSON strings and become objects before templating.
Responses reasoning maps back to assistant `reasoning_content`.
Top-level Responses instructions apply only to that response.
Continuation does not inherit these instructions.

## Grammar and speculative tokens

JSON object, JSON Schema, and strict tool grammars apply before sampling.
The constrained answer can follow reasoning.
A request cannot combine JSON answer format with enabled tools.
Strict schemas must have closed objects with each property required.
Unsupported schemas fail explicitly.

The supported keyword inventory is in `python/oh_my_vllm/worker/serving.py`.
Unknown string formats and pattern/format combined with length bounds are unsupported.

XGrammar supplies Qwen XML tool grammars and JSON answer grammars.
Each MTP or DSpark draft and bonus position uses its speculative-prefix mask.
Simulated grammar advancement rolls back. Kept output alone advances persistent state.
A draft that the mask rejects fails at its first invalid position.

CPU tests include rollback, rejection, and reasoning-end crossings.
Target-model MTP4 evidence includes the two APIs, off/medium, JSON object, JSON Schema, and strict tools.

DSpark proposals use the same masks before candidate selection.
For stochastic output, acceptance uses full conditional target/proposal probabilities with the configured sampling parameters.
Candidate acceptance is `min(1,p(x)/q(x))`. Rejection uses normalized `max(p-q,0)`, and the bonus uses target distribution `p`.
Verification uses target tokens with maximum scores for greedy sampling.
See [architecture](architecture.md#dspark) for context and state commit.

## XML tool restrictions

Tool parameters must have a direct object root, without root `$ref` or `anyOf`.
Properties must have known, unambiguous types.
Explicit open `additionalProperties` is unsupported.
Omission in a non-strict tool closes the object to known properties.
Local property `$ref` and unambiguous `anyOf` are permitted.

Mixed string/nonstring unions are ambiguous in XML and fail.

XML strings cannot use pattern, length, or format constraints.
String enum/const cannot contain the parameter closing delimiter.
Candidates cannot differ only in surrounding spaces, newlines, or tabs.
The parser restores padding from the unique schema value.
XML `anyOf` cannot have sibling const/enum.

JSON answer schemas have no XML-specific restrictions.

Unconstrained strings remove one leading/trailing newline from template markup.
Quotes, spaces, extra newlines, and function/tool-close literals stay.
Enum/const strings restore the same schema value, with its newlines.
Parsers recognize closing tags only when they are not in parameter values.
Literal tool tags in ordinary text or JSON answers stay text.

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
| Request timeout | 600 seconds |
| Prepare RPC deadline | 120 seconds |
| Worker step deadline | 120 seconds |
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
Complete OMP tasks through `--api chat` and `--api responses` with the target checkpoint in each draft mode.
Use `--omp` or the executable on `PATH`.
The client JSONL must show successful `read` calls for `crates/scheduler/src/lib.rs` and `python/oh_my_vllm/worker/model_runner.py`.
Directory listings alone are insufficient.
The model must receive the tool results.

The recording proxy keeps the forwarded model request that contains each source-read result.
Client response IDs for the last answer must agree with the recorded HTTP responses and active-mode server logs.

The answer must use those results and refer to the files.
Keep the repository unchanged during the agent task.

Examine measured queue/preparation time, TTFT, rate, steps, prefix reuse, and verified/accepted drafts.
`verified_draft_tokens` counts scheduled candidates, with rejected candidates. Future proposals are a different counter.
Completion logs keep `proposed_draft_tokens` as a legacy alias for the scheduled-candidate count.
Use the active mode in `BENCH_CONFIG` and the explicit `--speculative-mode` acceptance option.

Do constraints, mixed-batch/cancellation, and long-context checks again for native MTP4 and DSpark with the two APIs.
There is no additional HTTP throughput gate.
Script exit status or a scripted worker alone does not show target-model acceptance.
See [testing](testing.md) and [acceptance](acceptance.md).
