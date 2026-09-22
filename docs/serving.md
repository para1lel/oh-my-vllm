# OpenAI-compatible local serving

**Status (2026-09-21):** the independent Qwen Worker has passed real MTP4 JSON/tool
constraints, all thinking levels and lifecycle checks. Both oh-my-pi tasks still
need final independent-runtime acceptance. Historical V2 results remain in
[acceptance.md](acceptance.md); current progress is in [handoff.md](handoff.md).

## Run

```bash
scripts/with-env.sh cargo build --release
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-speculative-tokens 4 serve
```

Defaults: `127.0.0.1:8000`, model ID `qwen3.8-27b-fp8`, no authentication for the
local workflow. Set `serve --listen` and `--served-model-name` as needed. Rust
owns HTTP (Axum), protocol adaptation, response state, online admission,
cancellation, scheduling and KV. Python owns tokenizer/template application,
XGrammar state and masks, incremental detokenization, and project-owned GPU model execution.
No Python scheduler or tool executor is introduced. KV block size remains 784.

## Compatibility surface

- `GET /v1/models`.
- `POST /v1/chat/completions`: streaming SSE and non-streaming JSON, text and
  function tools, tool-result history, usage and finish reasons.
- `POST /v1/responses`: its own typed SSE events and JSON output items, text,
  function tools, full-history replay and `previous_response_id` continuation.
- `GET /v1/responses/{id}` and `DELETE /v1/responses/{id}`.
- Tool choice: auto, none, required, named function; multiple calls are allowed
  unless `parallel_tool_calls=false`. The service generates calls; OMP runs tools.
- Sampling: temperature, top_p, top_k, seed, frequency/presence/repetition penalties.
  Defaults come from the local generation_config.json (temperature 1, top_p .95,
  top_k 20). Offline Run/Bench retain greedy, fixed-length, ignore-EOS behavior.
- Default output budget: 8,192 tokens, including reasoning and text. The CLI
  context default is 65,536; prompt + output budget exceeding it is an error.
  No automatic truncation. EOS ends generation; up to four nonempty stop strings
  work for ordinary text, including across token/UTF-8 boundaries. Stops cannot
  interrupt enabled tools or structured output. Length exhaustion returns
  `length` / `incomplete`; an unfinished tool call is never delivered for execution.
- Text-only inputs, one completion per request. Unknown top-level fields and
  unsupported features such as logprobs, background mode and automatic truncation
  produce errors. This is a documented subset, not full OpenAI platform parity.

Protocol references: [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
[Responses SSE](https://developers.openai.com/api/reference/resources/responses/streaming-events),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The installed OMP 18.2.6 provider payloads and event consumers are also tested.

## Thinking and replay

Chat accepts `reasoning_effort`; Responses accepts `reasoning.effort`.
Default is medium. Native off/low/medium/xhigh map to the model template; high
explicitly aliases xhigh, and the OpenAI-style none aliases off. These are prompt
controls, not fixed reasoning budgets or a claim of monotonic quality/latency.
Unknown efforts fail. Thinking stays separate from content/JSON/tool arguments:
Chat uses reasoning_content; Responses uses readable reasoning summary events/items.
These contain the local model's emitted reasoning, not a separately generated summary.

OMP's validated chat_template_kwargs preserve_thinking/enable_thinking/reasoning_effort
extensions are supported; other template options fail. Direct enable_thinking=false
also selects off. History tool arguments arrive as JSON strings and become objects
before templating. Responses readable reasoning is mapped back to the assistant's
reasoning_content. Instructions supplied as a Responses top-level field apply only
to that response and are not inherited via previous_response_id.

## Constraints and MTP

JSON object, JSON Schema and tool argument grammars are applied **before sampling**.
Reasoning may precede the constrained answer. Response JSON and enabled tools cannot
be combined in one request. strict=true requires closed objects with every property
required. Unsupported schemas fail; there is no validate-and-retry substitute.

XGrammar builds Qwen-native XML tool grammars and JSON answer grammars. For MTP,
each scheduled draft position and the bonus position get the mask for their
speculative prefix. Simulated grammar advancement is rolled back; only accepted
output advances persistent state. An invalid draft is rejected at its first masked
position. This path does not turn off MTP or switch to vLLM's scheduler.
CPU tests cover mask rollback, rejected drafts, and reasoning-end crossings;
Real MTP4 runs additionally verify both APIs at off/medium with JSON object, JSON
Schema and strict tools; every completed case proposed and accepted actual drafts.

Supported schema keywords are explicitly listed in worker/serving.py; unsupported
backend features/combinations are rejected, including unknown string formats and
pattern/format combined with length bounds. Tool XML has extra encoding limits:

- Tool parameters require a direct object root (no root $ref/anyOf) and known,
  unambiguously typed properties. Explicit open additionalProperties is unsupported.
  Omitted additionalProperties in a non-strict tool is closed to known properties.
- Local property $ref and unambiguous anyOf are supported; mixed string/non-string
  XML unions are rejected because raw `null`/`123` cannot distinguish JSON types.
- XML strings with pattern, length or format constraints are rejected. String
  enum/const cannot contain the parameter closing delimiter or candidates that
  differ only in surrounding spaces/newlines/tabs; encoding padding is restored
  from the unique schema value. XML anyOf cannot have sibling const/enum. JSON answer schemas
  do not have these XML-specific restrictions.
- Unconstrained strings remove one leading/trailing newline from Qwen's XML
  template markup, matching the supported Qwen XML representation. Quotes, spaces, additional
  newlines and function/tool-close literals are preserved. Enum/const strings
  instead recover their exact schema value, including its own newlines.
  Parsers recognize closing tags only outside parameter values. A literal tool tag
  in ordinary text or a JSON answer is not interpreted as a tool invocation.

## State and resource limits

Responses defaults to store=true; OMP normally sends store=false. Stored responses
and their resolved conversation snapshots expire one hour after completion. Limits
are 1,000 records and 256 MiB of serialized response/history payload, configurable
with response-ttl-seconds, response-capacity and response-max-bytes. Object/allocation
overhead is additional to that serialized-byte budget. Oldest records are evicted
when capacity is reached; oversized records fail explicitly. Unknown/deleted/expired
IDs return 404. Restart invalidates IDs. Concurrent continuations copy a snapshot;
deleting an ancestor does not invalidate already-started requests or saved children.

HTTP bodies are limited to 8 MiB; admission channel and active request limit are
64 each; Rust schedules up to 32 concurrent sequences. Each stream has a 256-event
buffer. A disconnected or slow client cancels its request, not the engine.
The default request timeout is 600 seconds (configurable); preparation/step RPCs
have 120-second deadlines and detect worker exit. Fatal worker errors fail pending
requests; later submissions return unavailable until restart. Ctrl-C stops the
owned worker and closes streams. In-flight kernels are not individually preempted.

## OMP task and verification

Run the service with MTP enabled, then run both commands separately:

```bash
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --output-dir /tmp/oh-my-vllm-agentic-chat
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --output-dir /tmp/oh-my-vllm-agentic-responses
```

Also exercise the 12 real constrained-output cases (keep MTP enabled):

```bash
scripts/with-env.sh python scripts/serving-acceptance.py --output-dir /tmp/oh-my-vllm-serving-constraints
```

`scripts/serving-lifecycle.py --server-log /tmp/serve.log --output /tmp/serving-lifecycle.json` additionally
checks all five thinking levels, stored response retrieval/continuation/deletion,
concurrent plain/constrained requests, and a successful request after disconnect.
Start the service with `RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug` and
redirect its output to the supplied log file. The script requires log evidence
that both mixed requests shared a scheduled batch and the disconnected request
was canceled and released by the Worker RPC. It disconnects only after nonempty
generated SSE content. Use --base-url on all three scripts when the service uses
a non-default port. Check actual MTP activity in server logs, and stop the service
and its worker immediately after the runs.

The agentic script creates an isolated models.yml, allows read/grep/glob, and asks:

> Read this repository's README, architecture documentation and necessary source.
> Introduce the project goals, architecture, how to run it and its current
> completion status. Cite the files supporting your answer. Do not modify files.
> Read at least crates/scheduler/src/lib.rs and
> python/oh_my_vllm/worker/model_runner.py with the read tool before writing
> the final answer; directory listings alone do not count.

It saves client JSONL, stderr and exact command/configuration. Verify tool execution,
follow-up model requests with tool results, and a final answer grounded in files.
Verify successful reads of both named implementation files in the client JSONL.
Exit zero alone does not pass acceptance. The script kills its owned process group
on timeout. The dummy local API key is not a credential.

OMP uses native non-strict tool schemas because its strict normalization introduces
ambiguous nullable XML strings. Separate strict API tests cover supported schemas.
OMP minimal maps to off: the installed client's off control falls back to its
lowest supported effort. low→low, medium→medium, high/xhigh/max→xhigh. Both off and
medium client paths passed scripted protocol tests. Both real agentic API tasks
passed at medium, and all five thinking levels passed real API checks.

No new strict serving throughput threshold is set. Check real logs for queue wait,
preparation time, TTFT, end-to-end output tok/s, steps, cache hits and MTP proposed/
accepted tokens. Use DEBUG for RPC/worker host timing; do not label it CUDA kernel
time. Investigate persistent queueing, stalls, unexpected zero proposals/acceptance,
excess preemption, memory growth or worker errors. Compare like workloads after
warmup. Existing EngineCore >=95% requirements remain separate and unchanged.

The tokenizer uses Transformers with native Tokenizers DecodeStream for incremental
decoding. Vocabulary properties are obtained once during preparation, avoiding
repeated tokenizer metadata work per output token. Python DEBUG logs separate
message decoding, host execution and response encoding; those durations include
synchronization and are not CUDA kernel timings.
