# HTTP API usage

## Start

Set the environment and model as specified in [development](../development.md).
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

Use the [acceptance index](../acceptance.md) for source identities and measured scope.

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

The shared HTTP service default adds presence_penalty `0.5`. Frequency penalty stays `0`, and repetition penalty stays `1`.
Ordinary, MTP4, DSpark, and test services use the same defaults.
The separate reasoning budget stays `-1` by default.
Use `serve --override-generation-config` with a JSON object to change supported sampling defaults.
Explicit request values take precedence over service values, which take precedence over checkpoint values.

Only the supported sampling keys are permitted. Invalid CLI values fail before model loading.
A request's null sampling field uses the default. An explicit `0` disables the default presence penalty.

Diagnostics report service defaults. Request metrics report the merged service and explicit request sampling fields.
Python adds checkpoint defaults for fields without explicit values.

The [sampling decision](../decisions/ADR-015-shared-serving-sampling-defaults.md) records the model guidance, selected value, and verification limits.

The shared `repetition_guard` default is `true`. Requests and service overrides accept a boolean.
An explicit `false` allows intentional repeated output. A request's null value uses the service default.

The guard limits repeated generated output. It respects grammar continuations and does not change committed history during speculative checks.
See [cycle detection](../../python/oh_my_vllm/worker/repetition.py) and [sampling integration](../../python/oh_my_vllm/worker/sampler.py) for the algorithm.

The guard cannot guarantee task completion or prevent cycles beyond these limits.
It also affects intentional repeated text when enabled. Keep client progress and repetition checks effective.

Offline Run/Bench explicitly disable the guard and use greedy sampling, fixed output length, and ignored EOS.

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
[Reasoning budget decision](../decisions/ADR-012-reasoning-token-budget.md) gives the source comparisons and speculative contract.

Chat separates `reasoning_content` from text and tool arguments.
Responses supplies readable reasoning events/items from the model's emitted reasoning.
It does not generate another summary.
Supported `chat_template_kwargs` are `preserve_thinking`, `enable_thinking`, and `reasoning_effort`.
Direct `enable_thinking=false` selects off. Other template options fail.

History tool arguments arrive as JSON strings and become objects before templating.
Responses reasoning maps back to assistant `reasoning_content`.
Top-level Responses instructions apply only to that response.
Continuation does not inherit these instructions.
