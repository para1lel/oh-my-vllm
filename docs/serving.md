# OpenAI-compatible serving — discussion record

**Date:** 2026-09-21
**Status:** User-confirmed scope; implementation and acceptance not started.
This task records the discussion only; it does not implement or launch a service.

## Confirmed decisions

- Run the service and oh-my-pi on the current machine; run oh-my-pi in this repo.
- Implement both `/v1/chat/completions` and `/v1/responses`, streaming and
  non-streaming, plus `/v1/models`. Support text, function tools, tool-result
  history, usage, termination and errors using each API's own representation.
- Prefer Rust for HTTP entry points, protocol adaptation and request lifecycle.
  Rust keeps scheduling and KV ownership; Python wraps GPUWorker. Existing
  model, single-B200 selection, block784 and hybrid-cache constraints remain.
- Support configurable thinking strength through both APIs.
- Support Responses full-history replay and `store`/`previous_response_id`,
  response retrieval and deletion. Use bounded, expiring memory. Responses need
  not survive restart; exact capacities and expiration durations remain open.
- Test both APIs using oh-my-pi, and cover non-streaming, cancellation, errors
  and thinking parameter mapping with automated tests.

## Agentic acceptance

For each API separately, ask oh-my-pi:

> Read this repository's README, architecture documentation and necessary source.
> Introduce the project goals, architecture, how to run it and its current
> completion status. Cite the files supporting your answer. Do not modify files.

Observe actual tool calls executed by oh-my-pi, tool results returned in a later
model request, and a final answer grounded in the repository. A plausible answer
without tool execution is not sufficient. The inference service generates tool
calls; it does not execute repository tools itself. Save reproducible test
configuration and evidence without credentials or profiling traces.

## Observed implementation gaps

The existing binary has Run/Bench entry points, not HTTP serving. Its stepwise
token outputs, continuous batching and between-step cancellation are reusable.
Online admission and per-request delivery channels are still needed.

`crates/scheduler/src/request.rs` currently completes by max token count, and
`python/oh_my_vllm/worker/model_runner.py` defaults to greedy, ignore-EOS sampling.
Serving needs proper EOS/stop and length semantics. ZMQ registration currently
carries prompt IDs but not per-request sampling parameters. Preserve fixed-length
benchmark behavior while introducing serving behavior deliberately.

No serving chat template adapter, incremental text decoder, tool/reasoning
parser or SSE serializer exists. Client disconnects and failures must connect to
the existing cancellation/Worker cleanup lifecycle; slow consumers need a policy.
No serving performance or agentic success has been measured. Existing EngineCore
acceptance is separate and must not be presented as HTTP performance evidence.

## Local model and client facts (read-only inspection)

The model's `/data0/shared/Qwen3.8-27B-FP8/chat_template.jinja` supports tools and
tool history. Generated calls use XML-like `<tool_call><function=...>` syntax,
not OpenAI JSON. Historical function arguments must become dictionaries before
templating. The service must preserve tool call identifiers across turns.

Native thinking behavior observed in that template:

| Mode | Template behavior |
|---|---|
| off | `enable_thinking=false`, inserts an empty closed thinking section |
| low | Enables thinking and adds a brief-thinking instruction |
| medium | Enables thinking without an extra effort instruction |
| xhigh | Enables thinking and adds a more thorough-thinking instruction |
| high | Rejected when thinking is enabled |

The template defaults to thinking enabled with xhigh. These are prompt-level
controls, not hard token budgets; no monotonic quality/latency claim is supported.
The service default and external aliases are not decided. In particular, do not
silently accept high as an independent native level. The installed vLLM
`vllm/parser/qwen3.py` has a potential reusable reasoning/XML-tool parser; using
it would not require vLLM's scheduler. Reuse versus Rust parsing remains open.

The inspected `/home/dongwu.chen/.local/bin/omp` reports v18.2.6. Its providers
support `openai-completions` and `openai-responses` with a custom base URL.
Chat streaming consumes content/tool-call deltas and requires a finish reason.
Responses consumes typed SSE events with item IDs, output indices and call IDs;
it defaults to `store:false` and can also use stored continuation. A Chat SSE
serializer cannot be reused unchanged as a Responses serializer.

Responses thinking uses `reasoning.effort`; OMP supports effort maps. Its UI levels
are broader than the native model levels, so configuration and service validation
must agree. Installed/legacy state directories differ: verify the active config
or use an explicit isolated state directory when testing. No client config was
changed during this discussion.

## Remaining design choices (not user-confirmed)

- External thinking levels, aliases, default, disable semantics and how reasoning
  is exposed/replayed by each API. Native prompt differences need tests; actual
  model behavior still needs GPU validation.
- Detailed compatibility matrix: tool choice/parallel calls, strict schemas,
  structured output, supported sampling fields, and handling unsupported fields.
  Both APIs are required; exhaustive OpenAI platform parity was not agreed.
- Responses defaults, limits/TTL, unknown/expired ID errors, concurrent access,
  deletion semantics and any endpoints beyond creation/retrieval/deletion.
- HTTP library, port, listen address, authentication, request limits, timeout,
  queue/backpressure policy and exact tokenizer/template/parser placement.
- Context/output limits, default sampling, stop handling, and any additional
  serving performance gate. The existing EngineCore target remains unchanged.

A shared internal request/event model is recommended so both API adapters use
the same engine, cancellation and output parsing. This is a proposal, not code.

## Planned verification (not run)

- Both real oh-my-pi provider paths complete the repository-reading task.
- Both streaming and non-streaming represent the same text/tool outcomes;
  fragmented tool arguments, UTF-8 boundaries, final events and usage are tested.
- Full-history and stored Responses chains preserve tool-result associations;
  retrieval, deletion, expiry, capacity and restart invalidation are tested.
- EOS, output limits, stop strings, disconnects, malformed requests and worker
  failures terminate correctly and release resources.
- Thinking configuration produces the intended template and protocol fields;
  invalid levels are handled according to the eventual explicit mapping.

Future implementation should pin compatibility to the official
[Chat Completions reference](https://platform.openai.com/docs/api-reference/chat)
and [Responses reference](https://platform.openai.com/docs/api-reference/responses),
and verify against the installed client rather than assuming protocol equivalence.
