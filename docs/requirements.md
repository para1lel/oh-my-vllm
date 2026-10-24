# Runtime contracts

Rust owns the service, admission, scheduling, logical cache, and validated state commit.
Python owns model computation and physical tensors.
Use ZMQ DEALER and msgpack between processes.
Use separate dependency files for development and inference.

Model adapters have equal status.
Each adapter supplies configuration, capabilities, token codecs, computation, cache policy, and speculation rules.
Do not use one model's dimensions or syntax as defaults for another model.
See the [adapter contract](adapters.md), [Qwen adapter](models/qwen38/README.md), and [DeepSeek adapter](models/deepseek_v41/README.md).

Use CUDA C++ or CuTe DSL for model operators.
Native CUDA libraries can supply infrastructure and computation.
Do not add external operator libraries, such as DeepGEMM or FlashInfer, or vLLM dependencies.
Operator interfaces must permit beneficial fusion and PDL.
Keep model-specific numerical and state rules during optimization.

Use the [GPU execution rules](execution.md) for scheduling and resource ownership.
An Execute with prefill uses Eager execution for all its fragments.
A decode-only Execute uses CUDA Graph recording and replay.
The scheduler maximizes decode requests, then complete prefill blocks, then residual tokens.
After these priorities, it minimizes requests with partial suffixes.

## Service functions

Adapters implement these functions through the shared service.
Advertise a function only after its model-specific acceptance is complete.

| Area | Contract |
|---|---|
| HTTP | Model listing, Chat JSON/SSE, Responses JSON/typed SSE, diagnostics |
| Stored responses | Retrieval, deletion, continuation, storage limits, `store=false` |
| Offline | `run`, `bench`, token/prompt files, arrival control, prefix reuse, fixed output lengths |
| Sampling | Temperature, top-k, top-p, seed, frequency/presence/repetition penalties, cycle guard |
| Stops | EOS, `ignore_eos`, stop strings, minimum output, maximum output, context limit |
| Reasoning | Effort selection, thinking budget, incremental parsing, usage accounting |
| Tools | Auto/none/required/named choice, parallel-call control, history, streamed arguments |
| Structured output | JSON object/schema, token masks, independent speculative grammar state |
| Media | Ordered text/images/video, limits, frame timestamps, history, prefix isolation |
| Cache | Full and partial prefixes, copy-on-write, preemption, host offload, restoration, eviction |
| Lifecycle | Deadlines, disconnect cancellation, backpressure, worker failure/replacement, graceful shutdown |

There is one completion per request.
Logprobs, background execution, and automatic truncation are not supported.
Enabled tools and structured-output mode cannot be combined.
Model-specific rendering cannot change public API semantics silently.

Use the [test and reporting scope](testing.md) and each model's fixed numerical limits.
The [contribution procedure](../CONTRIBUTING.md) controls delivery.
