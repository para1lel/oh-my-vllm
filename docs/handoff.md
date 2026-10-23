# Work state

Use `nightly` for the planned [DeepSeek extension](deepseek-v41/design.md).
The service runs Qwen on one B200. DeepSeek inference is not enabled.
The architecture and measurement matrix have approval.
Execute the [phase plan](deepseek-v41/plan.md) without additional phase approval.

## Current state

Phase 1 [model adapters](deepseek-v41/adapters.md) and Phase 2 [distributed infrastructure](deepseek-v41/distributed.md) are implemented.
DeepSeek physical execution and state selection are pending.
Qwen keeps its single-worker path, with prefill dispatch, block scheduling, and host-cache startup fixes.

Phase 3 has isolated exact quantization, Engram hash, and shared pinned-host lookup with [fixed fixtures](deepseek-v41/fixtures.md).
Full checkpoint loading and model integration are pending.
The [reference tools](deepseek-v41/reference.md) separate FP64 mathematics, source-precision quantization, and released-source diagnostics.
Phase 0 compound numerical limits, long-context endpoints, and physical allocation bounds are pending.

## Next actions

- Generate the [long-context fixture](deepseek-v41/long-reference.md).
- Complete compound numerical limits and [allocation bounds](deepseek-v41/memory.md) before full-model integration.
- Continue the other operators and Engram model integration after their applicable gates close.
- Replace [Qwen results](deepseek-v41/qwen-results.md) only after the complete required HTTP regression.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md#delivery).
Keep host settings and process records in ignored `LOCAL.md`.
Keep raw artifacts external to the repository.
