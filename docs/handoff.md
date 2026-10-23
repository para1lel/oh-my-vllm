# Work state

Use `nightly` for the planned [DeepSeek extension](deepseek-v41/design.md).
The service runs Qwen on one B200. DeepSeek inference is not enabled.
The architecture and measurement matrix have approval.
Execute the [phase plan](deepseek-v41/plan.md) without additional phase approval.

## Current state

Phase 0 reference and acceptance work is active.
The [checkpoint tools](deepseek-v41/checkpoint.md) supply weight identities and a lazy CPU FP64 reader.
The [reference guide](deepseek-v41/reference.md) separates FP64 mathematics, source-precision quantization, and released-source capture.
The [fixed fixtures](deepseek-v41/fixtures.md) supply reviewed endpoints.
Released-source GPU capture and compound acceptance limits are pending.

[Long-context generation](deepseek-v41/long-reference.md) can resume from saved state.
The [allocation tools](deepseek-v41/memory.md) check design caps, not runtime capacity.

Phase 1 [model adapters](deepseek-v41/adapters.md) and Phase 2 infrastructure are completed.
The [typed execution interface](deepseek-v41/typed-execution.md) validates group completion before accepted-state publication.
Qwen keeps its single-worker path and cache coordinator.
DeepSeek state selection and physical model execution are subsequent work.

Qwen prefill dispatch, block scheduling, and host-cache startup changes are implemented.

## Next actions

- Generate the [long-context fixture](deepseek-v41/long-reference.md).
- Complete Phase 0 numerical limits and [allocation bounds](deepseek-v41/memory.md) before Phase 3 operators.
- Keep the approved [model](deepseek-v41/model.md), [state](deepseek-v41/state.md), and [sampling](deepseek-v41/sampling.md) contracts.
- Use the [test scope](testing.md) and [execution rules](execution.md) for subsequent implementation changes.
- Replace the [Qwen results](deepseek-v41/qwen-results.md) after the next full HTTP measurement.
- Obey the [delivery procedure](../CONTRIBUTING.md#delivery). Keep host settings and raw artifacts external to tracked documents.
