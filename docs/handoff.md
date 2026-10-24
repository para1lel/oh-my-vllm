# Work state

Work continues on `nightly` without additional phase approval.
Qwen and DeepSeek are peer [model adapters](adapters.md).
Qwen serves inference on one B200. DeepSeek service inference is not enabled.

## Current work

Shared code supplies model descriptors, cache policies, typed state ownership, and rank control.
The scheduler update follows decode-count, whole-block, residual-token, then partial-suffix priorities.
Adapter cleanup removes implicit Qwen defaults and obsolete migration descriptions.
Qwen numerical and HTTP validation must bind the final runtime candidate before delivery.

The [DeepSeek plan](models/deepseek_v41/plan.md) tracks the incomplete model work.
Isolated CUDA operators, rank-owned weights, grouped projections, and the shared pinned Engram loader are implemented.
Compound numerical limits and complete allocation bounds must close before full-model integration.
Physical state, complete inference, DSpark, feature acceptance, and the approved measurement matrices are pending.

## Next actions

- Complete scheduler and adapter cleanup, independent review, and applicable Qwen regression.
- Close DeepSeek numerical and allocation gates, then continue model integration.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md#delivery).
Keep host settings and process records in ignored `LOCAL.md`. Keep raw artifacts external.
