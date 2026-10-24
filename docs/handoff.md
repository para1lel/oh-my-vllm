# Work state

Work continues on `nightly` without additional phase approval.
Qwen and DeepSeek are peer [model adapters](adapters.md).
Qwen serves inference on one B200. DeepSeek service inference is not enabled.

## Current work

Shared code supplies model descriptors, cache policies, typed state ownership, and rank control.
Model selection supplies defaults explicitly. Qwen transfer rules stay in its adapter.
Model selection and readiness checks share one test module.

The scheduler maximizes decode requests, whole prefill blocks, then prefill residual tokens.
It then minimizes requests with partial suffixes. Extra decode verification uses the unused budget.

The [DeepSeek plan](models/deepseek_v41/plan.md) tracks the incomplete model work.
Isolated CUDA operators, rank-owned weights, shared expert inputs, text rotary tables, and the shared pinned Engram loader are implemented.
Disabled rank-local blocks compose HC, attention, normalization, and MoE with explicit collective boundaries.
DEP4 MoE keeps gate and shared-expert computation local to each request owner.

HC consumes completed FP32 attention and MoE results without separate conversion kernels.
Compound numerical limits and complete allocation bounds must close before public inference is enabled.
Physical state, complete inference, DSpark, feature acceptance, and the approved measurement matrices are pending.

## Next actions

- Complete disabled DeepSeek target computation while numerical and allocation checks continue.
- Integrate physical state and DSpark, then complete capacity and feature acceptance.
- Run the approved DeepSeek HTTP and AgentX matrices after correctness acceptance.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md#delivery).
Keep host settings and process records in ignored `LOCAL.md`. Keep raw artifacts external.
