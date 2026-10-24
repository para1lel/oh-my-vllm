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
Isolated CUDA operators, rank-owned weights, text rotary tables, and shared pinned Engram storage are implemented.
Disabled blocks compose HC, attention, normalization, and MoE through prepared TP4, TEP4, and DEP4 collectives.
DEP4 MoE keeps gate and shared-expert computation local to each request owner.
Engram composition connects host lookup, projection, and residual blending.
Source compressors connect projection, caller-supplied complete groups, normalization, RoPE, and FP4 encoding.

HC consumes completed FP32 attention and MoE results without separate conversion kernels.
Compound numerical limits and complete allocation bounds must close before public inference is enabled.
Physical state, complete inference, DSpark, feature acceptance, and the approved measurement matrices are pending.

## Next actions

- Resolve the memory-check failure when host-table registration follows prior GPU computation.
- Complete disabled DeepSeek target computation with shared temporary storage while numerical and allocation checks continue.
- Integrate physical state and DSpark, then complete capacity and feature acceptance.
- Run the approved DeepSeek HTTP and AgentX matrices after correctness acceptance.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md#delivery).
Keep host settings and process records in ignored `LOCAL.md`. Keep raw artifacts external.
