# DeepSeek V4.1 implementation plan

## Goal and status

Add DeepSeek inference on four B200 GPUs with all applicable Qwen functions.
The [architecture](design.md) and [acceptance scope](validation.md) have approval.
Work continues on `nightly` without additional phase approval.
DeepSeek service inference is not enabled.

Phases 1 and 2 are implemented.
Phase 0 reference and allocation work continues.
Phase 3 has isolated exact quantization and Engram hash operators.
Full-model computation and Phases 4-8 are pending.

## Constraints

- Use GPT-6.1 Sol with `high` effort for agents.
- Exclude physical GPU 0.
- Keep Qwen ordinary decode, MTP4, 262144 total tokens, and its single-B200 configuration.
- Add TP4, TEP4, DEP4, DSpark with at most five drafts, 64 admitted requests, and 1048576 total tokens per request.
- Keep Engram weights in one shared pinned-host store.
- Keep Rust scheduling and state ownership, Python GPU control, ZMQ DEALER, and msgpack.
- Use native CUDA operators without external operator libraries or vLLM dependencies.
- Apply the [Eager/Graph execution rules](../execution.md) to each complete Execute message.
- Follow the [delivery procedure](../../CONTRIBUTING.md) and [reporting policy](validation.md#results-and-delivery).

## Phase gates

Each operator must have frozen mathematics, fixtures, and numerical limits before implementation.
Isolated exact conversions and integer hashes can proceed while compound limits and long references are pending.
This exception does not permit performance tuning without frozen latency limits.
It does not permit full-model integration before the applicable Phase 0 gates close.

Do not modify sources or environments used by active reference generators.
Keep existing numerical fixtures and limits unchanged during candidate acceptance.
Before each new test, tell the maintainer why existing checks are not sufficient.
Use existing tests when they cover the required behavior.

Each delivery must include independent review, corrected findings, applicable hooks, selected-file commits, and a push to `nightly`.
Keep process and GPU-lock ownership explicit.
Stop temporary processes after their work.
Keep only current instructions and the latest complete approved measurements.

## Phase 0: Reference and allocation contracts

Completed prerequisites:

- Check all 48 shards, tensor layouts, scales, and source identities.
- Define [model mathematics](model.md), [state ownership](state.md), [sampling](sampling.md), and [media encoding](video.md).
- Store independent [short, DSpark, media, codec, and exact-operator fixtures](fixtures.md).
- Implement layer-streamed FP64 reference generation and the separate released-source capture tool.
- Define the three parallel strategies and the exact measurement matrix.

Next work:

- Complete and freeze the [long-context reference](long-reference.md).
- Freeze compound numerical limits and operator latency limits before tuning.
- Close the [allocation plan's open bounds](memory.md#6-allocation-caps-and-open-bounds) for 64 requests at the context limit.

Gate: No unresolved weight mapping or acceptance rule can affect an operator entering Phase 3.
Existing Qwen references do not cover these formats or state equations.

## Phase 1: Model adapters

Implemented [adapters](adapters.md) select descriptors, cache schemas, parallel plans, speculation, codecs, and model execution.
Qwen-specific dimensions and policies reside in its model package.
Unknown architectures, unsupported formats, and capability mismatches cause errors.

Gate: Qwen keeps its API behavior, accepted outputs, and state semantics through the adapter.
Keep existing Rust, service, constraint, execution, and frozen numerical checks.

## Phase 2: Typed state and rank control

Implemented [distributed infrastructure](distributed.md) supplies typed transactions, group epochs, readiness, completion, deadlines, and shutdown.
GPU selection acquires all required locks or releases them before waiting.
Collective plans specify order and buffer lifetime for Eager and Graph execution.

Gate: Synthetic four-rank execution cannot publish partial state or leak processes and locks.
Checks cover empty ranks, uneven schedules, stale results, startup failure, and collective failure.
Physical DeepSeek state selection is subsequent work.

## Phase 3: Weights and operators

- Implement FP8/FP4 projections, routing, hyper-connections, compression, and sparse attention with fixed fixtures.
- Load only rank-owned slices, without full FP16 or FP32 expansion.
- Implement Engram lookup through shared pinned storage and bounded staging.
- Separate Engram storage from host KV accounting.
- Check registration, NUMA placement, memory pressure, cancellation, and transfer ownership.
- Run numerical checks, fixed latency checks, and Compute Sanitizer for changed memory paths.

Use the existing API/plan/workspace structure.
Weight plans consume the checkpoint identity and parallel strategy.
Gate: Operators match their references and fit startup budgets.

## Phase 4: Ordinary inference

Implement full short-context target computation before decoder-work reduction.
Add TP4, then TEP4, then DEP4.
Bind shared experts, heads, vision, and DSpark weights to the approved placement plan.

Check uneven expert loads, empty ranks, top-k ties, and request reordering.
Compare all 129280 endpoint logits after Eager prefill and Graph decode.
Gate: All three strategies serve ordinary text through offline and HTTP interfaces.

## Phase 5: Context and capacity

Implement bounded prefill chunks, index workspaces, shared compressed KV, and private speculative state.
Add prefix sharing, pressure-driven offload, exact restoration, and partial-prefix continuation.
Validate reduced decoder work against full computation.
Approximate restoration must have separate approval and numerical limits.

Check 400000, 800000, and 1048576 total-token boundaries with output capacity reserved.
Exercise 64 admitted requests with ordinary decode in each strategy.
Check cancellation during memory pressure and repeated eviction/restoration.
Gate: Measured allocations agree with accounting, and the ordinary capacity cases complete.

## Phase 6: DSpark

Implement three prediction layers with at most five proposals and six target verification rows.
Start with fixed-length verification.
Add confidence scheduling after fixed-length acceptance passes.
Keep exact probability correction.

Check each rejection position, all-accept, EOS, stops, limits, schema masks, and restored transfers.
Check RNG and penalty history during partial verification and after batch reordering.
Separate committed lengths from new proposals.
Use accepted target features and state for continuation.
Repeat the 64-request capacity case with DSpark in each strategy.
Gate: Sampling and state contracts hold for all three strategies.

## Phase 7: API and media parity

Complete the [function checklist](validation.md#function-checklist) through model-selected behavior.
Check tools, reasoning, continuation, structured output, and ordered image/video input against fixed codecs.
Keep public request/response semantics and current restrictions.

Exercise HTTP streaming, offline execution, cancellation, and worker replacement.
Gate: Each advertised Qwen-equivalent function works for DeepSeek.

## Phase 8: Measurement and delivery

Extend the existing benchmark drivers for DeepSeek profiles and four-rank identities.
Check exact case enumeration, prompt/output lengths, and full response-ID joins.
Run the [38 HTTP cases and three AgentX cases](validation.md) after correctness gates pass.
Keep Qwen's 24-case regression separate.

Replace the existing result documents after each complete matrix.
Bind measurements to source, binaries, weights, tokenizer, topology, and configuration.
Report measured latency, throughput, acceptance, memory, and failures.
Do not reuse Qwen efficiency values or invent throughput thresholds.
Gate: Reviewed results and resource cleanup apply to the delivered source.
