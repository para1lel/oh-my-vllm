# ADR-009: Optional DSpark mode

Status: active. The [acceptance index](../acceptance.md) gives measured source identities and scope.

## Context

Native MTP4 supplies four draft tokens through the target model's MTP layer.
Use the local DSpark checkpoint.
The two HTTP APIs must complete OMP tool tasks with the target checkpoint in this mode.
All modes use the phase-latency, context, accuracy, and service gates.

DSpark uses five BF16 GQA layers, zero-based target layer outputs `(5,19,33,47,61)`, and Markov/confidence heads.
It proposes at most seven tokens for eight-row target verification.
Its draft block 7 is different from target block 784.

## Primary references

The [DSpark paper](https://arxiv.org/abs/2607.05147) (v1, 2026-07-06) and [DeepSpec](https://github.com/deepseek-ai/DeepSpec) supply algorithm references.
The [Qwen3.8-27B-DSpark model card](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) gives checkpoint information.
Project support uses the loaded local configuration/weight SHA-256 identities.
Their measurements use their hardware/runtime configurations. Project acceptance must use its own B200 records from the current source.

## Decision

Add explicit worker modes `none`, `mtp`, and `dspark`.
Keep the legacy zero/four-token CLI behavior.
Supply the DSpark checkpoint path through `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model`.
Validate the fixed architecture and all BF16 tensor names/shapes before execution.
Bind loaded tensors to stable configuration/weight file hashes.

Rust keeps scheduling, allocation, prefix reuse, and rollback.
Python calculates target features and DSpark proposals through project-owned model and IR code.
Project builds, tests, and inference keep their independent runtime contract.
Checkpoint Python code does not execute.

The draft shares target embedding and vocabulary weights.
Its context KV uses different physical storage at Rust-owned FA page IDs.
Only accepted target input rows enter that context.
Temporary bidirectional proposal KV does not enter persistent context.
The GDN pool keeps BF16 speculative state and the existing rounding constraints.

Use full conditional proposal probabilities for stochastic acceptance `min(1,p(x)/q(x))`.
On rejection, use normalized `max(p-q,0)`. The bonus token uses target probabilities.
Apply user sampling parameters and grammar masks to target and proposal distributions.
Use a different draft random generator and keep target generator ownership.

The confidence head can make the prefix shorter, with a default threshold of `0.2`.
A first confidence product less than the threshold returns zero candidates, and that step uses one target token.
Set the threshold to `0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.
The setting comes from a small text/tool-history experiment with inputs different from formal inputs.
The [handoff](../handoff.md) records its scope.

Keep target graphs and draft proposal graphs in different memory pools.
Compile target features, context injection, backbone, Markov step, and greedy proposal with `fullgraph=True`.
Keep explicit provider failures and semantic mutation contracts.

The worker selects one mode when it starts. It keeps that mode until it stops.

## Acceptance

Use the phase-latency protocol for each mode.
Acceptance rate is total accepted drafts divided by total scheduled drafts.

Keep full attempts that pass, fail, or stop before completion, with source, binary, environment, loaded checkpoint, and configuration identities.
Record scheduled draft counts, accepted counts, compile/capture audit, capacities, and peak GPU memory.
Returned next-step proposals cannot supply the acceptance-rate denominator.
Portable summaries stay derived evidence.

Add DSpark context boundary cases, two-API constraints, mixed batches, cancellation, and long-context prefix checks.
OMP must read two source files and forward their results into a subsequent model request before its last answer.
Compare response IDs with active-mode server logs and examine answer meaning after the automated checks.

Keep the original TileLang files, manifest hash, and case IDs unchanged.
Add a different supplemental reference and statically derived DSpark operator cases.
Apply the original accuracy and three-round operator speed/confidence gates to these cases.

## Alternatives and consequences

Greedy token equality alone cannot supply stochastic acceptance semantics.
Full conditional probabilities increase proposal storage and sampling work.
The implementation keeps this work for correct configured stochastic output.
The performance claim must use measured acceptance results.

See [requirements](../requirements.md) and [tests](../testing.md#framework-performance).
