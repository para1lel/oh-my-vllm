# GDN execution plan

Use the [model contract](model.md) and [full serving boundary](fusion.md).
This plan does not change the current CUDA implementation.

## Selection inputs

Select the algorithm from tokens per sequence, batch size, and state work per CTA.
A CTA is one CUDA thread block.
A value tile contains contiguous value rows and all 128 key elements of one state head.
The number of independent CTAs is `batch * 48 * (128 / value_rows_per_tile)`.
The number of CTAs alone does not give execution time or occupancy.

The modeled recurrent paths have 1 or 5 tokens per sequence.
The modeled prefill paths have at least 784 tokens per sequence.
The model supplies no algorithm comparison for lengths from 6 to 783.

## Short sequences

Use sequential token updates with parallel value tiles as the initial design.
Load BF16 state once. Keep FP32 live carry across the token updates.
Save the necessary private BF16 states. Keep read slots unchanged.

The [FlashInfer BF16 implementation](https://github.com/flashinfer-ai/flashinfer/blob/deca0e2959d521fe37efffdb6074cd357e836042/flashinfer/gdn_kernels/gdn_decode_bf16_state.py) supplies this design reference.
Its selection rules use state-head count and token count.
Its single-token and multiple-token paths have different selection rules.

Use these initial candidates for the project shapes:

| Batch | Tokens per sequence | Value rows per tile | CTAs |
|---|---:|---:|---:|
| 1 | 1 or 5 | 16 | 384 |
| 4 | 1 or 5 | 32 | 768 |

These candidates come from source rules. They are not project performance results.
More value rows per CTA can reduce repeated Q/K and gate computation when the batch supplies sufficient parallel work.
Include head norm, output gating, quantization, and projection in the comparison.
A separate norm launch can increase the time for the full operation.

## Long sequences

Use 64-token blocks as the initial prefill design.
The [FlashQLA Blackwell implementation](https://github.com/QwenLM/FlashQLA/tree/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/blackwell) supplies this design reference.
It solves a block triangular system and uses Tensor Core matrix products for state updates and output.
It keeps FP32 live carry and uses BF16 matrix operands.
Give the prefill operand types and rounding points in the model contract before implementation.
Use the pinned vLLM prefill source for that specification.

Divide computation blocks at sequence ends and checkpoint exports.
A 784-token interval contains twelve 64-token blocks and one 16-token tail.
Keep FP32 live carry after a BF16 checkpoint export.

Sequence partitions can supply more parallel work when batch, head, and value tiles are insufficient.
Include the initial-state preparation and correction costs before selection.
The [FlashQLA partition selector](https://github.com/QwenLM/FlashQLA/blob/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/cp/autocp/decision.py) compares these costs with execution without partitions.
Do not discard incoming state or earlier tokens to decrease execution time.

## Implementation gate

Use CUDA C++ or CuTe DSL. Use the external sources as design references.
Keep the current correct path until a replacement passes its numerical checks and full-boundary comparison.

Use existing modeled shapes to select project parameters. Add intermediate lengths only when a selection question makes them necessary.
Include all launches, state exports, temporary storage, and tail work in eager and graph timing.
Keep the workload probabilities and roofline calculation unchanged during each comparison.
