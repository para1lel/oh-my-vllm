# GDN execution plan

Use the [model contract](model.md) and [full serving boundary](fusion.md).
The worker has separate scan and chunked prefill paths.

## Selection inputs

Select the algorithm from tokens per sequence, batch size, and state work per CTA.
A CTA is one CUDA thread block.
A value tile contains contiguous value rows and all 128 key elements of one state head.
The number of independent CTAs is `batch * 48 * (128 / value_rows_per_tile)`.
The number of CTAs alone does not give execution time or occupancy.

The modeled recurrent paths have 1 or 5 tokens per sequence.
The modeled prefill paths have at least 784 tokens per sequence.
The chunked path selects prefill batches with more than 128 tokens in the longest sequence.
Sequence lengths can differ. Empty sequences produce no chunks.
Other inputs use the scan path.

## Short sequences

Use sequential token updates with parallel value tiles as the initial design.
Load BF16 state once. Keep FP32 live carry across the token updates.
Save the necessary private BF16 states. Keep read slots unchanged.

The [operator](../../python/oh_my_vllm/kernels/cuda_backend/gdn.cu) uses the same paired FP32 recurrence and BF16 output buffer for decode and speculative verification.
Single-query verification uses the BF16 Tensor Core gate projection.
Multiple-query verification with at most 32 rows keeps the gate projection fused with convolution.

Balanced batches with one or two sequences use fixed query counts for two to five tokens.
Separate inputs use host metadata checks. Interleaved serving inputs keep GPU metadata checks.
The [gate helper](../../python/oh_my_vllm/kernels/cuda_backend/gdn_recurrent_core.cuh#L137)<!-- source: __device__ inline float2 fast_gate_pair(float alog, float activation, float projected_b) { --> uses a fast path for a limited range of inputs.
Other inputs use the complete calculation. Single-token and generic paths use the same helper.

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

The [prepare kernel](../../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_native_prepare.cuh#L60)<!-- source: __global__ __launch_bounds__(256, 4) void native_prepare_bf16_inverse_kernel( --> computes causal QK coefficients and stores them as BF16.
The output kernel reads this buffer.
The [workspace definition](../../python/oh_my_vllm/kernels/gdn.py#L23)<!-- source: def chunk_workspace(rows, batch, max_queries, verify): --> includes this buffer for runtime allocation and startup memory reservation.

The [inverse implementation](../../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_bf16_inverse.cuh#L68)<!-- source: template <class Storage> __device__ inline void inverse_blocks(Storage &s) { --> uses 8-row diagonal solves in FP32.
It reads BF16 matrix entries and stores BF16 inverse blocks.
BF16 block products use FP32 accumulation.

Sequence partitions can supply more parallel work when batch, head, and value tiles are insufficient.
Include the initial-state preparation and correction costs before selection.
The [FlashQLA partition selector](https://github.com/QwenLM/FlashQLA/blob/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/cp/autocp/decision.py) compares these costs with execution without partitions.
Do not discard incoming state or earlier tokens to decrease execution time.

## Work calculation

Use the shared Python selection rule for scan and chunked work.
Split the valid token rows at 64-token computation boundaries, checkpoint exports, and sequence ends.
Do not count padded rows or repeated work for grouped Q/K heads.

For one chunk, let `m` be its valid rows, `D=128`, `H=48`, and `G=16`.
The BF16 matrix work is `6*H*D*D*m + H*D*m*(3*m-1) + 2*G*D*m*m` FLOPs, without the inverse.
This includes state projection, state updates, output, triangular W/U products, KK, and causal QK.
The W/U count omits copies through the unit diagonal.

The FP32 count includes the live-state update, diagonal inverse solves, gate scaling, and output combination.
For the inverse, let `T=m*(m-1)*(m-2)/3` and `d_i=min(8,max(0,m-8*i))` for `i=0..7`.
The diagonal work is `F=sum_i d_i*(d_i-1)*(d_i-2)/3`.
The inverse contributes `H*F` FP32 FLOPs and `H*(T-F)` BF16 FLOPs.
These counts omit products with structural zero and one values and padded rows.
The work model keeps the original FP32 token recurrence count for scan paths.

## Changes

Use CUDA C++ or CuTe DSL. Use external sources as design references.
Keep the current path until its replacement passes the applicable numerical and state checks.
Use the [GPU execution rules](../execution.md) for scheduling and graph resources.
Use the [HTTP comparison](../performance.md) for final performance acceptance.
