# Projection arithmetic

The [quantization contract](quantization.md) specifies one raw dot and one scale update for each K32 group.
The grouped FP4 path keeps the original operator interface, BF16 expert rounding, and complete down-projection sum.

## Exact scale combination

The [native scale update](../../../python/oh_my_vllm/operators/projection/cuda/fp8_k32.cuh#L32)<!-- source: __device__ __forceinline__ float fp8_scaled_add(float raw, uint8_t activation, uint8_t weight, --> can combine two scale exponents.
Each finite E4M3 product is a multiple of `2**-18`.
A raw K32 dot stays a multiple of this value and has magnitude below `2**23`.
For activation scale codes at most 232, its multiplication is exact, also for subnormal results.
The smallest scale exponent is -127, so the smallest nonzero product step is `2**-145`.
The largest permitted exponent is 105, so the result cannot overflow.

If the combined exponent is between -149 and 127, its power of two has an exact FP32 representation.
One FMA with that value then gives the same result as the required MUL and FMA.
Other scale pairs keep the original operations.
Invalid scale codes keep the original NaN behavior.

## Native reduction layout

The [FP16 tile conversion](../../../python/oh_my_vllm/operators/projection/cuda/half_k32.cuh#L19)<!-- source: __device__ __forceinline__ void half_store(uint16_t *destination, int row, int column, --> keeps the CUDA 13.4 FP8 MMA reduction layout.
For each K32 group, the first K16 operation uses positions `0,1,4,5,...,28,29`.
The second uses positions `2,3,6,7,...,30,31` and adds to the first result.
E4M3 and E2M1 values have exact FP16 representations.
The conversion changes storage and instruction selection without changing operand values or the K32 reduction order.

## Dense prefill tiles

The [dense prefill kernel](../../../python/oh_my_vllm/operators/projection/cuda/tcgen_k32.cuh#L15)<!-- source: constexpr int tcgen_rows = 128, tcgen_columns = 256, tcgen_slots = 2, tcgen_raw_stages = 3; --> uses 128 input rows and 256 output columns per tile.
It applies to at least 128 rows, output widths above 1024 and divisible by 256, and input widths at most 6144.
Input and weight addresses must have 128-byte alignment.
Other shapes keep their existing kernels.

TMA loads original FP8 values in K128 blocks through three shared buffers.
Separate threads convert values to FP16 and issue two K16 operations for each K32 dot.
Two TMEM slots keep independent raw dots.
Output threads apply one MUL and one FMA for each K32 group in order.
No FP16 weight copy stays resident between calls.

Prepare each output type and scale layout on the calling thread and device before Graph capture.
Capture can encode TMA descriptors for new quantization storage.
The kernel receives these descriptors by value, and Graph owners keep the input and weight storage valid.

## Shared expert projections

Shared experts keep full FP8 weights and scales on each rank.
Their original gate and up Linear modules share one quantized input and its scales when no hooks apply.
Replaced modules, forward overrides, hooks, and compiled calls keep their original execution.

TP4 and TEP4 compute shared experts once for each complete prefill chunk.
Routed experts keep their separate row batches.
Their native gate and up projections share one FP8 input and UE8M0 K32 scales.
The down projection quantizes the BF16 activation product separately.
DEP4 computes shared experts only for the owner's rows.
Each projection keeps raw K32 dots and ordered FP32 scale accumulation.

Add the shared output after the routed output exchange.
Do not reduce shared output across ranks.
The routed FP32 sum receives this BF16 shared output before the final BF16 conversion.

## Routed prefill waves

`OH_MY_VLLM_DEEPSEEK_MOE_PREFILL_ROWS` selects 1024, 4096, or 8192 token rows for supported Eager target prefill.
Four-rank TP4 and TEP4 use 8192 rows by default.
Other configurations use 1024.
Larger projection waves decrease the number of expert tiles that are not full and the number of device metadata setup operations.
Set this environment variable to 1024 to decrease temporary storage.
Router calls keep their existing row groups and source-step layouts.

TP4 gate, up, and SwiGLU computation use the projection wave.
Product gathers, down projections, local sums, and output gathers keep at most 1024 token rows for each step.
Each collective call uses the message dimensions and sequence of the existing path.

TEP4 gate, up, SwiGLU, down, and local sums use the projection wave.
Its routed FP32 reductions keep the existing 1024-row slices and message sequence.
Each expert output rounds to BF16 before the local FP32 sum in expert ID sequence.

One [operator workspace](../../../python/oh_my_vllm/models/deepseek_v41/moe_prefill.py#L29)<!-- source: class PrefillWorkspace: --> belongs to the model.
Its target layers use the buffers again on the calling stream.
The initial supported Eager call reserves storage for the selected wave and the configured routes for each token.
A changed buffer shape or device replaces the buffers, and `_apply` clears them.

Only calls with CUDA BF16 inputs, native FP4 experts, gate widths 576 or 2304, and SwiGLU limit 10 use this path.
Decode, Graph, DEP4, and unsupported expert bindings keep their existing paths.

## Local expert sum

The official MoE adds each local expert output in increasing expert ID order.
Each expert output first rounds to BF16.
The [local sum](../../../python/oh_my_vllm/models/deepseek_v41/kernel.py#L222)<!-- source: def ordered_expert_sum(values, sorted_routes, output, topk): --> uses one projection row per token and route slot.
Projection rows follow expert ID order.
The CUDA kernel sorts these row indices and adds their BF16 values to FP32 in that order.
Unused route slots contribute no value.

TP4, TEP4, and DEP4 use this sum before their output exchange.
The map uses at most six INT32 entries per token.
Its storage does not depend on the output width.
Graph replay rebuilds the map from the current routes.

## Fixed-capacity projections and sums

`projection.grouped_project_padded(encoded, scales, weight_ptrs, scale_ptrs, counts, output, scratch, max_rows, status)` uses a device-count prefix.
`counts` is CUDA INT32 `[G]` in expert ID order.
Each count is nonnegative and at most `max_rows`.
Their sum is at most the input capacity.
The input uses encoded FP8 values and UE8M0 K32 scales.

The caller keeps the prepared FP4 weight and scale storage until completion or Graph release.
Pointer arrays alone do not keep that storage.
The valid output prefix keeps the strict projection's arithmetic and BF16 or FP32 output type.
Each output row after the prefix is zero, so SwiGLU can read the complete buffer.
A separate kernel clears these rows.

The existing workspace query uses capacity as its row count and checks `capacity <= G * max_rows`.
Use `max_rows=max(1,capacity)` when no smaller bound is available.

`routing.expert_sum_padded(values, sorted_routes, counts, output, route_rows, status)` uses the same live assignment prefix.
`values` contains BF16 expert outputs.
`sorted_routes` contains INT64 token-and-slot indices in expert order.
The caller supplies INT32 `route_rows[token_capacity,topk]`, with one to six slots per token.
The kernel adds values in projection-row order and writes FP32 output.

Token rows with no assignments are zero, also for unused capacity rows.
Invalid counts, assignment overflow, invalid routes, and duplicate slots set an error bit.
The sum does not read the values or routes after the live prefix.

Each call uses caller-owned CUDA INT32 `status[1]` without a host read.
Clear status once at Execute start.
Errors stay set across calls and Graph replays.
Invalid projection counts or an existing error produce a zero projection output without reading inputs or weights.
A sum with nonzero status produces zero output without reading values.

Check status at the step boundary before cache commit.
Reject the step on any error.
Zero-capacity calls still check counts.
The strict `grouped_project` entry keeps its exact count sum and invalid-count NaN behavior.
Caller integration with TEP4 Graph and DEP4 dispatch is incomplete.

## Batched target arithmetic

Experimental batched CUDA targets must use the released configuration: width 5120, four HC copies, FP8 dense weights, and FP4 routed weights.
Attention must use 64 heads of width 512. Index scoring must use 32 heads of width 128.
Routing must use 384 experts and six selected experts.
The attention window has 128 slots, and index selection has at most 512 slots.

Projection scales must use UE8M0 K32 groups. Native sparse attention must be enabled and have no previous status failure.
Unsupported configuration settings raise an error before batch input binding and model computation.
CPU reference execution and targets without batching keep their existing paths.

TP4 and TEP4 batches keep each request's original BF16 matrix shape for index head weights, index keys, and `wo_a`.
Index scoring completes each request's score chunks, rank sums, and selection before starting the next request.
The query chunk budget is 256 MiB for one request, independent of batch size.
Each index score collective keeps the same shape and FP32 sum as the separate request call.
All ranks use request order, then query chunk order, also during Graph capture.
The MoE and attention output collectives keep their existing message sizes.

## Numerical checks

The [shape check](../../../development/model/check_quantized_gemm.py) compares BF16 and FP32 outputs with an immutable native baseline.
Its cases cover TP4 and whole-expert shapes, empty experts, unequal counts, and selected expert row counts from 1 through 32768.
Its native baseline comparison is an arithmetic regression check, not an independent frozen model reference.
The [literal checks](../../../tests/test_projection_precision.py) also check sparse K32 values and Graph replay with changed expert counts.
Keep raw records external to the repository.
