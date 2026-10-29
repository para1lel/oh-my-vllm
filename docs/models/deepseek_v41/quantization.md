# Quantization contract

The adapter uses the released [DeepSeek reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) for block quantization.
The reference source SHA-256 is `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
This contract is applicable to activations, window KV, compressed KV, and index keys and queries.
It does not change the original weight values or scales.

## Mathematics

Let `A` be the largest absolute FP32 value in one group.
Use these normalization scales:

| Format | Normalization scale |
|---|---|
| FP8, unrounded | `max(A, 1e-4) * (1 / 448)` |
| FP8, rounded | Power-of-two ceiling of `max(A, 1e-4) * (1 / 448)` |
| FP4, UE8M0 scale | Power-of-two ceiling of `max(A, 6 * 2**-126) * (1 / 6)` |
| FP4, E4M3 scale | `E4M3(max(A, 6 * 2**-9) / 6)` |

Power-of-two rounding uses the FP32 exponent and mantissa bits.
It does not use a rounded `log2` result.
FP4 E4M3 scale rounding occurs before value normalization.
UE8M0 scale storage rounds up.
Unrounded FP8 normalization still uses the original FP32 scale when its stored scale uses UE8M0.

Use nearest-even rounding for E4M3 and E2M1 values.
Clamp normalized FP8 values to `[-448, 448]`.
Clamp normalized FP4 values to `[-6, 6]`.
E2M1 magnitude codes `0` to `7` represent `0, 0.5, 1, 1.5, 2, 3, 4, 6`.

Even channels use the low nibble.
Negative zero keeps its sign.
A zero FP4 group keeps a nonzero scale.

An in-place call rounds normalized values to the selected format before dequantization.
The result returns to the input dtype.
An encoded call returns E4M3 values or packed E2M1 bytes, with scales in different storage.
FP4 encoded storage has half the logical width.

FP4 projection activations use FP8, not FP4.
The native projection consumes 32-channel activation scale groups.
Each 128-channel activation scale repeats across four native groups.
Reference FP8 projection expands weight scales across row and column groups.
FP32 scale inputs use the reference projection path, not the native UE8M0 byte interface.

Dense and grouped projections use raw K32 dot products with FP32 accumulators.
FP4 weights convert accurately to FP8 before each raw dot.
FP4 weight scales apply to each output row.
FP8 weight scales apply to 32-row groups.

Start each raw dot at zero.
Apply `scaled = mul_rn(raw_dot, activation_scale)`.
Apply `sum = fma_rn(scaled, weight_scale, sum)`.
Do not multiply the two scales before the dot product.
Round only the completed sum to the selected output type.

RMS normalization rounds each square before it adds the square to an accumulator.
Use the full logical row count and width to select the mean reduction layout.
Multiply the sum by `FP32(1 / width)` after the reduction.
Do not replace this multiplication with division.

Multiply each input by the inverse RMS in FP32.
Multiply this FP32 result by its weight in FP32.
Round the result to BF16.

The [router](../../../python/oh_my_vllm/models/deepseek_v41/model.py) projects inputs and weights in FP32 before score calculation.
Do not round router logits to BF16 before expert selection.
Use `sqrt(x > 20 ? x : log1p(exp(x)))` for each score.

The six-score sum is `((s0 + s4) + s2) + ((s1 + s5) + s3)`.
The three-score sum is `(s0 + s2) + s1`.
Divide each selected score by `sum + 1e-20`.
Multiply the quotient by `1.5`.
Native routing selects the lowest global expert ID when corrected scores are equal.
Released top-k can select different IDs for such ties.

Tensor-parallel expert down projections must keep FP32 partial sums until rank reduction.
The grouped and non-grouped TP4 paths keep these FP32 partial sums.
Round the reduced result to BF16.

Grouped up projections and TEP4 down projections keep BF16 output.
Single-rank grouped down projections return full BF16 expert outputs, not rank partial sums.
The grouped interface accepts BF16 or FP32 output without a workspace size change.
These output types do not specify the full tensor-parallel accumulation order.

Shared experts keep full weights on each rank.
Add their output after routed rank reduction.
Do not reduce the shared output across ranks.
Attention output projections keep their local BF16 rounding before FP32 rank reduction.

Sparse attention processes selected indices in their original order, in blocks of 64.
QK and probability-V products use BF16 operands with FP32 accumulation.
The running maximum starts at `-1e30` and does not include the sink.
Each block rescales the earlier FP32 numerator and denominator when the maximum changes.
Unrounded exponentials contribute to the denominator.
Round these exponentials to BF16 before multiplication by V.

The denominator first sums across warps with XOR offsets `128, 64, 32`.
It then sums across four lanes with XOR offsets `2, 1`.
Use `fma_rn(previous_sum, scale, block_sum)` for the denominator update.

After all selected blocks, add the sink exponential to the denominator.
Divide the numerator by the denominator.
Round the output to BF16.
Keep duplicate indices and masked slots in their original positions.
Empty and fully masked rows return zero, even when a finite sink exponential underflows.

DSpark proposal queries share the full draft-index block.
Target attention keeps its causal index rules.

Add `hc_eps` to row softmax values before the first column normalization in Sinkhorn mixing.
Use `hc_eps=1e-6` for the released model.
For four copies, row sums use XOR offsets `2, 1`.
Column sums use XOR offsets `8, 4`.

HC collapse rounds each FP32 product before a left-to-right sum.
HC expansion first adds the rounded residual products in that order.
Add the separately rounded branch product after the residual sum.
Round FP32 multiplication and addition independently in these sums.

HC mixing accumulates squared inputs independently in four FP32 accumulators.
The full logical row count selects the reduction layout.
Apply the inverse RMS after the FP32 projection.
The native projection uses the tensor runtime's cuBLAS library and a caller-owned 32 MiB workspace.
This workspace size is part of the numerical contract.

## Target row groups

Initial target prefill, single-token target calls, and MTP keep their existing row layouts.
Nonzero multi-token target calls keep batch-major `[batch, step, ...]` storage.
Their logical row count is the request batch size.
HC and router FP32 projections use this row count for each source step.
Their SGEMM calls write directly to strided physical output storage.

RMS selects its mean reduction layout from the logical row count and processes all physical rows.
Engram uses four times the token logical row count for per-copy mean and dot reductions.
FP32 multiplication and addition round independently in its final residual addition.
Engram embedding, KV projection, target attention, and routed expert computation stay batched.
The router can use the HC cuBLAS workspace after HC projection completes.

## Execution

The [model quantizer](../../../python/oh_my_vllm/models/deepseek_v41/quantization.py) selects the native BF16 path or reference computation.
The [native interface](../../../python/oh_my_vllm/operators/transform/api.py) specifies shapes, formats, alignment, and overlap rules.
It uses the caller's CUDA stream.
Each warp reads one full group before an in-place write.
Native quantization does not allocate temporary device storage.

HC preparation keeps an FP32 residual copy, mean and inverse statistics, and the cuBLAS workspace.
Native HC calls accept up to 32768 rows.
Small-row workspace plans keep their 384-row limit.
Prepare the thread-local cuBLAS handle before Graph recording.
Keep the captured workspace until Graph release.
The native manifest records cuBLAS library image and header identities for the tensor runtime package.

Tensor preparation can make an aligned contiguous input copy.
Encoded output and scales use different storage.

Dense projection does not use device scratch storage.
Grouped projection uses device row offsets without host descriptor uploads.
Each device count must be between zero and the prepared maximum row count.
The counts must sum to the full input row count.
Invalid device counts produce NaN output without input or weight access.

Prepared weight storage must stay valid until execution completes or the Graph is released.
Prepared cache entries keep the original weight and scale owners, not only their addresses.
Use different prepared entries for source views with different dtypes, shapes, or strides.
Do not change weights or scales after preparation.

Host metadata failures occur before launch.
Invalid device values or scales produce invalid scale codes.
Native attention checks these codes before it accepts the output.
Inputs must have finite reconstruction results, as specified by the interface.

Native attention keeps window and compressed KV in two tables.
The attention call does not quantize these BF16 tables.
The operator also accepts encoded tables with their scale arrays.
The reference path joins their BF16 tables before computation.

Native sparse attention is the default for supported inputs.
Set `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=0` to select reference attention for short queries.
Long queries keep native selection unless a native status failure occurs.
Attention status failures disable native selection and use the reference path.

## Acceptance

The [quantization checks](../../../tests/test_deepseek_quantization.py) use independent literal values and encoded bytes.
They check scale rules, nearest-even ties, signed zero, exponent boundaries, and projection scale groups.
CUDA cases check stream order, invalid metadata, empty history, and temporary storage bounds.
These checks do not replace independent full-model endpoint fixtures or the approved HTTP and AgentX workloads.

## Reference execution

Use an isolated development environment for the released TileLang reference.
Do not add TileLang to inference dependencies.
With this quantizer source, TileLang 0.1.8 can reuse input shared storage before all input stores complete.
Use a CUDA block barrier before each reference quantizer writes its output shared storage.
Add `T.sync_threads()` immediately before `T.copy(y_local, y_shared)` in `act_quant_kernel` and `fp4_quant_kernel`.
Do not change the equations, casts, or original checkpoint files.

Record the original source, patch, modified source, compiler, and runtime identities in each reference manifest.
Check packed and in-place formats with Compute Sanitizer Racecheck before you generate expected arrays.
Do not freeze arrays unless repeated target and DSpark endpoints are bit-exact.
Identify this backend as a reference with a synchronization patch, not an unmodified release.
Do not import candidate modules to generate expected arrays.
