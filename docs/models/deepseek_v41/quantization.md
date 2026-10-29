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
UE8M0 scale storage rounds upward.
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
An encoded call returns E4M3 values or packed E2M1 bytes, with separate scales.
FP4 encoded storage has half the logical width.

FP4 projection activations use FP8, not FP4.
The native projection consumes 32-channel activation scale groups.
Each 128-channel activation scale repeats across four native groups.
Reference FP8 projection expands weight scales across row and column groups.
FP32 scale inputs use the reference projection path, not the native UE8M0 byte interface.

## Execution

The [model quantizer](../../../python/oh_my_vllm/models/deepseek_v41/quantization.py) selects the native BF16 path or reference computation.
The [native interface](../../../python/oh_my_vllm/operators/transform/api.py) specifies shapes, formats, alignment, and overlap rules.
It uses the caller's CUDA stream.
Each warp reads one complete group before an in-place write.
Native quantization does not allocate temporary device storage.

Tensor preparation can make an aligned contiguous input copy.
Encoded output and scales have separate storage.

Metadata failures occur before launch.
Invalid device values or scales produce invalid scale codes.
Native attention checks these codes before it accepts the output.
Inputs must have finite reconstruction results, as specified by the interface.

Native attention uses separate window and compressed KV tables.
The reference path joins their BF16 tables before computation.
Short-query native selection and attention status fallback keep their existing rules.

## Acceptance

The [quantization checks](../../../tests/test_deepseek_quantization.py) use independent literal values and encoded bytes.
They check scale rules, nearest-even ties, signed zero, exponent boundaries, and projection scale groups.
CUDA cases check stream order, invalid metadata, empty history, and temporary storage bounds.
These checks do not replace independent full-model endpoint fixtures or the approved HTTP and AgentX workloads.
