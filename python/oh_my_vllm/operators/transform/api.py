"""Normalization, adjacent-pair rotation, and grouped activation quantization.

RMS parameters contain original BF16 weights, without an offset. RoPE parameters
contain FP32 cosine/sine pairs for each row, shared by its attention heads.
RMS values have shape (rows, width); RoPE values have shape (rows, heads, width).
RMS width is 128, 512, 1024, 1280, or 5120. Epsilon is 1e-20 or 1e-6.
rms_grouped(values, weight, output, epsilon, logical_rows) keeps the RMS
arithmetic of logical_rows source rows but writes all physical rows. Nonempty
values require positive logical_rows that divides the physical row count.
Empty values require logical_rows == 0. rms retains physical-row selection.

engram(residual, kv, q_weight, k_weight, token_mask, output, logical_rows)
uses contiguous BF16 residual/output (rows, 4, 5120), kv (rows, 25600), and
q_weight/k_weight (4, 5120). rows is at most 32768. All base addresses must
be 16-byte aligned and on the same CUDA device. Output must not overlap any
input. token_mask is contiguous U8 (rows,) with canonical bytes 0 or 1, or
U8 (0,) for all valid rows. A zero mask selects source arithmetic with a
positive-zero gate, including its signed-zero multiply and add behavior.
logical_rows counts token rows, not HC copies. Zero selects the physical row
count. Otherwise it must be positive, at most rows, and divide rows. Empty
values require zero. Mean and dot reductions use 4 * logical_rows source
rows; the grid writes every physical token and HC copy. The native operation
uses epsilon 1e-20 and clamp 1e-6. It allocates no device workspace and does
not synchronize. Metadata errors raise before launch. The caller retains all
buffers through Eager completion or graph release.

engram_lookup_stage(weight, scales, ids, packed, start) gathers host U8
FP8 rows and UE8M0 scales into caller-owned host storage. ids is contiguous
I64. Each packed row contains width value bytes, width / 32 scale bytes,
one mask byte, and padding to 16 bytes. Out-of-shard rows have mask byte 1.
engram_lookup_decode(packed, output) decodes CUDA packed rows to BF16 on the
caller's stream. width is divisible by 32 and at most 8192; each call has
at most 65536 rows. Inputs and outputs must not overlap. The caller protects
pinned storage until transfer completion and retains device storage through
execution. Stage outside CUDA Graph recording.

Inputs must be finite and bounded
so FP32 intermediate values and BF16 outputs remain finite.
Retain inputs through Eager completion or graph release.

quantize(values, output, scales, bits, group_size, scale_format, round_scale,
         inplace) writes caller-owned buffers on the caller's CUDA stream.
values is contiguous BF16 (rows, width). width is positive and divisible by
group_size, which is 16, 32, or 128. Empty rows are valid. All tensor base
addresses must be 16-byte aligned and on the same CUDA device.
bits is 4 or 8. Encoded output is U8 (rows, width * bits / 8), with even FP4
channels in the low nibble. inplace selects BF16 (rows, width) output containing
quantized then dequantized values. Only this mode permits exact values/output
aliasing; partial overlap is invalid. scales must not overlap either buffer.
scale_format is 0 for FP32, 1 for U8 UE8M0 codes, or 2 for U8 E4M3 codes.
scales has shape (rows, width / group_size).

FP8 permits formats 0 and 1. Its scale is max(absmax, 1e-4) * (1 / 448).
round_scale rounds that normalization scale upward to a power of two. UE8M0
storage always rounds upward, even when round_scale is false; normalization
still uses the unrounded scale in that case, as in the released quantizer.
FP4 permits formats 1 and 2 and ignores round_scale. Format 1 uses the power-of-2
ceiling of max(absmax, 6 * 2**-126) * (1 / 6). Format 2 uses
E4M3(max(absmax, 6 * 2**-9) / 6), rounded before normalization.
E4M3 and E2M1 conversions use nearest-even rounding and retain signed zero.
BF16 output is BF16(decoded_quantized_value * normalization_scale).

Inputs must be finite, with representable positive scales and finite BF16 QDQ
results. Invalid device data or an unrepresentable scale produces a NaN scale
(FP32 NaN, UE8M0 0xff, E4M3 0x7f), not a host exception. Such groups have zero
FP4 codes, NaN FP8 codes, or NaN BF16 output. Consumers must check invalid scales.
Metadata and overlap failures raise before any launch. The operation allocates
no temporary device storage and does not synchronize. Retain every buffer
through Eager completion or graph release.

gather_fp4(codes, scales, table, slots, output, status, page_rows, group_size,
           format) restores complete history keys into caller-owned BF16 output
[B,T,D]. codes is U8 [pool_rows,D/2], and scales is U8 [pool_rows,D/group_size].
The I64 table maps [active_slots,logical_pages] to physical pages. I64 slots
has B entries. Format 1 uses UE8M0, and format 2 uses E4M3. D is divisible
by group_size, which is 16, 32, or 128. Page zero contains masked padding.
The caller initializes status I32 [1] to zero. Invalid metadata, invalid
scales, or nonfinite results set status to one and write NaN for rejected
values. Check status before state commit. Inputs, output, and status must
not overlap. The operation uses FP32 multiplication and BF16 nearest-even
rounding, without temporary storage or synchronization. Keep all buffers
alive through execution or graph release. Map their storage before recording.
"""

from typing import NamedTuple

FUNCTIONS = (
    "rms",
    "rope",
    "quantize",
    "rms_grouped",
    "engram",
    "engram_lookup_stage",
    "engram_lookup_decode",
    "gather_fp4",
)


class Inputs(NamedTuple):
    values: object
    parameters: object


class Outputs(NamedTuple):
    value: object
