# Native matrix and affine-scan diagnostics

These experiments do not select a production change or show framework acceptance.
The [source-bound summary](../evidence/affine-diagnostics/verification.json) binds original sources, loaded modules, inputs, results, and failed attempts.
Use the [measurement method](../profiling/measurement.md) and [algorithm research](optimization.md).

## Complete emulation cost

The [cuBLAS guide](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#floating-point-emulation) gives FP32 emulation with three BF16 components and nine products.
Input reconstruction does not show IEEE FP32 compliance or complete GDN accuracy.
Special-value detection and output corrections can have more cost than matrix computation.

Seven shapes in three modes gave 21 comparisons against CPU FP64 results.
Each comparison checked the first eight batches or all batches when the batch count was less than eight.
Fixed finite random inputs passed the probe tolerance. Extreme values and overflow were not checked.

For 24576 batches of `128 x 128 x 64`, eager emulation took 9.654 ms with default special-value handling.
Its timeline contains approximately 0.517 ms of matrix computation, 1.624 ms of detection, and 7.508 ms of output correction.
These profiler intervals are diagnostic, not formal timing samples.

A separate finite-only probe disabled special-value handling. The same shape took 0.545 ms, compared with 1.567 ms for native FP32.
Finite inputs alone do not prevent intermediate overflow or show correct signed-zero behavior.
This probe does not supply a production input contract.

A finite-only FP64 probe compared four larger shapes in native and eager emulation modes.
All eight first-eight-batch comparisons passed the CPU FP64 probe tolerance.
Native medians were 0.051–3.305 ms. Eager emulation medians were 1.021–49.503 ms.
The sampled emulation timeline includes scaling, packing, matrix computation, and output conversion.
With special-value handling disabled, eager emulation stayed slower than native FP64 for these shapes.

## Affine scans and layout

A chunk maps its incoming state through a matrix and a displacement: `next_state = state * T + D`.
Chunk composition gives a map for adjacent chunks. An exclusive prefix supplies each chunk's incoming state.
The candidates kept the full initial state. They did not use decay truncation.

The first candidate copied matrix views, concatenated prefix states, and used addition kernels.
Later candidates used matrix views and matrix epilogues with different input and output matrices.
The layouts and matrix algorithm selection changed. The measurements do not show which change caused the timing difference.

| 32768-token candidate | CUDA baseline | Candidate median | Preprocessing in capture |
| --- | --- | --- | --- |
| First BF16 emulation chain | 16.050 ms | 20.696 ms | No |
| Pointer-array layout | 16.062 ms | 13.336 ms | Yes |
| Strided layout | 16.066 ms | 11.921 ms | Yes |

Each configuration had ten interleaved samples. This is not the formal paired admission protocol.
Changed-query replay verification is not available for the first candidate. It cached normalized queries before capture.

Three model-input prefixes of 768 tokens and a synthetic 384-token case passed independent FP64 comparisons.
The 32768-token case used repository CUDA, not an independent long reference.
The candidates process one sequence with complete chunks. Verification of ragged tails and multiple sequences is not available.
For some chunk counts, the strided prototype copies matrix views. Five chunks with two heads fail its no-copy assertion.

## Cancellation and replay

The [cancellation tests](../../tests/numerical/test_gdn_prefill_replay.py) check small nonzero final states after almost complete removal of an initial state.
Eight diagnostic workloads used zero values, zero decay, almost equal keys, and a nonzero initial state.
They included 384-token and 4096-token inputs.

Repository CUDA passed all eight independent FP64 comparisons.
The strided BF16, first native FP32, and first BF16 affine candidates each failed all eight final-state comparisons.
The pointer-array candidate was not part of this cancellation collection.
The strided candidate's state NRMSE was 4.20%–173.66%.
The outputs passed their absolute tolerance. Only final-state comparisons failed.

Native FP32 matrix computation also failed. The results do not show that BF16 emulation or address mapping caused the failures.
Examine upstream projection precision and affine-composition rounding separately.
None of the affine candidates is selected.

The strided candidate passed three graph replays at 384 tokens, with one query head and two value heads.
Two replays followed in-place changes to queries, keys, values, gates, and initial state.
Sequence offsets and shapes stayed equal. This result does not correct the cancellation failures.

## Methods to keep

Measure preprocessing and the complete matrix chain. Use device timelines to locate copies and elementwise kernels.
Use matrix views and epilogues when paired measurements show a complete-operation benefit.
Check the incoming initial state, final-state cancellation, and changed-input graph replay independently.
Keep failed numerical results even when a candidate has a timing benefit.
Use the unchanged contracts before production selection, then repeat framework acceptance with profiling disabled.
