# HTTP reference assumptions

Version `http-reference-v1` uses the [canonical schedule](performance.md#lower-bound-calculation) and [nominal rates](execution.md#http-performance).
This is a conservative resource estimate, not a minimum physical latency proof.
The [calculation source](../development/serving/accounting.py) specifies the dimensions and formulas.

## Memory

Count matrix weights once per logical projection pass, shared across its batch.
FP8 matrices use one byte per element and FP32 scales for each 128-by-128 region.
BF16 matrices use two bytes per element.
Count each normalization vector once per pass.
Intermediate hidden states, logits, and activation buffers have no memory credit.

FA reads prior KV history once per request and pass.
New query KV can pass directly to attention without another HBM read.
Count accepted KV writes once, with 4 KV heads, width 256, and BF16 K and V.
MTP draft attention starts at position 1. The first draft reads its cached context endpoint.
Subsequent drafts compute new KV.

GDN reads one BF16 recurrent and convolution state when a request starts above position zero.
Count unique 784-token checkpoints and the accepted endpoint in that pass.
A checkpoint at the endpoint counts once.
Verification advances by the accepted proposal count plus one.
Each GDN layer has 48 recurrent heads of 128-by-128 elements and 10240 convolution channels with 3 stored values.

Private candidate writes and subsequent state copies have no credit.
This omission does not remove the private-state correctness checks.

## Arithmetic

Matrix products count two operations per multiply-add at their specified operand precision.
The calculation covers all 64 target layers and the full vocabulary projection.
It counts all verification rows, also when some proposals are rejected.
FA counts the full causal attention pairs and three FP32 operations per pair and query head.
These scalar operations represent score scaling, probability summation, and probability normalization.

GDN prefill uses fixed 64-row chunks, split at 784-token checkpoints.
Its reference block recurrence counts BF16 products and FP32 diagonal inversion and state updates.
Decode and verification use the fixed FP32 recurrent formula.
Runtime algorithm changes cannot change this reference choice.

RMS normalization counts four operations per element and two scalar operations per row.
Residual addition counts one operation per element.
The FFN activation counts two products per intermediate element.
GDN convolution counts seven operations per token and channel.

FA head normalization uses width 256. Rotary products use the specified 64 dimensions.
The attention output gate counts one product per element.
MTP context, draft, and vocabulary projections follow their recorded causal positions.

## Omissions

The HTTP cases use greedy sampling without penalties.
Selection comparisons, sorting, integer indexing, random generation, and greedy MTP acceptance have no floating-point credit.
The estimate does not include sigmoid evaluation, transcendental instructions, datatype conversion, quantization, and scale application arithmetic.
It also does not include GDN head normalization, gates, and input normalization arithmetic beyond the listed terms.
There is no empirical throughput factor for operations that have no credit.

The trace does not include all input token IDs for embedding-region deduplication.
Embedding memory reads therefore have no credit.
Checkpoint hidden storage, MTP endpoint storage, metadata, and returned token traffic also have no credit.
These conservative exclusions can reduce reported efficiency.
They cannot increase it when an implementation adds work.

Vision and initialization do not occur inside these text-only HTTP timers.
Their task coverage and correctness are independent final acceptance conditions.
Changes to these assumptions must change the accounting version and invalidate earlier performance evidence.
