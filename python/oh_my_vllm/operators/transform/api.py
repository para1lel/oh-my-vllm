"""RMS normalization, tail rotation, or exact BF16-to-FP32 widening.

RMS parameters contain original BF16 weights, without an offset. RoPE parameters
contain FP32 cosine/sine pairs for each row, shared by its attention heads.
RMS values have shape (rows, width); RoPE values have shape (rows, heads, width).
RMS and RoPE inputs must be finite and bounded
so FP32 intermediate values and BF16 outputs remain finite.
Retain inputs through Eager completion or graph release.

Startup rope_table consumes 32 finite FP64 inverse frequencies. It produces
FP32 cosine/sine pairs for positions [0, length), with length at most 1048576.
Frequency and phase arithmetic use FP64. Prepare tables outside Graph recording.

Widening uses [rows, 5120] BF16 values and no parameters. Finite values widen
exactly, including signed zeros and subnormals. Each nonfinite element becomes
positive zero and sets its row status to 1. Every call resets status; reject
nonzero status before state commit.
"""

from typing import NamedTuple

FUNCTIONS = ("rms", "rope", "widen", "rope_table")


class Inputs(NamedTuple):
    values: object
    parameters: object


class Outputs(NamedTuple):
    value: object


class WidenOutputs(NamedTuple):
    value: object
    status: object
