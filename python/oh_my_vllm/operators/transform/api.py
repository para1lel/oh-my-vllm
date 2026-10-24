"""RMS normalization, tail rotation, or exact BF16-to-FP32 widening.

RMS parameters contain original BF16 weights, without an offset. RoPE parameters
contain FP32 cosine/sine pairs for each row, shared by its attention heads.
RMS values have shape (rows, width); RoPE values have shape (rows, heads, width).
RMS and RoPE inputs must be finite and bounded
so FP32 intermediate values and BF16 outputs remain finite.
Retain inputs through Eager completion or graph release.

Widening uses [rows, 5120] BF16 values and no parameters. Finite values widen
exactly, including signed zeros and subnormals. Each nonfinite element becomes
positive zero and sets its row status to 1. Every call resets status; reject
nonzero status before state commit.
"""

from typing import NamedTuple

FUNCTIONS = ("rms", "rope", "widen")


class Inputs(NamedTuple):
    values: object
    parameters: object


class Outputs(NamedTuple):
    value: object


class WidenOutputs(NamedTuple):
    value: object
    status: object
