"""Attention over explicitly selected packed rows, with a denominator-only sink.

Queries are BF16 [rows, heads, 512]; sinks are original F32 [heads].
Window rows use E4M3/UE8M0 groups of 32. Compressed rows use packed
E2M1/E4M3 groups of 16, with the even channel in the low nibble.
Indices are int32 [rows, slots]: -1 masks a slot; window IDs precede
compressed IDs. Repeated IDs contribute repeated mass.

Status is uint8 [rows, heads/16]. A nonzero entry invalidates that head
tile and its output is zero. Invalid IDs and nonfinite computation are
rejected without invalid table reads. Check status before state commit.
Inputs and workspace must remain live through execution or graph release.
Run once before Graph recording for native launch configuration.
"""

from typing import NamedTuple

FUNCTIONS = ("attend",)


class Inputs(NamedTuple):
    query: object
    sink: object
    indices: object
    window: object
    window_scales: object
    compressed: object
    compressed_scales: object


class Outputs(NamedTuple):
    value: object
    status: object
