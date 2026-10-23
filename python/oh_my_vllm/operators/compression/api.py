"""Complete ratio2 FP32 KV/gate pairs pool to BF16 before normalization.

Inputs are contiguous [rows, 2, 512]. Softmax acts on the pair independently
for each channel. The finite numerical domain is KV [-64, 64], gates [-128, 128].
There is no projection, normalization, rotary transform or partial-group state.
Retain inputs and workspace through Eager completion or graph release.
Recording and replay use the same caller stream.
"""

from typing import NamedTuple

FUNCTIONS = ("pool_ratio2",)


class Inputs(NamedTuple):
    kv: object
    gates: object


class Outputs(NamedTuple):
    value: object
