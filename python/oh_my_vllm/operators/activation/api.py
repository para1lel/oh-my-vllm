"""SwiGLU with BF16 gate/up and optional FP32 per-row routing weights.

Text uses an upper gate clamp and a symmetric up clamp at ten. Vision has no
clamp or route. FP32 arithmetic rounds once to BF16 before output projection.
Finite inputs cover gate [-80, 80], up [-32, 32], and route [0, 1.5].
Retain inputs through Eager completion or graph release.
"""

from typing import NamedTuple

FUNCTIONS = ("swiglu",)


class Inputs(NamedTuple):
    gate: object
    up: object
    route: object = None


class Outputs(NamedTuple):
    value: object
