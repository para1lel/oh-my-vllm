"""Inputs stay alive through Eager completion or graph release.

Preparation uses incoming_pre to collapse the residual. Its next_pre belongs
to the following sublayer. Expansion consumes the current sublayer's output.
The branch can be BF16 or a completed FP32 collective result. Expansion rounds
an FP32 branch to BF16 in registers before the existing HC arithmetic.
All input and output tensor pairs must use disjoint storage.

MoE expansion consumes completed, row-aligned routed FP32 and shared BF16 or
FP32 results. It rounds shared to BF16, adds routed in FP32 RN-even, then
rounds the branch to BF16 before the existing expansion. Both BF16 boundaries
remain observable. All rounding uses ties-to-even. Inputs and all arithmetic
intermediates must stay finite and representable in their required types.
The caller completes collectives and checks sublayer status before state commit.
This boundary does not establish tensor-sharded projection equivalence.
"""

from typing import NamedTuple

FUNCTIONS = ("hc_prepare", "hc_expand", "hc_expand_moe", "hc_collapse")


class Weights(NamedTuple):
    projection: object
    scale: object
    base: object


class Inputs(NamedTuple):
    residual: object
    incoming_pre: object
    branch: object


class Outputs(NamedTuple):
    collapsed: object
    expanded: object
    next_pre: object
