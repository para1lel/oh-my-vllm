"""Inputs stay alive through Eager completion or graph release.

Preparation uses incoming_pre to collapse the residual. Its next_pre belongs
to the following sublayer. Expansion consumes the current sublayer's output.
Sinkhorn mixing adds epsilon after the initial row softmax and before the
first column normalization.
"""

from typing import NamedTuple

FUNCTIONS = ("hc_prepare", "hc_expand")


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
