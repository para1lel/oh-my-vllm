"""Finite BF16/F32 projections without bias, quantization, or activation.

One caller uses the preparation stream. Retain inputs and the workspace through
Eager completion or graph destruction. Hold a graph lease through graph destruction.
"""

from typing import NamedTuple

FUNCTIONS = (
    "linear_prepare",
    "linear_graph_lease",
    "linear_project",
    "linear_images",
)


class Weights(NamedTuple):
    value: object


class Inputs(NamedTuple):
    value: object


class Outputs(NamedTuple):
    value: object
