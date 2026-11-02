"""Retain inputs through Eager completion or graph release.

Logits and biases must be finite, with finite corrected scores. Mask values
are zero for text and one for images. Invalid rows produce NaN coefficients.
"""

from typing import NamedTuple

FUNCTIONS = ("route",)


class Weights(NamedTuple):
    bias: object
    bias_vl: object


class Inputs(NamedTuple):
    logits: object
    image_mask: object


class Outputs(NamedTuple):
    expert_ids: object
    coefficients: object
