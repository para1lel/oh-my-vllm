"""Retain inputs through Eager completion or graph release.

Logits and biases must be finite, with finite corrected scores. Mask values
are zero for text and one for images. Invalid rows produce NaN coefficients.

Dispatch rejects a whole row for invalid IDs, duplicate IDs or nonfinite routes.
Unused capacity is zero; source_map uses -1 for absent slots. Combine follows
dispatch on the same caller stream and sums local down outputs by global ID.
Malformed maps, nonfinite active down values and FP32 overflow give zero rows
and status one. Finite BF16 bits are widened exactly before each FP32 RN-even add.
The caller retains down inputs and rejects nonzero status before state commit.
"""

from typing import NamedTuple

FUNCTIONS = ("route", "dispatch", "combine")


class Weights(NamedTuple):
    bias: object
    bias_vl: object


class Inputs(NamedTuple):
    logits: object
    image_mask: object


class Outputs(NamedTuple):
    expert_ids: object
    coefficients: object


class DispatchInputs(NamedTuple):
    values: object
    expert_ids: object
    coefficients: object


class DispatchOutputs(NamedTuple):
    values: object
    coefficients: object
    counts: object
    offsets: object
    source_map: object
    status: object


class CombineOutputs(NamedTuple):
    value: object
    status: object
