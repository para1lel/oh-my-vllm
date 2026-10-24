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

from ..projection.api import PreparedGroupedInputs

FUNCTIONS = ("route", "dispatch", "combine", "dispatch_encoded")


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


class EncodedDispatchInputs(NamedTuple):
    """Exact FP8/UE8M0 outputs: encoded[rows,5120], scales/status[rows,160].

    Any group status rejects its whole row before expert ownership filtering.
    Codes/scales come from the finite quantizer; dispatch does not re-quantize.
    Reuse outputs.grouped read-only for both w1/w3 before the next dispatch.
    """

    encoded: object
    scales: object
    status: object
    expert_ids: object
    coefficients: object


class EncodedDispatchOutputs(NamedTuple):
    encoded: object
    packed_scales: object
    coefficients: object
    counts: object
    offsets: object
    source_map: object
    status: object

    @property
    def grouped(self):
        return PreparedGroupedInputs(
            self.encoded, self.packed_scales, self.counts, self.offsets
        )


class CombineOutputs(NamedTuple):
    value: object
    status: object
