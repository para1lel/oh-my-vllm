"""Retain inputs through Eager completion or graph release.

Logits and biases must be finite, with finite corrected scores. Mask values
are zero for text and one for images. Invalid rows produce NaN coefficients.

Expert sums consume BF16 projection rows in increasing expert order. The padded
sum uses device counts for the live prefix of values and sorted route positions.
Its INT32 map has shape (token_capacity, topk), with at most six route slots.
Unused token rows get zero FP32 output. Invalid counts, positions, or duplicate
slots set caller-owned INT32 status storage of shape (1,). Status persists until
the caller resets it at Execute start. A nonzero status prevents values reads
and zeros the complete output. Check it at the step boundary before cache commit.
"""

from typing import NamedTuple

FUNCTIONS = ("route", "expert_sum", "expert_sum_padded")


class Weights(NamedTuple):
    bias: object
    bias_vl: object


class Inputs(NamedTuple):
    logits: object
    image_mask: object


class Outputs(NamedTuple):
    expert_ids: object
    coefficients: object
