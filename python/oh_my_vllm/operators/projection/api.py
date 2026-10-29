"""Encoded projections with UE8M0 group-32 activation scales.

Callers retain inputs and the workspace through Eager completion. Enter a
graph lease before capture and release it after graph destruction. Weights
are prepared once outside capture; their stored values stay unchanged.
Dense and grouped project accept BF16 or FP32 output storage. FP32 output keeps
partial sums for a later rank reduction. Each workspace query covers both
output types.
Dense and grouped projections reset the raw FP32 dot for each K32 group.
The activation scale uses MUL. The weight scale uses FMA into the running
sum. FP4 values convert accurately to FP8 before the raw dot. FP4 weight scales
apply to each output row; FP8 weight scales apply to 32-row groups.
Dense projection needs no device scratch. Grouped scratch holds device row
offsets, rounded to 16 bytes. Device counts must be within max_rows and sum
to the input rows; otherwise, the output is NaN. Host metadata and storage overlap
failures occur before launch. Grouped calls do not upload host descriptors.
"""

from typing import NamedTuple

FUNCTIONS = (
    "pack_weights",
    "workspace_size",
    "project",
    "grouped_workspace_size",
    "grouped_project",
)


class Weights(NamedTuple):
    encoded: object
    scales: object


class Inputs(NamedTuple):
    encoded: object
    scales: object


class Outputs(NamedTuple):
    value: object
