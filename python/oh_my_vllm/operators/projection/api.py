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
Dense projection needs no device scratch. Grouped scratch holds device row and
tile offsets, and the compacted nonempty groups.
The `half_k32_kernel` blocks use tiles with 16 or 32 rows and 64 columns.
FP4 and FP8 values convert accurately to FP16 with the native K32 reduction layout.
At most 64 routed rows use parallel raw dots and ordered scale accumulation.
The launch does not use a host expert row bound. Device counts must be within
max_rows and sum to the input rows; otherwise, the output is NaN. Host metadata
and storage overlap failures occur before launch. Grouped calls do not upload
host descriptors.

The padded entry accepts counts whose sum is at most the input capacity.
It reads only this prefix and writes zeros to the output tail. Caller-owned
INT32 status storage has shape (1,). Invalid device counts set an error bit.
Invalid counts or a prior status error zero the complete padded output.
Status persists until the caller resets it at Execute start. Check status at the
step boundary before cache commit. Use the same workspace query with capacity
as its row count. Its capacity <= groups * max_rows condition still applies.
"""

from typing import NamedTuple

FUNCTIONS = (
    "pack_weights",
    "workspace_size",
    "project",
    "grouped_workspace_size",
    "grouped_project",
    "grouped_project_padded",
)


class Weights(NamedTuple):
    encoded: object
    scales: object


class Inputs(NamedTuple):
    encoded: object
    scales: object


class Outputs(NamedTuple):
    value: object
