"""Encoded projections with UE8M0 group-32 activation scales.

Callers retain inputs and the workspace through Eager completion. Enter a
graph lease before capture and release it after graph destruction. Weights
are prepared once outside capture; their stored values stay unchanged.
"""

from typing import NamedTuple

FUNCTIONS = (
    "pack_weights",
    "workspace_size",
    "project",
    "grouped_sizes",
    "project_grouped",
    "project_grouped_prepared",
)


class Weights(NamedTuple):
    encoded: object
    scales: object


class Inputs(NamedTuple):
    encoded: object
    scales: object


class Outputs(NamedTuple):
    value: object


class GroupedInputs(NamedTuple):
    encoded: object
    scales: object
    counts: object
    offsets: object


class PreparedGroupedInputs(NamedTuple):
    """Padded A8 rows and per-expert native scale atoms, shared read-only.

    Encoded padding and unused rows are zero. Scale holes are 127. The producer
    must not overwrite storage until every consumer finishes on the caller stream.
    Retain producer storage through Eager completion or graph destruction.
    Kp is ceil(K/128)*128. Encoded shape is [capacity,Kp]. Expert g's scale slab
    starts at (ceil(offsets[g]/128)*128+128*g)*(Kp/32). The flat scale buffer has
    (ceil(capacity/128)*128+128*groups)*(Kp/32) bytes in CUTLASS group-32 atoms.
    """

    encoded: object
    packed_scales: object
    counts: object
    offsets: object


class GroupedOutputs(NamedTuple):
    value: object
    status: object
