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
