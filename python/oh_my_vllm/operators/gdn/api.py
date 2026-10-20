from dataclasses import dataclass
from typing import NamedTuple

from .schedule import Schedule as Schedule


@dataclass(frozen=True)
class Inputs:
    value: object
    residual: object
    tokens: object = None
    media: object = None
    media_rows: object = None


class Outputs(NamedTuple):
    value: object
    residual: object


@dataclass(frozen=True)
class Weights:
    norm: object
    qkvz: object
    qkvz_scale: object
    ba: object
    convolution: object
    a_log: object
    dt_bias: object
    head_norm: object
    output: object
    output_scale: object
    embedding: object = None


@dataclass(frozen=True)
class BatchView:
    sequence_rows: object
    read_slots: object
    write_slots: object


@dataclass(frozen=True)
class StateView:
    recurrent: object
    convolution: object


def run(weights, workspace, inputs, batch, state):
    return workspace.run(weights, inputs, batch, state)


FUNCTIONS = ("gdn",)
