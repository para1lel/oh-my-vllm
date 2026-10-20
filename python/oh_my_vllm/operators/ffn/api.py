from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Inputs:
    value: object
    residual: object


class Outputs(NamedTuple):
    value: object
    residual: object


@dataclass(frozen=True)
class Weights:
    norm: object
    gate_up: object
    gate_up_scale: object
    down: object
    down_scale: object


def run(weights, workspace, inputs):
    return workspace.run(weights, inputs)


FUNCTIONS = ("ffn",)
