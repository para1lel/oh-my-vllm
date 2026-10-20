from dataclasses import dataclass
from typing import NamedTuple

FUNCTIONS = (
    "vision_patch",
    "vision_block",
    "vision_merge",
    "vision_classic_prepare",
    "vision_classic_close",
    "vision_classic_images",
    "vision_classic_state",
    "vision_classic_graph_lease",
)


@dataclass(frozen=True)
class Weights:
    blocks: tuple
    patch: tuple
    merger: tuple


@dataclass(frozen=True)
class LayerWeights:
    block: tuple
    patch: tuple
    merger: tuple


@dataclass(frozen=True)
class Inputs:
    value: object


class Outputs(NamedTuple):
    hidden: object


def layer_weights(weights, layer):
    if not 0 <= layer < 27:
        raise ValueError("vision layer must be between zero and 26")
    return LayerWeights(weights.blocks[layer], weights.patch, weights.merger)


def run(weights, workspace, inputs):
    return workspace.run(weights, inputs)
