from dataclasses import dataclass
from typing import NamedTuple

FUNCTIONS = (
    "sample_workspace",
    "sample_project",
    "sample_distribution",
    "sample",
    "final_hidden",
)


@dataclass(frozen=True)
class Weights:
    norm: object
    projection: object = None


@dataclass(frozen=True)
class Inputs:
    value: object
    residual: object


@dataclass(frozen=True)
class HiddenState:
    cache: object
    slots: object


@dataclass(frozen=True)
class Settings:
    history: object
    mask: object
    parameters: object
    greedy: bool = False


class Outputs(NamedTuple):
    hidden: object
    logits: object = None
    probabilities: object = None
    tokens: object = None
    selected_probability: object = None


def hidden(weights, workspace, inputs, state):
    return workspace.hidden(weights, inputs, state)


def distribution(weights, workspace, inputs, settings):
    workspace.project(weights, inputs)
    workspace.distribution(settings)
    return workspace.outputs


def sample(weights, workspace, inputs, settings, rng):
    workspace.project(weights, inputs)
    workspace.draw(settings, rng)
    return workspace.outputs
