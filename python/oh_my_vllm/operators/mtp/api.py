from dataclasses import dataclass
from typing import NamedTuple

FUNCTIONS = ("mtp_input", "mtp_endpoints", "mtp_context", "mtp_accept")


@dataclass(frozen=True)
class InputWeights:
    embedding: object
    hidden_norm: object
    embedding_norm: object
    projection: object


@dataclass(frozen=True)
class Weights:
    input: InputWeights
    attention: object
    ffn: object = None
    output: object = None


@dataclass(frozen=True)
class InputView:
    hidden: object
    tokens: object
    hidden_rows: object
    boundary_hidden: object
    boundary_rows: object
    media: object = None
    media_rows: object = None


@dataclass(frozen=True)
class Inputs:
    input: InputView
    batch: object
    state: object
    first: object = None


class ContextOutputs(NamedTuple):
    value: object
    residual: object


@dataclass(frozen=True)
class AcceptInputs:
    target: object
    proposal: object
    drafts: object
    rng: object
    private_slots: object
    capacity: int


class AcceptOutputs(NamedTuple):
    tokens: object
    counts: object
    candidate_slots: object


def context(weights, workspace, inputs):
    return workspace.run(weights, inputs)


def draft(weights, workspace, inputs, settings, rng):
    workspace.project(weights, inputs)
    workspace.output.draw(settings, rng)
    return workspace.outputs


def accept(workspace, inputs):
    return workspace.run(inputs)
