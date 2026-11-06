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
    projection: object
    projection_scale: object
    query_norm: object
    key_norm: object
    output: object
    output_scale: object


@dataclass(frozen=True)
class BatchView:
    positions: object
    sequence_rows: object
    lengths: object


@dataclass(frozen=True)
class CacheView:
    codes: object
    scales: object
    slots: object
    tables: object
    write_mask: object = None


def run(weights, workspace, inputs, batch, state):
    return workspace.run(weights, inputs, batch, state)


FUNCTIONS = ("fa_fp8_run", "fa_fp8_copy_pages")
