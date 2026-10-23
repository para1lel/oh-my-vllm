from dataclasses import dataclass
from typing import NamedTuple

FUNCTIONS = ("engram_hash",)


@dataclass(frozen=True)
class Inputs:
    tokens: object
    history: object
    multipliers: object
    primes: object
    offsets: object


class Outputs(NamedTuple):
    hashes: object
    history: object


def hash_rows(workspace, inputs):
    return workspace.run(inputs)
