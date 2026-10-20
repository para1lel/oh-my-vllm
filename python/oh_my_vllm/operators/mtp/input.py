from typing import NamedTuple

from ..fa.api import Inputs
from ..runtime.native import load
from .api import FUNCTIONS


class Storage(NamedTuple):
    packed: object
    value: object
    zero: object
    partials: object
    empty_media: object


class InputWorkspace:
    def __init__(self, storage, weights, allocate):
        self.module, self.source_id, self.library = load("mtp", FUNCTIONS)
        self.weights = tuple(weights)
        self.storage = Storage(*(allocate(*field) for field in storage))
        self.outputs = Inputs(self.storage.value, self.storage.zero)

    def run(self, weights, inputs):
        w, s, i = weights, self.storage, inputs
        self.module.mtp_input(
            i.hidden,
            i.tokens,
            w.embedding,
            w.hidden_norm,
            w.embedding_norm,
            w.projection,
            i.hidden_rows,
            i.boundary_hidden,
            i.boundary_rows,
            s.empty_media if i.media is None else i.media,
            i.hidden_rows if i.media_rows is None else i.media_rows,
            i.media is not None,
            s.packed,
            s.value,
            s.zero,
            s.partials,
        )
        return self.outputs
