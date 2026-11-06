from typing import NamedTuple

from ..fa_fp8.api import Inputs
from ..runtime.native import load
from .api import FUNCTIONS


class Storage(NamedTuple):
    packed: object
    value: object
    zero: object
    partials: object
    empty_media: object


class InputWorkspace:
    def __init__(self, rows, storage, weights, allocate):
        self.module, self.source_id, self.library = load("mtp_fp8", FUNCTIONS)
        self.weights = tuple(weights)
        self.storage = Storage(*(allocate(*field) for field in storage))
        self.outputs = Inputs(self.storage.value, self.storage.zero)
        for weight in self.weights:
            self.module.mtp_fp8_prepare_input(weight.projection, rows)

    def run(self, weights, inputs):
        w, s, i = weights, self.storage, inputs
        if (i.media is None) != (i.media_rows is None):
            raise ValueError(
                "MTP media values and row indices must be supplied together"
            )
        self.module.mtp_fp8_input(
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
