from dataclasses import dataclass

from .native import load


@dataclass(frozen=True)
class Weights:
    norm: object
    projection: object


class Sample:
    def __init__(self, weights, rows, allocate, *, allow_invalid=False):
        self.module, self.source_id, self.library = load(
            "sample", ("sample_workspace", "sample", "mtp_accept")
        )
        self.weights = weights
        self.allow_invalid = allow_invalid
        self.buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
                ((rows, 5120), "bfloat16"),
                ((rows, 248320), "bfloat16"),
                ((rows, 248320), "uint64"),
                ((rows, 248320), "uint64"),
                ((rows, 248320), "float32"),
                ((rows, 248320), "int32"),
                ((rows, 248320), "float32"),
                ((rows,), "int64"),
                ((rows,), "float32"),
                ((max(1, self.module.sample_workspace(rows)),), "uint8"),
            )
        )

    def __call__(self, value, residual, history, mask, parameters, rng):
        self.module.sample(
            value,
            residual,
            self.weights.norm,
            self.weights.projection,
            history,
            mask,
            parameters,
            rng,
            *self.buffers,
            self.allow_invalid,
        )
        return self.buffers[7], self.buffers[8], self.buffers[6]


class MTPAccept:
    def __init__(self, batch, allocate):
        self.module, self.source_id, self.library = load(
            "sample", ("sample_workspace", "sample", "mtp_accept")
        )
        self.buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
                ((batch, 5), "int64"),
                ((batch,), "int32"),
                ((batch,), "int32"),
            )
        )
        self.scratch = (
            allocate((batch,), "float64"),
            allocate((batch, (248320 + 255) // 256), "float32"),
        )

    def __call__(self, target, proposal, drafts, rng, private_slots, capacity):
        self.module.mtp_accept(
            target,
            proposal,
            drafts,
            rng,
            private_slots,
            *self.buffers,
            *self.scratch,
            capacity,
        )
        return self.buffers
