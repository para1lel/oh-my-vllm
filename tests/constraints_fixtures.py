from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from oh_my_vllm.models.qwen38.speculation import Draft as RealDraft


def sampling_mask(inputs, index=0):
    return np.unpackbits(inputs.mask[index], bitorder="little")


class ChainDevice:
    def __init__(self):
        self.downloads = 0
        self.module = SimpleNamespace(copy_bytes=self.copy, gather_rows=self.gather)

    def empty(self, shape, dtype):
        if isinstance(dtype, torch.dtype):
            dtype = {
                torch.bfloat16: np.float32,
                torch.float32: np.float32,
                torch.int64: np.int64,
                torch.int32: np.int32,
                torch.uint8: np.uint8,
            }[dtype]
        return np.empty(shape, np.float32 if dtype == "bfloat16" else dtype)

    def scratch(self, shape, dtype):
        return self.empty(shape, dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, np.float32 if dtype == "bfloat16" else dtype)

    def zero(self, value):
        value[...] = 0
        return value

    def upload(self, value):
        return value.copy()

    def view(self, value, shape, offset=0):
        return value.reshape(-1)[offset : offset + int(np.prod(shape))].reshape(shape)

    def copy(self, destination, source):
        np.copyto(
            destination,
            source if isinstance(source, np.ndarray) else np.from_dlpack(source),
        )

    def uploads(self):
        prepared = []

        def prepare(destination, source):
            snapshot = torch.zeros_like(source)
            prepared.append((destination, snapshot))
            return snapshot

        def update(writer, *args):
            writer(*args)
            for destination, source in prepared:
                self.copy(destination, source)

        return SimpleNamespace(
            copy=self.copy, prepare=prepare, update=update, close=lambda: None
        )

    def gather(self, destination, source, indices):
        destination[:] = source[indices]

    def download(self, value, dtype):
        self.downloads += 1
        return value.astype(dtype, copy=True)

    def synchronize(self):
        pass

    @contextmanager
    def graph(self, operation):
        yield operation


class ChainSample:
    def __init__(self, plan, device, weights, *, allocate=None):
        rows = plan.rows
        allocate = device.empty if allocate is None else allocate
        self.outputs = SimpleNamespace(
            hidden=allocate((rows, 5120), "bfloat16"),
            probabilities=allocate((rows, 248320), "float32"),
            tokens=allocate((rows,), "int64"),
        )

    def project(self, weights, inputs):
        self.outputs.hidden[:] = inputs.value + inputs.residual + 1

    def draw(self, settings, rng):
        rng[:, 1] += (settings.parameters[:, 0] > 0).astype(np.uint64)
        self.outputs.tokens[:] = self.outputs.hidden[:, 0].astype(np.int64)
        self.outputs.probabilities[:] = 0
        self.outputs.probabilities[
            np.arange(len(self.outputs.hidden)), self.outputs.tokens
        ] = 1
        return self.outputs

    def advance_presence(self, history):
        for row, token in enumerate(self.outputs.tokens):
            history[row, token // 8] |= 1 << (token % 8)

    def close(self):
        pass

    @contextmanager
    def graph_lease(self):
        yield


class ChainDraft:
    def __init__(self, device, weights, batch, first):
        from oh_my_vllm.models.qwen38.sampling import Sampling

        self.device, self.first = device, first
        self.hidden = device.zeros((batch, 5120), "bfloat16")
        self.tokens = device.zeros((batch,), "int64")
        self.rows = device.upload(np.arange(batch, dtype=np.int32))
        self.ffn = SimpleNamespace(outputs=(self.hidden.copy(), self.hidden.copy()))
        self.output = self.ffn.outputs
        self.sampling = Sampling(device, None, batch, hidden=self.output)
        self.metadata_inputs = SimpleNamespace(
            host_views=tuple(
                np.zeros(shape, dtype=dtype)
                for shape, dtype in (
                    ((3, batch), "int64"),
                    ((batch,), "int32"),
                    ((batch, 335), "int32"),
                    ((batch + 1,), "int32"),
                    ((batch,), "int32"),
                    ((batch,), "int64"),
                )
            ),
            update=self.update_metadata,
            close=lambda: None,
        )

    def update_metadata(self, arrays):
        self.metadata_arrays = tuple(np.array(value, copy=True) for value in arrays)
        self.tokens[:] = arrays[-1]

    def row_views(self):
        if not hasattr(self, "_row_bindings"):
            self._row_bindings = self._row_views = None
        return RealDraft.row_views(self)

    def run(self):
        self.output[0][:] = self.hidden + (0 if self.first else self.tokens[:, None])
        self.output[1][:] = 0
        self.sampling.project()
        return self.output

    def execute(self, graph):
        self.run()

    def copy(self, destination, source):
        self.device.copy(destination, source)

    @contextmanager
    def graph_lease(self):
        yield

    def close(self):
        self.sampling.close()


class AcceptDevice(ChainDevice):
    def __init__(self):
        super().__init__()
        self.module = SimpleNamespace(
            copy_bytes=self.copy, gather_rows=lambda *arguments: None
        )

    def download(self, value, dtype):
        if dtype is not None:
            return super().download(value, dtype)
        self.downloads += 1
        return torch.as_tensor(value)


class PackedAcceptance:
    def __init__(self, kept):
        self.greedy = False
        self.capacity = None
        self.proposal_rows = ("proposal",)
        self.gather, self.drafts = "gather", "drafts"
        self.slots, self.rng = "slots", "rng"
        self.target_token_rows = None
        self.copied = {}
        self.metadata_inputs = SimpleNamespace(update=self.update_metadata)
        self.operation = SimpleNamespace(outputs=SimpleNamespace(tokens=kept))

    def update_metadata(self, values):
        self.copied.update(
            zip(("gather", "drafts", "slots", "rng"), values, strict=True)
        )

    def copy(self, destination, source):
        self.copied[destination] = source

    def execute(self, graph):
        pass

    def close(self):
        pass
