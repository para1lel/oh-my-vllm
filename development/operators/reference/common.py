import numpy as np
import torch

from development.model.reference import Weights as CheckpointWeights

from ..bindings import weight_names


class Weights(CheckpointWeights):
    def __init__(self, path):
        super().__init__(path)
        self.cache = {}
        self.names = ()

    def select(self, case):
        self.names = weight_names(case)
        self.cache.clear()
        for name in self.names:
            self.checkpoint.raw_tensor(name)

    @property
    def identity(self):
        files = {"config", "index"} | {
            self.checkpoint.index[name] for name in self.names
        }
        return {
            "files": {name: self.checkpoint.identity[name] for name in sorted(files)},
            "weights": self.names,
        }

    def tensor(self, name, rows=None):
        if name not in self.names:
            raise ValueError(
                f"reference used a tensor outside the case binding: {name}"
            )
        if rows is not None:
            return super().tensor(name, rows)
        if name not in self.cache:
            self.cache[name] = super().tensor(name)
        return self.cache[name]


def rms(value, weight, *, offset=True):
    return (
        value
        * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
        * (weight + int(offset))
    )


def linear(value, weights, prefix, *, bias=False):
    result = value @ weights.tensor(prefix + ".weight").T
    return result + weights.tensor(prefix + ".bias") if bias else result


def tensor(array, dtype, device="cuda"):
    raw = torch.from_numpy(np.array(array, copy=True)).to(device)
    if dtype == "bfloat16":
        return raw.view(torch.bfloat16).to(torch.float64)
    if dtype == "float8_e4m3fn":
        return raw.view(torch.float8_e4m3fn).to(torch.float64)
    return raw.to(torch.float64) if raw.is_floating_point() else raw


def arrays(values):
    return {
        name: value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
        for name, value in values.items()
    }
