"""CUDA factories reject stale and invalid launch configuration."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.kernels import (
    cuda_backend,
)


class FakeCudaTensor:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.is_cuda = True

    def __len__(self):
        return self.shape[0]

    def numel(self):
        return prod(self.shape)

    def stride(self, dim):
        return prod(self.shape[dim + 1 :])

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 16

    def index_select(self, dim, indices):
        assert dim == 0
        return FakeCudaTensor((len(indices), *self.shape[1:]), self.dtype)


@pytest.mark.parametrize(
    ("module", "name", "live", "stale"),
    [
        ("fp8", "_quantize", (False, True), (128, "bfloat16", False, True, 16)),
        ("elementwise", "_silu_mul", (), (128, 1024)),
        ("convolution", "_conv", (), (10240, 10240, 2, 1, ("int32",) * 3, 8)),
        (
            "gdn",
            "_recurrent",
            (),
            (1, 2, 16, 48, 16, (2048, 2048, 6144), "float32", ("int32",) * 3),
        ),
        (
            "decode_attention",
            "_partials",
            (128, 0, False, "int32"),
            (16, 16, 2, 84, 128, 0, 32, 16, False, ("int32",) * 3, "int32"),
        ),
        ("decode_attention", "_merge", (), (16, 128)),
    ],
)
def test_cuda_factory_rejects_discarded_configuration(
    monkeypatch, module, name, live, stale
):
    fake = SimpleNamespace(
        **{entry: lambda *args: None for entry in cuda_backend._FUNCTIONS}
    )
    monkeypatch.setattr(cuda_backend, "compiled", lambda: fake)
    factory = cuda_backend.factory_for(module, name)
    assert callable(factory(*live))
    with pytest.raises(TypeError):
        factory(*stale)


def test_cuda_partial_factory_keeps_live_split_and_position_contract(monkeypatch):
    calls = []
    fake = SimpleNamespace(attention_partial=lambda *args: calls.append(args))
    monkeypatch.setattr(cuda_backend, "compiled", lambda: fake)
    launch = cuda_backend.factory_for("decode_attention", "_partials")(
        16, 1, True, "int64"
    )
    partial = SimpleNamespace(ndim=4, shape=(1, 16, 16, 256))
    lse = SimpleNamespace(ndim=3, shape=(1, 16, 16))
    launch("q", "cache", "tables", "lengths", "starts", partial, lse)
    assert calls == [
        ("q", "cache", "tables", "lengths", "starts", partial, lse, 1, True, True, 5)
    ]
    lse.shape = (1, 16, 15)
    with pytest.raises(ValueError, match="split count"):
        launch("q", "cache", "tables", "lengths", "starts", partial, lse)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="position dtype"):
        cuda_backend.factory_for("decode_attention", "_partials")(
            16, 1, True, "float32"
        )
