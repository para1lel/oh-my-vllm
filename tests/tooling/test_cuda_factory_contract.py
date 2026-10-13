"""CUDA factories accept only live configuration; frozen TileLang calls stay intact."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.kernels import (
    convolution,
    cuda_backend,
    decode_attention,
    elementwise,
    fp8,
    gdn,
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


@pytest.mark.parametrize("selected", ["cuda", "tilelang"])
def test_public_wrappers_keep_backend_specific_factory_arguments(monkeypatch, selected):
    calls = {}

    def capture(key):
        def factory(*config):
            calls[key] = config
            return lambda *args: None

        return factory

    for module, name in (
        (fp8, "_quantize"),
        (elementwise, "_silu_mul"),
        (convolution, "_conv"),
        (gdn, "_recurrent"),
        (decode_attention, "_partials"),
        (decode_attention, "_merge"),
    ):
        monkeypatch.setattr(module, "NAME", selected)
        monkeypatch.setattr(module, name, capture(name))

    fp8.quantize(FakeCudaTensor((128, 256)))
    elementwise.silu_mul(FakeCudaTensor((128, 256)))
    x = FakeCudaTensor((128, 10240))
    weight = FakeCudaTensor((10240, 4))
    pool = FakeCudaTensor((2, 10240, 3))
    ids = FakeCudaTensor((128,), torch.int32)
    starts = FakeCudaTensor((2,), torch.int32)
    reads = FakeCudaTensor((1,), torch.int32)
    writes = FakeCudaTensor((128,), torch.int32)
    convolution.causal_conv(x, weight, pool, ids, starts, reads, writes)
    q = FakeCudaTensor((1, 16, 128))
    v = FakeCudaTensor((1, 48, 128))
    gate = FakeCudaTensor((1, 48), torch.float32)
    state = FakeCudaTensor((2, 48, 128, 128), torch.float32)
    gdn.recurrent(
        q, q, v, gate, gate, state, starts, reads, FakeCudaTensor((1,), torch.int32)
    )
    monkeypatch.setattr(
        decode_attention.torch,
        "empty_like",
        lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype),
    )
    decode_attention.decode(
        FakeCudaTensor((1, 16, 256)),
        FakeCudaTensor((2, 2, 784, 16, 256)),
        FakeCudaTensor((1, 84), torch.int32),
        FakeCudaTensor((1,), torch.int32),
    )
    if selected == "cuda":
        assert calls == {
            "_quantize": (False, False),
            "_silu_mul": (),
            "_conv": (),
            "_recurrent": (),
            "_partials": (128, 0, False, "int32", 5),
            "_merge": (),
        }
    else:
        assert calls == {
            "_quantize": (256, "bfloat16", False, False, 16),
            "_silu_mul": (128, 1024),
            "_conv": (10240, 10240, 2, 1, ("int32",) * 3, 8),
            "_recurrent": (
                1,
                2,
                16,
                48,
                16,
                (2048, 2048, 6144),
                "float32",
                ("int32",) * 3,
            ),
            "_partials": (
                16,
                16,
                2,
                84,
                128,
                0,
                32,
                16,
                False,
                ("int32",) * 3,
                "int32",
            ),
            "_merge": (16, 128),
        }
