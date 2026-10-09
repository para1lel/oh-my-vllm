"""Every formal operation builds callables without invoking a CUDA kernel."""

from functools import wraps

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from development.kernels import fixtures

CONFIGURATIONS = {
    "prepare_attention": {"tokens": 2, "index_dtype": "int64"},
    "norm": {"tokens": 2},
    "add_norm": {"tokens": 2},
    "add_norm_fp8_linear": {"tokens": 2, "columns": 34816},
    "fp8_linear": {"tokens": 624, "width": 17408, "columns": 5120, "silu": True},
    "gated_norm": {"tokens": 2},
    "norm_rope": {"tokens": 2, "heads": 24},
    "quant": {"tokens": 2, "width": 5120, "column": True},
    "silu_quant": {"tokens": 2, "width": 17408, "column": True},
    "gates": {"tokens": 2},
    "qk": {"tokens": 2},
    "recurrent": {"counts": (2, 1), "state_dtype": "bfloat16"},
    "convolution": {"counts": (2, 1)},
    "append": {"tokens": 2},
    "attention": {
        "batch": 1,
        "queries": 8,
        "length": 784,
        "first": 0,
        "grouped": True,
        "table_dtype": "int32",
        "length_dtype": "int64",
        "max_tokens": 1568,
    },
    "dspark_rms_norm": {"tokens": 7},
    "dspark_norm_rope": {
        "tokens": 7,
        "heads": 32,
        "position_base": 262143,
        "draft": True,
    },
    "dspark_append": {"tokens": 8},
    "dspark_attention": {"batch": 1, "length": 784},
}


@pytest.mark.parametrize("operation", sorted(CONFIGURATIONS))
def test_every_formal_fixture_builds_complete_operation_callables(
    operation, monkeypatch
):
    assert set(CONFIGURATIONS) == set(fixtures.ENTRIES)
    monkeypatch.setattr(
        fixtures, "reference_operator", lambda *_: lambda *args, **kwargs: None
    )
    mode = FakeTensorMode()

    def on_cpu(factory):
        @wraps(factory)
        def create(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return create

    # Fake CPU allocations permit the real static fixture shapes without
    # allocating pools, touching a CUDA context, or invoking either backend.
    class FixtureTorch:
        def __getattr__(self, name):
            if name == "Generator":
                return lambda **_: torch.Generator(device="cpu")
            if name in ("randn", "rand", "arange", "zeros", "empty", "tensor", "full"):
                return on_cpu(getattr(torch, name))
            return getattr(torch, name)

    from oh_my_vllm.models import dspark

    yarn = dspark.yarn_frequencies
    monkeypatch.setattr(
        dspark, "yarn_frequencies", lambda config, _: yarn(config, "cpu")
    )
    monkeypatch.setattr(fixtures, "torch", FixtureTorch())
    with mode:
        reference, candidate, witnesses = fixtures.fixture(
            {"operation": operation, **CONFIGURATIONS[operation]}, observe=True
        )
    assert callable(reference) and callable(candidate)
    if operation in (
        "prepare_attention",
        "convolution",
        "recurrent",
        "append",
        "dspark_append",
    ):
        assert witnesses and all(
            callable(old) and callable(new) for _, old, new in witnesses
        )
