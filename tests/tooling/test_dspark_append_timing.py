"""Append timing shares addresses only after independent output verification."""

from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.ir.dspark import _reference_append

from development.kernels import fixtures
from development.kernels.output_verification import verify


class _CpuAllocations:
    """Exercise real fixture storage without starting a CUDA context."""

    def __getattr__(self, name):
        if name == "Generator":
            return lambda **_: torch.Generator(device="cpu")
        if name in ("randn", "arange", "empty"):

            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                result = getattr(torch, name)(*args, **kwargs)
                return result.fill_(-17) if name == "empty" else result

            return allocate
        if name == "empty_like":
            return lambda tensor: torch.empty_like(tensor).fill_(-17)
        return getattr(torch, name)


def _providers(monkeypatch, reference, candidate):
    monkeypatch.setattr(fixtures, "torch", _CpuAllocations())
    monkeypatch.setattr(fixtures, "reference_operator", lambda *_: reference)
    monkeypatch.setattr(
        fixtures.importlib, "import_module", lambda _: SimpleNamespace(append=candidate)
    )


@pytest.mark.parametrize("shared", [False, True])
def test_append_timing_address_and_repeated_write_contract(monkeypatch, shared):
    calls = []

    def append(cache, key, value, slots):
        calls.append((cache, key.clone(), value.clone(), slots.clone()))
        _reference_append(cache, key, value, slots)

    _providers(monkeypatch, append, append)
    reference, candidate = fixtures.fixture(
        {"operation": "dspark_append", "tokens": 8},
        shared_append_destination=shared,
    )
    reference()
    expected = calls[0][0].clone()
    candidate()
    assert (calls[0][0].data_ptr() == calls[1][0].data_ptr()) is shared
    torch.testing.assert_close(calls[1][0], expected, atol=0, rtol=0)
    for left, right in zip(calls[0][1:], calls[1][1:], strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_independent_append_verification_detects_missing_writes(monkeypatch):
    _providers(monkeypatch, _reference_append, lambda *_: None)
    reference, candidate, witnesses = fixtures.fixture(
        {"operation": "dspark_append", "tokens": 8}, observe=True
    )
    with pytest.raises(AssertionError, match="written_cache"):
        verify("dspark_append", reference, candidate, witnesses)


@pytest.mark.parametrize(
    "configuration,observe",
    [
        ({"operation": "dspark_append", "tokens": 8}, True),
        ({"operation": "recurrent", "counts": (1,)}, False),
    ],
)
def test_shared_destination_rejects_verification_and_other_mutations(
    configuration, observe
):
    with pytest.raises(ValueError, match="only for timing"):
        fixtures.fixture(configuration, observe=observe, shared_append_destination=True)
