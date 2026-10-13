"""Formal operator inputs must be reproducible independent of caller RNG state."""

import pytest
import torch

from development.kernels import fixtures

pytestmark = pytest.mark.gpu


def test_fixture_seed_is_local_and_reproducible(monkeypatch):
    monkeypatch.setattr(
        fixtures, "reference_operator", lambda *_: lambda *args, **_: args
    )
    config = {"operation": "gates", "tokens": 2}
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state()
    first, _ = fixtures.fixture(config, seed=784)
    second, _ = fixtures.fixture(config, seed=784)
    different, _ = fixtures.fixture(config, seed=785)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    first_args, second_args, different_args = first(), second(), different()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(784)
        old_first_input = torch.randn(2, 96, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(first_args[0], old_first_input, rtol=0, atol=0)
    for a, b in zip(first_args, second_args, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(
        not torch.equal(a, b) for a, b in zip(first_args, different_args, strict=True)
    )
