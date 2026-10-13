"""CPU contracts for DSpark's rounding, full rotation and block visibility."""

import math

import pytest
import torch
from oh_my_vllm.ir.dspark import _reference_attention, _reference_norm, _reference_rms


def test_hidden_rms_rounds_before_checkpoint_multiplier():
    generator = torch.Generator().manual_seed(5120)
    x = torch.randn(2, 5120, generator=generator).bfloat16()
    weight = torch.linspace(0.3, 1.7, 5120)
    values = x.double()
    normalized = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    expected = (normalized * weight.bfloat16()).bfloat16()
    actual = _reference_rms(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A single final BF16 cast omits an observable checkpoint rounding stage.
    single_round = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        * weight.double()
    ).bfloat16()
    assert not torch.equal(actual, single_round)


@pytest.mark.parametrize("position", [0, 8192, 262143])
def test_full_rotary_identity_frequency_and_checkpoint_rounding(position):
    x = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128).bfloat16() / 128
    weights = torch.linspace(0.5, 1.5, 128).bfloat16().float()
    positions = torch.tensor([position], dtype=torch.int64)
    frequency = torch.zeros(64)
    expected = (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    expected = (expected * weights.bfloat16()).bfloat16()
    actual = _reference_norm(x, weights, positions, frequency, 1.0, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rotated = _reference_norm(
        x, weights, torch.tensor([1]), torch.full((64,), math.pi / 2), 1.0, 1e-6
    )
    torch.testing.assert_close(
        rotated[..., :64], -expected[..., 64:], rtol=0.01, atol=0.01
    )
    torch.testing.assert_close(
        rotated[..., 64:], expected[..., :64], rtol=0.01, atol=0.01
    )


def test_all_queries_see_all_noise_rows_without_writing_context():
    query = torch.zeros(2, 7, 32, 128, dtype=torch.bfloat16)
    cache = torch.full((3, 2, 784, 8, 128), float("nan"), dtype=torch.bfloat16)
    cache[1, :, :2] = 0
    cache[1, 1, 0] = 7
    cache[1, 1, 1] = 14
    table = torch.tensor([[1], [2]], dtype=torch.int32)
    lengths = torch.tensor([2, 0], dtype=torch.int32)
    key = torch.zeros(2, 7, 8, 128, dtype=torch.bfloat16)
    value = (
        torch.arange(7, dtype=torch.float32)
        .bfloat16()[None, :, None, None]
        .expand(2, 7, 8, 128)
        .contiguous()
    )
    before = cache.clone()
    output = _reference_attention(query, cache, table, lengths, key, value)
    torch.testing.assert_close(
        output[0], torch.full_like(output[0], 42 / 9), rtol=0, atol=0
    )
    torch.testing.assert_close(output[1], torch.full_like(output[1], 3), rtol=0, atol=0)
    torch.testing.assert_close(cache, before, rtol=0, atol=0, equal_nan=True)
