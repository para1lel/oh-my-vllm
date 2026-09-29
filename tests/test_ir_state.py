"""The IR cache-write contract matches its executable PyTorch reference."""

import re

import pytest
import torch
from oh_my_vllm.ir import compile_forward, operations, state


def test_attention_prepare_mutation_schema_and_reference():
    schema = str(operations()["prepare_attention"].target._schema)
    assert re.search(r"Tensor\(a\d+!\) cache", schema)
    packed = torch.randn(2, 14336, dtype=torch.bfloat16)
    qw, kw = torch.ones(256), torch.ones(256)
    positions = torch.tensor([0, 262143], dtype=torch.int64)
    slots = torch.tensor([-1, 785], dtype=torch.int32)
    cache = torch.full((2, 2, 784, 4, 256), 0.125, dtype=torch.bfloat16)
    before = cache.clone()
    q = state._native_prepare(packed, qw, kw, positions, cache, slots)
    assert q.shape == (2, 24, 256)
    torch.testing.assert_close(cache[0], before[0], rtol=0, atol=0)
    assert not torch.equal(cache[1, :, 1], before[1, :, 1])
    torch.testing.assert_close(cache[1, :, 2:], before[1, :, 2:], rtol=0, atol=0)
    result = torch.library.opcheck(
        operations()["prepare_attention"].reference,
        (packed, qw, kw, positions, before, slots),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_attention_prepare_compiled_cache_effect_matches_cuda_entry():
    from oh_my_vllm.kernels.attention_prepare import prepare_attention as old_prepare

    packed = torch.randn(2, 14336, device="cuda", dtype=torch.bfloat16)
    qw, kw = torch.randn(256, device="cuda"), torch.randn(256, device="cuda")
    positions = torch.tensor([0, 262143], device="cuda", dtype=torch.int64)
    slots = torch.tensor([-1, 785], device="cuda", dtype=torch.int32)
    cache = torch.full((2, 2, 784, 4, 256), 0.125, device="cuda", dtype=torch.bfloat16)
    expected_cache = cache.clone()
    expected = old_prepare(packed, qw, kw, positions, expected_cache, slots)
    eager = state.prepare_attention(packed, qw, kw, positions, cache, slots)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    cache.fill_(0.125)

    def forward(packed, qw, kw, positions, cache, slots):
        q = state.prepare_attention(packed, qw, kw, positions, cache, slots)
        return q, cache[1, 0, 1].clone()

    compiled, read_after_write = compile_forward(forward)(
        packed, qw, kw, positions, cache, slots
    )
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        read_after_write, expected_cache[1, 0, 1], rtol=0, atol=0
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
