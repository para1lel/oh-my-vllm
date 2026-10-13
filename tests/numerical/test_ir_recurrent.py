"""Persistent convolution and GDN writes remain ordered in compiled graphs."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward, operations, recurrent


def test_recurrent_state_schemas_and_native_reference():
    for name in ("causal_conv", "gdn_recurrent"):
        assert "!" in str(operations()[name].target._schema)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    pool = torch.randn(3, 4, 3, dtype=torch.bfloat16)
    ids = torch.zeros(2, dtype=torch.int32)
    starts = torch.tensor([0, 2], dtype=torch.int32)
    reads = torch.tensor([0], dtype=torch.int32)
    writes = torch.tensor([1, 2], dtype=torch.int32)
    before = pool.clone()
    output = recurrent._native_conv(x, weight, pool, ids, starts, reads, writes)
    assert output.shape == x.shape
    torch.testing.assert_close(pool[2, :, 1:], x.T, rtol=0, atol=0)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)
    result = torch.library.opcheck(
        operations()["causal_conv"].reference,
        (x, weight, before, ids, starts, reads, writes),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


def test_native_convolution_checks_multiple_sequence_ids():
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.ones(4, 4, dtype=torch.bfloat16)
    pool = torch.randn(4, 4, 3, dtype=torch.bfloat16)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    reads = torch.tensor([0, 1], dtype=torch.int32)
    writes = torch.tensor([2, 3], dtype=torch.int32)
    ids = torch.tensor([0, 1], dtype=torch.int32)
    before = pool.clone()
    recurrent._native_conv(x, weight, pool, ids, starts, reads, writes)
    torch.testing.assert_close(pool[2, :, 2], x[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[3, :, 2], x[1], rtol=0, atol=0)
    torch.testing.assert_close(pool[:2], before[:2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="sequence IDs disagree"):
        recurrent._native_conv(
            x,
            weight,
            before,
            torch.tensor([0, 0], dtype=torch.int32),
            starts,
            reads,
            writes,
        )


def test_native_gdn_recurrent_updates_only_selected_state():
    q = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    decay = torch.zeros(1, 1)
    beta = torch.ones(1, 1)
    pool = torch.zeros(2, 1, 128, 128)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    reads = torch.tensor([0], dtype=torch.int32)
    writes = torch.tensor([1], dtype=torch.int32)
    output = recurrent._native_recurrent(
        q, k, v, decay, beta, pool, starts, reads, writes
    )
    assert output.shape == v.shape
    torch.testing.assert_close(pool[0], torch.zeros_like(pool[0]), rtol=0, atol=0)
    expected_k = 1 / (128 + 1e-6) ** 0.5
    torch.testing.assert_close(
        pool[1], torch.full_like(pool[1], expected_k), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        output.float(),
        torch.full_like(output.float(), 1 / 128**0.5),
        rtol=0.01,
        atol=0.01,
    )
    result = torch.library.opcheck(
        operations()["gdn_recurrent"].reference,
        (q, k, v, decay, beta, pool, starts, reads, writes),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compiled_convolution_matches_existing_cuda_and_ordered_write():
    from oh_my_vllm.kernels.convolution import causal_conv as old_conv

    x = torch.randn(3, 10240, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(10240, 4, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(4, 10240, 3, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(3, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    reads = torch.tensor([0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([-1, -1, 2], device="cuda", dtype=torch.int32)
    expected_pool = pool.clone()
    expected = old_conv(x, weight, expected_pool, ids, starts, reads, writes)

    def forward(x, weight, pool, ids, starts, reads, writes):
        output = recurrent.causal_conv(x, weight, pool, ids, starts, reads, writes)
        return output, pool[2].clone()

    output, written = compile_forward(forward)(
        x, weight, pool, ids, starts, reads, writes
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(written, expected_pool[2], rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compiled_gdn_recurrent_matches_existing_cuda_and_ordered_write():
    from oh_my_vllm.kernels.gdn import recurrent as old_recurrent

    q = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 48, 128, device="cuda", dtype=torch.bfloat16)
    decay = -torch.rand(2, 48, device="cuda")
    beta = torch.rand_like(decay)
    pool = torch.randn(4, 48, 128, 128, device="cuda")
    starts = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    reads = torch.tensor([0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    expected_pool = pool.clone()
    expected = old_recurrent(q, k, v, decay, beta, expected_pool, starts, reads, writes)

    def forward(q, k, v, decay, beta, pool, starts, reads, writes):
        output = recurrent.gdn_recurrent(
            q, k, v, decay, beta, pool, starts, reads, writes
        )
        return output, pool[2].clone()

    output, written = compile_forward(forward)(
        q, k, v, decay, beta, pool, starts, reads, writes
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(written, expected_pool[2], rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
