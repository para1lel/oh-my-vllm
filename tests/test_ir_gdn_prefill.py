"""FlashInfer GDN prefill is an opaque, schema-checked IR node."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward, operations
from oh_my_vllm.ir import gdn_prefill as prefill_ir


def test_native_gdn_prefill_reference_and_schema():
    q = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    decay = torch.zeros(1, 1)
    beta = torch.ones(1, 1)
    initial = torch.zeros(1, 1, 128, 128)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    output, final = prefill_ir._native_prefill(q, k, v, decay, beta, initial, starts)
    assert output.shape == v.shape
    assert final.shape == initial.shape
    assert torch.all(output > 0)
    torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
    result = torch.library.opcheck(
        operations()["gdn_prefill"].reference,
        (q, k, v, decay, beta, initial, starts),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compiled_gdn_prefill_matches_existing_flashinfer_path():
    from oh_my_vllm.kernels.gdn import prefill as old_prefill

    q = torch.randn(7, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(7, 48, 128, device="cuda", dtype=torch.bfloat16)
    decay = -torch.rand(7, 48, device="cuda")
    beta = torch.rand_like(decay)
    initial = torch.zeros(1, 48, 128, 128, device="cuda")
    starts = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
    args = q, k, v, decay, beta, initial, starts
    expected = old_prefill(*args)
    eager = prefill_ir.gdn_prefill(*args)
    compiled = compile_forward(prefill_ir.gdn_prefill)(*args)
    for actual, reference in zip(
        (*eager, *compiled), (*expected, *expected), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gdn_prefill_reference_multi_sequence_chunk_boundary():
    from oh_my_vllm.kernels.gdn import prefill as old_prefill

    torch.manual_seed(129)
    q = torch.randn(132, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(132, 48, 128, device="cuda", dtype=torch.bfloat16)
    decay = -torch.rand(132, 48, device="cuda") * 0.1
    beta = torch.rand_like(decay)
    initial = torch.randn(2, 48, 128, 128, device="cuda") * 0.01
    starts = torch.tensor([0, 129, 132], device="cuda", dtype=torch.int32)
    expected_output, expected_states = old_prefill(
        q, k, v, decay, beta, initial, starts
    )
    actual_output, actual_states = prefill_ir._native_prefill(
        q, k, v, decay, beta, initial, starts
    )
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=0.03, atol=0.03
    )
    torch.testing.assert_close(actual_states, expected_states, rtol=0.03, atol=0.03)
