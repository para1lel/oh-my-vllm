"""The small-batch vocabulary projection keeps its FlashInfer path in IR."""

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.ir import compile_forward, operations
from oh_my_vllm.ir.logits import logits_gemm


def test_logits_native_semantics_and_fake_contract():
    hidden = torch.randn(2, 128, dtype=torch.bfloat16)
    head = torch.randn(256, 128, dtype=torch.bfloat16)
    actual = operations()["logits_gemm"].reference(hidden, head)
    torch.testing.assert_close(actual, F.linear(hidden, head).float(), rtol=0, atol=0)
    result = torch.library.opcheck(
        operations()["logits_gemm"].reference,
        (hidden, head),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_logits_flashinfer_compiles_and_matches_existing_path():
    from flashinfer.gemm import mm_bf16

    hidden = torch.randn(2, 5120, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(256, 5120, device="cuda", dtype=torch.bfloat16)
    expected = mm_bf16(hidden, head.T, backend="cute-dsl").float()
    eager = logits_gemm(hidden, head)
    compiled = compile_forward(logits_gemm)(hidden, head)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)
