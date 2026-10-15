"""The small-batch vocabulary projection keeps its Torch path in IR."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward, operations
from oh_my_vllm.ir.logits import logits_gemm

from development.kernels import oracles
from tests.support.fp64 import close, cpu


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="GPU required"
                ),
            ],
        ),
    ],
)
def test_logits_preserves_bf16_projection_before_float32_output(device):
    hidden = torch.full((1, 128), 0.1, device=device, dtype=torch.bfloat16)
    head = torch.full((1, 128), 0.3, device=device, dtype=torch.bfloat16)
    expected = oracles.dense_linear(*cpu((hidden, head))).float()
    assert expected.item() == 3.859375
    projection = (
        operations()["logits_gemm"].reference if device == "cpu" else logits_gemm
    )
    results = [projection(hidden, head)]
    if device == "cuda":
        results.append(compile_forward(projection)(hidden, head))
    for actual in results:
        close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, actual.bfloat16().float(), rtol=0, atol=0)


def test_logits_native_semantics_and_fake_contract():
    hidden = torch.randn(2, 128, dtype=torch.bfloat16)
    head = torch.randn(256, 128, dtype=torch.bfloat16)
    actual = operations()["logits_gemm"].reference(hidden, head)
    expected = oracles.dense_linear(hidden, head).float()
    close(actual, expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(actual, actual.bfloat16().float(), rtol=0, atol=0)
    result = torch.library.opcheck(
        operations()["logits_gemm"].reference,
        (hidden, head),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_logits_compiles_and_matches_fp64():
    hidden = torch.randn(2, 5120, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(256, 5120, device="cuda", dtype=torch.bfloat16)
    expected = oracles.dense_linear(*cpu((hidden, head))).float()
    eager = logits_gemm(hidden, head)
    compiled = compile_forward(logits_gemm)(hidden, head)
    close(eager, expected, rtol=0.03, atol=0.03)
    close(compiled, expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    for actual in (eager, compiled):
        torch.testing.assert_close(actual, actual.bfloat16().float(), rtol=0, atol=0)
