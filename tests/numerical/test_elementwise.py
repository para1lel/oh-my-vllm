"""Validate pointwise fusion against independent FP64 storage equations."""

import pytest
import torch
from oh_my_vllm.kernels.elementwise import delta_gates, silu_mul

from development.kernels import oracles
from tests.support.fp64 import close, cpu

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]


@pytest.mark.parametrize("rows", [1, 5, 37, 128, 129])
def test_pointwise(rows):
    torch.manual_seed(rows)
    packed = torch.randn(rows, 34816, device="cuda", dtype=torch.bfloat16)
    expected = oracles.silu_mul(cpu(packed))
    close(silu_mul(packed), expected, atol=0.0001, rtol=0.008)
    ba = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16) * 10
    a_log, bias = torch.randn(48, device="cuda"), torch.randn(48, device="cuda")
    expected_g, expected_beta = oracles.gates(*cpu((ba, a_log, bias)))
    g, beta = delta_gates(ba, a_log, bias)
    close(g, expected_g, atol=2e-6, rtol=2e-6)
    close(beta, expected_beta, atol=1e-7, rtol=1e-6)
