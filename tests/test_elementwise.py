"""Validate pointwise fusion against the original rounded PyTorch equations."""

import pytest
import torch
from oh_my_vllm.kernels.elementwise import delta_gates, silu_mul

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]


@pytest.mark.parametrize("rows", [1, 5, 37, 128, 129])
def test_pointwise(rows):
    torch.manual_seed(rows)
    packed = torch.randn(rows, 34816, device="cuda", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, -1)
    torch.testing.assert_close(
        silu_mul(packed), torch.nn.functional.silu(gate) * up, atol=0.0001, rtol=0.008
    )
    ba = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16) * 10
    a_log, bias = torch.randn(48, device="cuda"), torch.randn(48, device="cuda")
    b, a = ba.float().chunk(2, -1)
    g, beta = delta_gates(ba, a_log, bias)
    torch.testing.assert_close(
        g, -a_log.exp() * torch.nn.functional.softplus(a + bias), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(beta, b.sigmoid(), atol=1e-7, rtol=1e-6)
