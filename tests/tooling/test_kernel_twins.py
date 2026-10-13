"""Runtime analysis adapters must cover the complete HIR and preserve its work."""

import pytest
import torch
from oh_my_vllm.ir import normalized_linear as ir
from oh_my_vllm.ir.fp8 import _native_linear
from oh_my_vllm.ir.pointwise import _native_rms_norm

from development.kernels import twins
from development.kernels.semantics import Operators


def test_runtime_twin_binds_all_hir_functions():
    runtime = twins.TileLang()
    assert all(
        callable(getattr(runtime, function.name)) for function in Operators.functions
    )


def test_gated_twin_composes_head_rounding_and_projection(monkeypatch):
    generator = torch.Generator().manual_seed(1627)
    x = torch.randn(2, 48, 128, generator=generator).bfloat16()
    gate = torch.randn_like(x)
    gamma = torch.randn(128, generator=generator)
    weight = torch.randn(256, 6144, generator=generator)
    scale = torch.rand(2, 48, generator=generator) * 0.01
    monkeypatch.setattr(
        twins.normalization,
        "rms_norm",
        lambda x, gamma, *, gate: _native_rms_norm(x, gamma, 1e-6, gate),
    )
    monkeypatch.setattr(
        twins.fp8,
        "linear",
        lambda x, weight, scale: _native_linear(x, weight, scale, False),
    )
    expected = ir._gated_native(x, gamma, gate, weight, scale)
    actual = twins.TileLang().gated_norm_fp8_linear(x, gamma, gate, weight, scale)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_prefill_twin_preserves_already_normalized_query_and_key():
    # A known non-unit norm makes accidental extra normalization observable.
    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0] = 0.5
    k[..., 0] = 0.25
    v = torch.ones(1, 6, 128, dtype=torch.bfloat16)
    g = torch.zeros(1, 6)
    beta = torch.full((1, 6), 0.5)
    state = torch.zeros(6, 128, 128)
    output, final = twins.TileLang().prefill_delta(q, k, v, g, beta, state)
    expected_state = torch.zeros_like(state)
    expected_state[:, :, 0] = 0.125
    torch.testing.assert_close(final, expected_state, atol=0, rtol=0)
    expected_output = torch.full_like(v, 0.0625 * 128**-0.5)
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gated_twin_runs_actual_frozen_kernels():
    generator = torch.Generator(device="cuda").manual_seed(1631)
    x = torch.randn(2, 48, 128, device="cuda", generator=generator).bfloat16()
    gate = torch.randn(2, 48, 128, device="cuda", generator=generator).bfloat16()
    gamma = torch.randn(128, device="cuda", generator=generator)
    weight = torch.randn(256, 6144, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(2, 48, device="cuda", generator=generator) * 0.01
    actual = twins.TileLang().gated_norm_fp8_linear(x, gamma, gate, weight, scale)
    expected = ir._gated_native(x, gamma, gate, weight, scale)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
