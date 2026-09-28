"""Supplementary PyTorch SDPA vs CPU FP64; not actual model-path acceptance.

Production paged attention is covered by test_independent_kernels.py,
test_independent_decode_attention.py and the actual-model probe_worker.py.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# CPU FP64 reference
# ---------------------------------------------------------------------------


def gqa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Grouped-query attention on CPU in FP64.

    Args:
        q: (batch, num_heads, seq_len, head_dim) float64
        k: (batch, num_kv_heads, seq_len, head_dim) float64
        v: (batch, num_kv_heads, seq_len, head_dim) float64

    Returns:
        (batch, num_heads, seq_len, head_dim) float64
    """
    _batch, num_heads, _seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    groups = num_heads // num_kv_heads

    # Expand k/v to match q head count.
    k_exp = k.repeat_interleave(groups, dim=1)  # (B, H, S, D)
    v_exp = v.repeat_interleave(groups, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    # (B, H, S, S)
    attn_weights = torch.einsum("bhid,bhjd->bhij", q * scale, k_exp)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", attn_weights, v_exp)


# ---------------------------------------------------------------------------
# GPU path via torch.nn.functional.scaled_dot_product_attention
# ---------------------------------------------------------------------------


def gqa_gpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """GQA on GPU using PyTorch's fused SDPA (dispatches to FlashAttention when
    available).

    Args:
        q: (batch, num_heads, seq_len, head_dim) float16, on CUDA
        k: (batch, num_kv_heads, seq_len, head_dim) float16, on CUDA
        v: (batch, num_kv_heads, seq_len, head_dim) float16, on CUDA

    Returns:
        (batch, num_heads, seq_len, head_dim) float16, on CUDA
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    groups = num_heads // num_kv_heads

    k_exp = k.repeat_interleave(groups, dim=1)
    v_exp = v.repeat_interleave(groups, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "batch,num_heads,num_kv_heads,seq_len,head_dim",
    [
        # Qwen3.8-27B full-attention layer shape:
        #   24 query heads, 4 KV heads, head_dim=256.
        (1, 24, 4, 128, 256),
        # Smaller shape for faster CI runs.
        (2, 16, 4, 64, 64),
        # Prefill length matching our benchmark (seq_len chunk).
        (1, 24, 4, 512, 256),
    ],
)
def test_gqa_accuracy(
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> None:
    torch.manual_seed(42)

    q64 = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float64)
    k64 = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float64)
    v64 = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float64)

    ref = gqa_reference(q64, k64, v64)

    q16 = q64.half().cuda()
    k16 = k64.half().cuda()
    v16 = v64.half().cuda()

    gpu_out = gqa_gpu(q16, k16, v16).cpu().double()

    max_err = (gpu_out - ref).abs().max().item()
    mean_err = (gpu_out - ref).abs().mean().item()

    print(
        f"  shape=({batch},{num_heads},{num_kv_heads},{seq_len},{head_dim})  "
        f"max_err={max_err:.4e}  mean_err={mean_err:.4e}"
    )

    assert torch.allclose(gpu_out, ref, atol=1e-2, rtol=1e-2), (
        f"GQA output exceeds tolerance: max_err={max_err:.4e}"
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    cases = [
        (1, 24, 4, 128, 256),
        (2, 16, 4, 64, 64),
        (1, 24, 4, 512, 256),
    ]
    passed = 0
    for args in cases:
        try:
            test_gqa_accuracy(*args)
            print("  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")

    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)
