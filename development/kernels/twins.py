"""Thin TileFoundry adapters calling the frozen TileLang comparison.

Logical state is made explicit for HIR comparison. Cache/graph safety and the
full shape/dtype matrix remain covered by the existing project tests.
"""

import torch
from oh_my_vllm.kernels import dspark_tilelang_reference as dspark
from oh_my_vllm.kernels.tilelang_reference import (
    attention,
    convolution,
    decode_attention,
    elementwise,
    fp8,
    gdn,
    normalization,
)
from tilefoundry.runtime import runtime_func, runtime_module

from development.kernels.semantics import Operators


def _delta(q, k, v, g, beta, state):
    count = len(q)
    pool = torch.empty((count + 1, *state.shape), device=q.device, dtype=state.dtype)
    pool[0].copy_(state)
    starts = torch.tensor([0, count], device=q.device, dtype=torch.int32)
    reads = torch.zeros(1, device=q.device, dtype=torch.int32)
    writes = torch.arange(1, count + 1, device=q.device, dtype=torch.int32)
    out = gdn.recurrent(q, k, v, g, beta, pool, starts, reads, writes)
    return out, pool[1:]


def _prefill_delta(q, k, v, g, beta, state):
    """Evaluate the HIR recurrence with already normalized BF16 Q/K.

    The formal GDN harness uses the full chunked inference operation. This
    twin follows the small HIR recurrence without another Q/K normalization.
    """
    current = state.float()
    outputs = []
    for row in range(len(q)):
        query = q[row].float().repeat_interleave(v.shape[1] // q.shape[1], dim=0)
        key = k[row].float().repeat_interleave(v.shape[1] // k.shape[1], dim=0)
        decayed = current * g[row].exp()[:, None, None]
        recalled = (decayed * key[:, None, :]).sum(-1)
        residual = (v[row].float() - recalled) * beta[row, :, None]
        current = decayed + residual[:, :, None] * key[:, None, :]
        output = (current * query[:, None, :]).sum(-1) * (128**-0.5)
        outputs.append(output.to(v.dtype))
    return torch.stack(outputs), current


@runtime_module(Operators)
class TileLang:
    @runtime_func
    def silu(self, x):
        return elementwise.silu_mul(x)

    @runtime_func
    def gates(self, ba, log, bias):
        return elementwise.delta_gates(ba, log, bias)

    @runtime_func
    def norm(self, x, weight):
        return normalization.rms_norm(x, weight)

    @runtime_func
    def dspark_rms_norm(self, x, weight):
        return dspark.rms_norm(x, weight)

    @runtime_func
    def dspark_norm_rope(self, x, weight, positions, inv_freq):
        return dspark.normalize_rope(x, weight, positions, inv_freq, 1.3465735902799727)

    @runtime_func
    def dspark_append(self, k, v, new_k, new_v):
        cache = torch.zeros((2, 2, 784, 8, 128), device=k.device, dtype=k.dtype)
        prior = torch.arange(783, device=k.device, dtype=torch.int64)
        slots = torch.tensor([783, 784], device=k.device, dtype=torch.int64)
        dspark.append(cache, k, v, prior)
        dspark.append(cache, new_k, new_v, slots)
        logical = torch.arange(785, device=k.device)
        return cache[logical // 784, 0, logical % 784], cache[
            logical // 784, 1, logical % 784
        ]

    @runtime_func
    def dspark_attention(self, q, k, v, block_k, block_v):
        cache = torch.zeros((1, 2, 784, 8, 128), device=k.device, dtype=k.dtype)
        slots = torch.arange(783, device=k.device, dtype=torch.int64)
        dspark.append(cache, k, v, slots)
        tables = torch.zeros((1, 1), device=k.device, dtype=torch.int32)
        contexts = torch.tensor([783], device=k.device, dtype=torch.int32)
        return dspark.attention(
            q[None], cache, tables, contexts, block_k[None], block_v[None]
        )[0]

    @runtime_func
    def add_norm(self, x, residual, weight):
        return normalization.add_rms_norm(x, residual, weight)

    @runtime_func
    def gated_norm(self, x, weight, gate):
        return normalization.rms_norm(x, weight, gate=gate)

    @runtime_func
    def gated_norm_fp8_linear(self, x, gamma, gate, weight, scale):
        normalized = normalization.rms_norm(x, gamma, gate=gate)
        return fp8.linear(normalized.flatten(1), weight, scale)

    @runtime_func
    def add_norm_fp8_linear(self, x, residual, gamma, weight, scale):
        summed, normalized = normalization.add_rms_norm(x, residual, gamma)
        return summed, fp8.linear(normalized, weight, scale)

    @runtime_func
    def fp8_linear(self, x, weight, scale):
        return fp8.linear(x, weight, scale)

    @runtime_func
    def fp8_silu_linear(self, packed, weight, scale):
        return fp8.linear(packed, weight, scale, silu_gate=True)

    @runtime_func
    def rope(self, x, positions):
        return normalization.rotary(x, positions)

    @runtime_func
    def norm_rope(self, x, weight, positions):
        return normalization.rms_rotary(x, weight, positions)

    @runtime_func
    def prepare_attention(self, packed, qw, kw, positions, old_k, old_v):
        from oh_my_vllm.kernels.attention_prepare import tilelang_prepare

        cache = torch.zeros(
            (2, 2, 784, 4, 256), device=packed.device, dtype=packed.dtype
        )
        prior = torch.arange(783, device=packed.device, dtype=torch.int64)
        slots = torch.tensor([783, 784], device=packed.device, dtype=torch.int64)
        attention.append(cache, old_k, old_v, prior)
        q = tilelang_prepare(packed, qw, kw, positions, cache, slots)
        logical = torch.arange(785, device=packed.device)
        return (
            q,
            cache[logical // 784, 0, logical % 784],
            cache[logical // 784, 1, logical % 784],
        )

    @runtime_func
    def prepare_context(self, packed, kw, positions, old_k, old_v):
        from oh_my_vllm.kernels.partial_attention import tilelang_context

        cache = torch.zeros(
            (2, 2, 784, 4, 256), device=packed.device, dtype=packed.dtype
        )
        prior = torch.arange(783, device=packed.device, dtype=torch.int64)
        slots = torch.tensor([783, 784], device=packed.device, dtype=torch.int64)
        attention.append(cache, old_k, old_v, prior)
        tilelang_context(packed, kw, positions, cache, slots)
        logical = torch.arange(785, device=packed.device)
        return cache[logical // 784, 0, logical % 784], cache[
            logical // 784, 1, logical % 784
        ]

    @runtime_func
    def prepare_query(self, packed, qw, positions):
        from oh_my_vllm.kernels.partial_attention import tilelang_query

        return tilelang_query(packed, qw, positions)

    @runtime_func
    def quant(self, x):
        return fp8.quantize(x)

    @runtime_func
    def silu_quant(self, x):
        return fp8.quantize(x, silu_gate=True)

    @runtime_func
    def qk(self, q, k):
        return gdn.normalize_qk(q, k)

    @runtime_func
    def delta_step(self, q, k, v, g, beta, state):
        out, snapshots = _delta(q, k, v, g, beta, state)
        return out, snapshots[0]

    @runtime_func
    def recurrent(self, q, k, v, g, beta, state):
        return _delta(q, k, v, g, beta, state)

    @runtime_func
    def prefill_delta(self, q, k, v, g, beta, state):
        return _prefill_delta(q, k, v, g, beta, state)

    @runtime_func
    def gdn_prefill(self, q, k, v, g, beta, state):
        normalized_q, normalized_k = gdn.normalize_qk(q, k)
        return _prefill_delta(normalized_q, normalized_k, v, g, beta, state)

    @runtime_func
    def convolution(self, x, weight, state):
        pool = torch.empty((3, 128, 3), device=x.device, dtype=x.dtype)
        pool[0].copy_(state)
        seq = torch.zeros(2, device=x.device, dtype=torch.int32)
        starts = torch.tensor([0, 2], device=x.device, dtype=torch.int32)
        reads = torch.zeros(1, device=x.device, dtype=torch.int32)
        writes = torch.tensor([1, 2], device=x.device, dtype=torch.int32)
        out = convolution.causal_conv(x, weight, pool, seq, starts, reads, writes)
        return out, pool[1:]

    @runtime_func
    def append(self, k, v, new_k, new_v):
        cache = torch.zeros((2, 2, 784, 4, 256), device=k.device, dtype=k.dtype)
        prior = torch.arange(783, device=k.device, dtype=torch.int64)
        slots = torch.tensor([783, 784], device=k.device, dtype=torch.int64)
        attention.append(cache, k, v, prior)
        attention.append(cache, new_k, new_v, slots)
        logical = torch.arange(785, device=k.device)
        return cache[logical // 784, 0, logical % 784], cache[
            logical // 784, 1, logical % 784
        ]

    @runtime_func
    def attention(self, q, k, v):
        # HIR receives only valid rows; physical draft position zero is absent.
        cache = torch.full(
            (2, 2, 784, 4, 256), float("nan"), device=k.device, dtype=k.dtype
        )
        slots = torch.arange(1, 785, device=k.device, dtype=torch.int64)
        attention.append(cache, k, v, slots)
        tables = torch.tensor([[0, 1]], device=k.device, dtype=torch.int32)
        lengths = torch.tensor([785], device=k.device, dtype=torch.int32)
        return decode_attention.decode(
            q, cache, tables, lengths, first=1, max_tokens=1568
        )
