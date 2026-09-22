"""Thin TileFoundry adapters calling the frozen TileLang comparison.

Logical state is made explicit for HIR comparison. Cache/graph safety and the
full shape/dtype matrix remain covered by the existing project tests.
"""

import torch
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
    def add_norm(self, x, residual, weight):
        return normalization.add_rms_norm(x, residual, weight)

    @runtime_func
    def gated_norm(self, x, weight, gate):
        return normalization.rms_norm(x, weight, gate=gate)

    @runtime_func
    def rope(self, x, positions):
        return normalization.rotary(x, positions)

    @runtime_func
    def norm_rope(self, x, weight, positions):
        return normalization.rms_rotary(x, weight, positions)

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
