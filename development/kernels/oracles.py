"""Independent FP64 operator equations with explicit contract storage casts.

No production implementation is a numerical oracle. Cache copies keep their
exact storage contract. Arithmetic uses FP64; casts implement visible storage
boundaries. The full-model oracle instead keeps all intermediates in FP64.
"""

import math
from itertools import pairwise

import torch

from development import fp64_math as ref


def attention_gate(x, gate):
    factor = ref.store(gate.double().sigmoid(), gate.dtype)
    return ref.store(x.double() * factor.double(), x.dtype)


def normalize_qk(q, k):
    return ref.store(ref.l2(q), q.dtype), ref.store(ref.l2(k), k.dtype)


def append(cache, k, v, slots):
    valid = slots >= 0
    rows = slots[valid].long()
    cache[rows // 784, 0, rows % 784] = k[valid]
    cache[rows // 784, 1, rows % 784] = v[valid]


def rms(x, weight, *, gate=None, eps=1e-6):
    result = ref.rms(x, weight, eps)
    if gate is not None:
        result *= ref.silu(gate)
    return ref.store(result, x.dtype).contiguous()


def add_rms(x, residual, weight):
    summed = ref.store(x.double() + residual.double(), x.dtype)
    return summed, rms(summed, weight)


def silu_mul(x):
    gate, up = x.chunk(2, -1)
    activated = ref.store(ref.silu(gate), x.dtype)
    return ref.store(activated.double() * up.double(), x.dtype)


def gates(ba, a_log, bias):
    b, a = ba.chunk(2, -1)
    decay, beta = ref.delta_gates(b, a, a_log, bias)
    return decay.float(), beta.float()


def rms_rotary(x, weight, positions):
    normalized = ref.store(ref.rms(x, weight), x.dtype)
    return ref.store(ref.rotary(normalized, positions), x.dtype)


def prepare_query(packed, weight, positions):
    return rms_rotary(packed.reshape(-1, 24, 512)[..., :256], weight, positions)


def prepare_context(packed, weight, positions, cache, slots):
    key = rms_rotary(packed[:, :1024].reshape(-1, 4, 256), weight, positions)
    append(cache, key, packed[:, 1024:].reshape(-1, 4, 256), slots)


def prepare_attention(packed, q_weight, k_weight, positions, cache, slots):
    query, context = packed.split((12288, 2048), -1)
    prepare_context(context, k_weight, positions, cache, slots)
    return prepare_query(query, q_weight, positions)


def decode(
    query,
    cache,
    tables,
    lengths,
    *,
    first=0,
    max_tokens=65536,
    starts=None,
    max_query_len=5,
):
    outputs = []
    for row in range(len(query)):
        indices = torch.arange(first, int(lengths[row]), device=query.device)
        pages = tables[row, indices // 784].long()
        k, v = cache[pages, 0, indices % 784], cache[pages, 1, indices % 784]
        repeat = query.shape[1] // k.shape[1]
        key, value = (
            k.double().repeat_interleave(repeat, 1),
            v.double().repeat_interleave(repeat, 1),
        )
        scores = torch.einsum("hd,thd->ht", query[row].double(), key) / math.sqrt(
            query.shape[-1]
        )
        outputs.append(
            ref.store(
                torch.einsum("ht,thd->hd", scores.softmax(-1), value), query.dtype
            )
        )
    return torch.stack(outputs)


def prefill(q, k, v, log_decay, beta, initial, starts):
    outputs, states = [], []
    for sequence, (left, right) in enumerate(pairwise(starts.cpu().tolist())):
        output, state = ref.delta_chunk(
            q[left:right],
            k[left:right],
            v[left:right],
            log_decay[left:right],
            beta[left:right],
            initial[sequence],
            normalization_dtype=q.dtype,
        )
        outputs.append(ref.store(output, v.dtype))
        states.append(ref.store(state, initial.dtype))
    return torch.cat(outputs), torch.stack(states)


def recurrent(q, k, v, log_decay, beta, pool, starts, reads, writes):
    sources = pool.index_select(0, reads.long()).double().clone()
    outputs = []
    for sequence, (left, right) in enumerate(pairwise(starts.cpu().tolist())):
        state = sources[sequence]
        for token in range(left, right):
            output, state = ref.delta_recurrence(
                q[token : token + 1],
                k[token : token + 1],
                v[token : token + 1],
                log_decay[token : token + 1],
                beta[token : token + 1],
                state,
            )
            outputs.append(ref.store(output, v.dtype))
            if writes[token] >= 0:
                pool[writes[token]] = ref.store(state, pool.dtype)
    return torch.cat(outputs)


def convolution(x, weight, pool, sequence_ids, starts, reads, writes):
    sources = pool.index_select(0, reads.long()).clone()
    outputs = []
    for sequence, (left, right) in enumerate(pairwise(starts.cpu().tolist())):
        if not bool((sequence_ids[left:right] == sequence).all()):
            raise ValueError("convolution sequence IDs disagree with offsets")
        output, _ = ref.causal_conv(x[left:right], weight, sources[sequence])
        outputs.append(ref.store(output, x.dtype))
        history = torch.cat((sources[sequence], x[left:right].T), -1)
        for token in range(left, right):
            if writes[token] >= 0:
                offset = token - left + 1
                pool[writes[token]] = history[:, offset : offset + 3]
    return torch.cat(outputs)


def dspark_rms(x, weight, eps=1e-6):
    normalized = ref.store(ref.rms(x, torch.ones_like(weight), eps), x.dtype)
    return ref.store(normalized.double() * ref.store(weight, x.dtype).double(), x.dtype)


def dspark_norm_rope(x, weight, positions, inv_freq, attention_factor, eps=1e-6):
    normalized = dspark_rms(x, weight, eps).double()
    phase = positions.double()[:, None, None] * inv_freq.double()[None, None]
    cosine = ref.store(phase.cos() * attention_factor, x.dtype).double()
    sine = ref.store(phase.sin() * attention_factor, x.dtype).double()
    left, right = normalized[..., :64], normalized[..., 64:]
    first = (
        ref.store(left * cosine, x.dtype).double()
        - ref.store(right * sine, x.dtype).double()
    )
    second = (
        ref.store(right * cosine, x.dtype).double()
        + ref.store(left * sine, x.dtype).double()
    )
    return ref.store(torch.cat((first, second), -1), x.dtype)


def dspark_attention(query, cache, tables, lengths, block_key, block_value):
    outputs = []
    for request, length in enumerate(lengths.cpu().tolist()):
        positions = torch.arange(length, device=query.device)
        pages = tables[request].long()[positions // 784]
        key = torch.cat((cache[pages, 0, positions % 784], block_key[request])).double()
        value = torch.cat(
            (cache[pages, 1, positions % 784], block_value[request])
        ).double()
        rows = query[request].double().reshape(7, 8, 4, 128)
        scores = torch.einsum("qhrd,khd->qhrk", rows, key) / math.sqrt(128)
        out = torch.einsum("qhrk,khd->qhrd", scores.softmax(-1), value)
        outputs.append(ref.store(out.reshape(7, 32, 128), query.dtype))
    return torch.stack(outputs)


def quantize(x, *, column_major=False, silu_gate=False):
    if silu_gate:
        x = silu_mul(x)
    values = x.cpu().double().reshape(x.shape[0], -1, 128)
    magnitudes = torch.nan_to_num(
        values.abs(), nan=0, posinf=float("inf"), neginf=float("inf")
    )
    # Scale is a public FP32 buffer. Use its stored value in FP64 division.
    minimum = float(torch.tensor(1e-10, dtype=torch.float32))
    scale = (magnitudes.amax(-1).clamp_min(minimum) / 448).float()
    # Quantization prescribes an IEEE FP32 division result before encoding.
    # Emulate that rounding explicitly after independent FP64 division.
    divided = (values / scale.double()[..., None]).float().double()
    saturated = torch.fmin(
        torch.full_like(divided, 448),
        torch.fmax(torch.full_like(divided, -448), divided),
    )
    encoded = ref.encode_fp8(saturated).reshape(x.shape).to(x.device)
    scale = scale.to(x.device)
    if column_major:
        column = torch.empty(
            (scale.shape[1], scale.shape[0]), dtype=scale.dtype, device=x.device
        ).T
        column.copy_(scale)
        scale = column
    return encoded, scale


def linear(x, weight, scale, *, silu_gate=False):
    encoded, activation_scale = quantize(x, silu_gate=silu_gate)
    return groupwise(encoded, weight, activation_scale, scale)


def groupwise(encoded, weight, activation_scale, weight_scale):
    activation = encoded.double() * activation_scale.double().repeat_interleave(128, -1)
    return ref.store(
        activation @ ref.decode_weight(weight, weight_scale).T, torch.bfloat16
    )


def dense_linear(x, weight):
    return ref.store(x.double() @ weight.double().T, torch.bfloat16)


def gdn_prepare(
    x,
    weight,
    scale,
    ba_weight,
    conv_weight,
    pool,
    ids,
    starts,
    reads,
    writes,
    a_log,
    bias,
):
    packed = linear(x, weight, scale)
    mixed, z = packed.split((10240, 6144), -1)
    mixed = convolution(mixed, conv_weight, pool, ids, starts, reads, writes)
    decay, beta = gates(dense_linear(x, ba_weight), a_log, bias)
    return mixed, z, decay, beta


def add_norm_linear(x, residual, gamma, weight, scale):
    summed, normalized = add_rms(x, residual, gamma)
    return summed, linear(normalized, weight, scale)


def gated_norm_linear(x, gamma, gate, weight, scale):
    return linear(rms(x, gamma, gate=gate).flatten(1), weight, scale)


def operator(module, name):
    return {
        ("elementwise", "attention_gate"): attention_gate,
        ("attention", "append"): append,
        ("attention_prepare", "prepare_attention"): prepare_attention,
        ("partial_attention", "prepare_context"): prepare_context,
        ("partial_attention", "prepare_query"): prepare_query,
        ("normalization", "add_rms_norm"): add_rms,
        ("normalization", "rms_rotary"): rms_rotary,
        ("normalization", "rms_norm"): rms,
        ("elementwise", "silu_mul"): silu_mul,
        ("elementwise", "delta_gates"): gates,
        ("gdn", "normalize_qk"): normalize_qk,
        ("gdn", "recurrent"): recurrent,
        ("gdn", "prefill"): prefill,
        ("convolution", "causal_conv"): convolution,
        ("decode_attention", "decode"): decode,
        ("fp8", "quantize"): quantize,
        ("fp8", "linear"): linear,
        ("fp8", "add_norm_linear"): add_norm_linear,
        ("fp8", "gated_norm_linear"): gated_norm_linear,
        ("dspark_attention", "rms_norm"): dspark_rms,
        ("dspark_attention", "normalize_rope"): dspark_norm_rope,
        ("dspark_attention", "append"): append,
        ("dspark_attention", "attention"): dspark_attention,
    }[module, name]
