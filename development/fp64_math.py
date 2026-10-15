"""Independent FP64 equations for numerical checks, without inference providers.

Inputs represent their stored values. All arithmetic and returned mathematical
results use float64; callers perform explicit output-storage casts if required.
"""

from __future__ import annotations

import math

import torch


def rms(x, weight, eps=1e-6):
    values = x.double()
    return (
        values
        * (values.square().mean(-1, keepdim=True) + eps).rsqrt()
        * weight.double()
    )


def l2(x, eps=1e-6):
    values = x.double()
    return values * (values.square().sum(-1, keepdim=True) + eps).rsqrt()


def silu(x):
    values = x.double()
    return values * values.sigmoid()


def store(values, dtype):
    """Explicit direct rounding to storage, avoiding FP64 -> FP32 -> BF16."""
    values = values.double()
    if dtype != torch.bfloat16:
        return values.to(dtype)
    magnitude = values.abs()
    _, exponent = torch.frexp(magnitude)
    step_exponent = (exponent - 8).clamp_min(-133)
    rounded = torch.ldexp(
        torch.round(torch.ldexp(magnitude, -step_exponent)), step_exponent
    )
    # Every rounded finite value is exactly BF16-representable. The final
    # Torch cast cannot introduce a midpoint double-rounding error here.
    return torch.copysign(rounded, values).to(torch.bfloat16)


def rotary(x, positions, *, theta=1e7, rotary_dim=64):
    values = x.double()
    if rotary_dim % 2 or not 0 < rotary_dim <= values.shape[-1]:
        raise ValueError("rotary width must be positive, even, and within the head")
    half = rotary_dim // 2
    frequencies = theta ** (
        -torch.arange(half, device=x.device, dtype=torch.float64) / half
    )
    angles = positions.double()[:, None, None] * frequencies[None, None]
    left, right = values[..., :half], values[..., half:rotary_dim]
    return torch.cat(
        (
            left * angles.cos() - right * angles.sin(),
            right * angles.cos() + left * angles.sin(),
            values[..., rotary_dim:],
        ),
        -1,
    )


def decode_weight(weight, scale=None, *, block=128):
    values = weight.double()
    if scale is None:
        return values
    expected = tuple(math.ceil(n / block) for n in weight.shape)
    if weight.ndim != 2 or scale.shape != expected:
        raise ValueError("weight scale does not match its quantization blocks")
    expanded = scale.double().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return values * expanded[: weight.shape[0], : weight.shape[1]]


def encode_fp8(values):
    """Round finite, saturated FP64 values directly to E4M3, ties to even.

    A generic float64-to-FP8 cast can pass through a float32 constructor. An
    explicit codebook avoids that extra rounding in the independent oracle.
    """
    values = values.double()
    if not bool(torch.isfinite(values).all()) or bool((values.abs() > 448).any()):
        raise ValueError("FP8 encoding requires finite values within saturation")
    codes = torch.arange(127, device=values.device, dtype=torch.uint8)
    levels = codes.view(torch.float8_e4m3fn).double()
    magnitude = values.abs().contiguous()
    high = torch.searchsorted(levels, magnitude).clamp_max(126)
    low = (high - 1).clamp_min(0)
    distance_low, distance_high = magnitude - levels[low], levels[high] - magnitude
    use_low = (distance_low < distance_high) | (
        (distance_low == distance_high) & (low % 2 == 0)
    )
    selected = torch.where(use_low, low, high).to(torch.uint8)
    selected = selected | (torch.signbit(values).to(torch.uint8) * 128)
    return selected.view(torch.float8_e4m3fn)


def causal_conv(x, weight, history):
    """Depthwise causal convolution and SiLU; history is oldest to newest."""
    values, kernel, prior = x.double(), weight.double(), history.double()
    width = kernel.shape[1]
    if prior.shape != (x.shape[1], width - 1):
        raise ValueError("convolution history has the wrong shape")
    packed = torch.cat((prior, values.T), -1)
    result = torch.zeros_like(values)
    for tap in range(width):
        result += packed[:, tap : tap + len(x)].T * kernel[:, tap]
    return silu(result), packed[:, -(width - 1) :].clone()


def attention(q, k, v, *, first_position=0, query_block=128, key_block=1024):
    """Causal GQA with FP64 online softmax and bounded score storage."""
    query, key, value = q.double(), k.double(), v.double()
    if query_block <= 0 or key_block <= 0:
        raise ValueError("attention blocks must be positive")
    if len(k) != len(v) or q.shape[1] % k.shape[1]:
        raise ValueError("incompatible attention heads or sequence lengths")
    if first_position < 0 or first_position + len(q) != len(k):
        raise ValueError("queries must be the final causal positions in KV")
    outputs = torch.empty_like(query)
    repeat = query.shape[1] // key.shape[1]
    for head in range(query.shape[1]):
        kv_head = head // repeat
        for begin in range(0, len(query), query_block):
            end = min(begin + query_block, len(query))
            queries = query[begin:end, head] / math.sqrt(query.shape[-1])
            positions = torch.arange(
                begin + first_position, end + first_position, device=q.device
            )
            maximum = torch.full(
                (end - begin,), -torch.inf, device=q.device, dtype=torch.float64
            )
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(queries)
            for left in range(0, end + first_position, key_block):
                right = min(left + key_block, end + first_position)
                scores = queries @ key[left:right, kv_head].T
                visible = (
                    torch.arange(left, right, device=q.device)[None]
                    <= positions[:, None]
                )
                scores = scores.masked_fill(~visible, -torch.inf)
                new_maximum = torch.maximum(maximum, scores.amax(-1))
                # Every query sees key zero in the first block; later blocks can
                # be entirely masked for some rows, with a finite running max.
                correction = (maximum - new_maximum).exp()
                probability = (scores - new_maximum[:, None]).exp()
                numerator = (
                    numerator * correction[:, None]
                    + probability @ value[left:right, kv_head]
                )
                denominator = denominator * correction + probability.sum(-1)
                maximum = new_maximum
            outputs[begin:end, head] = numerator / denominator[:, None]
    return outputs


def delta_recurrence(q, k, v, log_decay, beta, initial):
    """Direct token recurrence, independent of the chunked equation below."""
    repeat = v.shape[1] // q.shape[1]
    query = l2(q).repeat_interleave(repeat, 1) / math.sqrt(q.shape[-1])
    key = l2(k).repeat_interleave(repeat, 1)
    state = initial.double().clone()
    outputs = []
    for token in range(len(q)):
        state = state * log_decay[token].double().exp()[:, None, None]
        residual = v[token].double() - (state @ key[token, :, :, None]).squeeze(-1)
        state = (
            state
            + beta[token].double()[:, None, None]
            * residual[:, :, None]
            * key[token, :, None]
        )
        outputs.append((state @ query[token, :, :, None]).squeeze(-1))
    return torch.stack(outputs) if outputs else v.double().clone(), state


def delta_chunk(
    q, k, v, log_decay, beta, initial, *, chunk=64, normalization_dtype=None
):
    """FP64 triangular solve equivalent to the token delta recurrence."""
    if chunk <= 0:
        raise ValueError("delta chunk must be positive")
    repeat = v.shape[1] // q.shape[1]
    query, key = l2(q), l2(k)
    if normalization_dtype is not None:
        query, key = (
            store(query, normalization_dtype).double(),
            store(key, normalization_dtype).double(),
        )
    query, key = query.repeat_interleave(repeat, 1), key.repeat_interleave(repeat, 1)
    values, decay, update = v.double(), log_decay.double(), beta.double()
    state, outputs = initial.double().clone(), []
    for begin in range(0, len(q), chunk):
        end = min(begin + chunk, len(q))
        qs, ks, vs = (x[begin:end].transpose(0, 1) for x in (query, key, values))
        cumulative = decay[begin:end].T.cumsum(-1)
        # Mask before exponentiation. Large negative decays would otherwise
        # overflow the unused upper triangle and create inf*0 -> NaN.
        difference = cumulative[:, :, None] - cumulative[:, None, :]
        lower = torch.ones(
            (end - begin, end - begin), device=q.device, dtype=torch.bool
        ).tril()
        multiplier = difference.masked_fill(~lower, -torch.inf).exp()
        interaction = (
            (ks @ ks.transpose(-1, -2)) * multiplier * update[begin:end].T[:, :, None]
        )
        triangular = interaction.tril(-1) + torch.eye(
            end - begin, device=q.device, dtype=torch.float64
        )
        residual = vs - (ks @ state.transpose(-1, -2)) * cumulative.exp()[:, :, None]
        delta = torch.linalg.solve_triangular(
            triangular, residual * update[begin:end].T[:, :, None], upper=False
        )
        result = (qs @ state.transpose(-1, -2)) * cumulative.exp()[:, :, None]
        result += ((qs @ ks.transpose(-1, -2)) * multiplier) @ delta
        outputs.append((result / math.sqrt(q.shape[-1])).transpose(0, 1))
        carried = ks * (cumulative[:, -1:] - cumulative).exp()[:, :, None]
        state = (
            state * cumulative[:, -1].exp()[:, None, None]
            + delta.transpose(-1, -2) @ carried
        )
    return torch.cat(outputs) if outputs else values.clone(), state


def distribution_metrics(reference_logits, candidate_logits, targets):
    """All normalization, NLL and KL arithmetic starts in FP64."""
    reference = reference_logits.double().log_softmax(-1)
    candidate = candidate_logits.double().log_softmax(-1)
    rows = torch.arange(len(targets), device=reference.device)
    return {
        "kl": (reference.exp() * (reference - candidate)).sum(-1),
        "reference_nll": -reference[rows, targets],
        "candidate_nll": -candidate[rows, targets],
    }


def delta_gates(b, a, a_log, bias):
    shifted = a.double() + bias.double()
    softplus = torch.logaddexp(torch.zeros_like(shifted), shifted)
    return -a_log.double().exp() * softplus, b.double().sigmoid()
