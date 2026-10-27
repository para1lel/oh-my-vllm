"""PyTorch fallback kernels for the DeepSeek model seam.

The adapter uses native CUDA workspaces for production execution. These functions keep
model construction and small reference cases usable before a workspace is attached.
"""

import torch
import torch.nn.functional as F

_PROJECTION_MODULE = None
_ROUTING_MODULE = None
_ACTIVATION_MODULE = None
_PACKED_WEIGHTS = {}


def _native_available(x):
    return x.is_cuda


def _projection_module():
    global _PROJECTION_MODULE
    if _PROJECTION_MODULE is None:
        from ...operators.runtime.native import load

        _PROJECTION_MODULE, _, _ = load(
            "projection", ("pack_weights", "workspace_size", "project")
        )
    return _PROJECTION_MODULE


def _routing_module():
    global _ROUTING_MODULE
    if _ROUTING_MODULE is None:
        from ...operators.runtime.native import load

        _ROUTING_MODULE, _, _ = load("routing", ("route",))
    return _ROUTING_MODULE


def _activation_module():
    global _ACTIVATION_MODULE
    if _ACTIVATION_MODULE is None:
        from ...operators.runtime.native import load

        _ACTIVATION_MODULE, _, _ = load("activation", ("swiglu",))
    return _ACTIVATION_MODULE


def _round128(value):
    return (value + 127) // 128 * 128


def _scale_bytes(rows, width):
    return _round128(rows) * (_round128(width) // 32)


def _packed_weight(weight, scales, bits):
    key = (weight.data_ptr(), scales.data_ptr(), bits, weight.device.index)
    packed = _PACKED_WEIGHTS.get(key)
    if packed is not None:
        return packed
    module = _projection_module()
    raw_weight = weight.view(torch.uint8)
    raw_scales = scales.view(torch.uint8)
    outputs, stored_width = raw_weight.shape
    width = stored_width * (8 // bits)
    padded = _round128(width) if bits == 4 else width
    packed_weight = torch.empty(
        (outputs, padded * bits // 8), dtype=torch.uint8, device=weight.device
    )
    packed_scales = torch.empty(
        (_scale_bytes(outputs, width),), dtype=torch.uint8, device=weight.device
    )
    module.pack_weights(raw_weight, raw_scales, packed_weight, packed_scales, bits)
    packed = (packed_weight, packed_scales, width)
    _PACKED_WEIGHTS[key] = packed
    return packed


def _block_scales(x, block_size):
    width = x.size(-1)
    if width % block_size:
        raise ValueError("quantization width must be divisible by the block size")
    view = x.float().reshape(*x.shape[:-1], width // block_size, block_size)
    return view.abs().amax(-1).clamp_min(1e-4)


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    scales = _block_scales(x, block_size)
    if scale_fmt is not None:
        scales = torch.ldexp(
            torch.ones_like(scales), torch.ceil(torch.log2(scales / 448))
        )
    encoded = (
        x.float().reshape(*x.shape[:-1], -1, block_size) / scales.unsqueeze(-1)
    ).clamp(-448, 448)
    if inplace:
        x.copy_((encoded * scales.unsqueeze(-1)).reshape_as(x).to(x.dtype))
        return x
    return encoded.reshape_as(x).to(torch.float8_e4m3fn), scales.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    scales = _block_scales(x, block_size)
    encoded = (
        x.float().reshape(*x.shape[:-1], -1, block_size) / scales.unsqueeze(-1)
    ).clamp(-6, 6)
    if inplace:
        x.copy_((encoded * scales.unsqueeze(-1)).reshape_as(x).to(x.dtype))
        return x
    return encoded.reshape_as(x).to(x.dtype), scales.to(scale_dtype)


def _scale_values(scales):
    if scales.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        return scales.float()
    return scales.float()


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    if _native_available(a) and block_size == 32:
        return _projection_gemm(a, a_s, b, b_s, bits=8)
    a_value = a.float() * _scale_values(a_s).repeat_interleave(block_size, dim=-1)
    b_value = b.float() * _scale_values(b_s).repeat_interleave(block_size, dim=-1)
    return F.linear(a_value, b_value).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    if _native_available(a) and act_block_size in (32, 128):
        return _projection_gemm(a, a_s, b, b_s, bits=4)
    if b.dtype == torch.float4_e2m1fn_x2:
        b = b.to(torch.bfloat16)
        b = b.repeat_interleave(2, dim=-1)
    a_value = a.float() * _scale_values(a_s).repeat_interleave(act_block_size, dim=-1)
    b_value = b.float() * _scale_values(b_s).repeat_interleave(32, dim=-1)
    return F.linear(a_value, b_value).to(torch.get_default_dtype())


def _projection_gemm(a, a_s, b, b_s, bits):
    module = _projection_module()
    packed_weight, packed_scales, width = _packed_weight(b, b_s, bits)
    values = a.reshape(-1, a.size(-1))
    encoded = values.view(torch.uint8)
    scales = a_s.reshape(-1, a_s.size(-1)).view(torch.uint8)
    output = torch.empty(
        (values.size(0), b.size(0)), dtype=torch.bfloat16, device=a.device
    )
    padded = torch.empty(
        (
            values.size(0),
            _round128(width) if bits == 4 and _round128(width) != width else 0,
        ),
        dtype=torch.uint8,
        device=a.device,
    )
    packed_input_scales = torch.empty(
        (_scale_bytes(values.size(0), width),), dtype=torch.uint8, device=a.device
    )
    scratch = torch.empty(
        (module.workspace_size(values.size(0), b.size(0), width, bits),),
        dtype=torch.uint8,
        device=a.device,
    )
    module.project(
        encoded,
        scales,
        packed_weight,
        packed_scales,
        output,
        padded,
        packed_input_scales,
        scratch,
        width,
        bits,
    )
    return output.reshape(*a.shape[:-1], b.size(0))


def swiglu(gate, up, route=None, limit=10):
    if not _native_available(gate):
        value = gate.float()
        up_value = up.float()
        if limit:
            value = value.clamp_max(limit)
            up_value = up_value.clamp(-limit, limit)
        result = F.silu(value) * up_value
        if route is not None:
            result = result * route.unsqueeze(-1)
        return result.to(gate.dtype)
    output = torch.empty_like(gate)
    route_value = (
        torch.empty((0,), dtype=torch.float32, device=gate.device)
        if route is None
        else route.float().reshape(-1)
    )
    native_limit = int(limit)
    if native_limit != limit:
        raise ValueError("native SwiGLU limit must be an integer")
    _activation_module().swiglu(
        gate.to(torch.bfloat16),
        up.to(torch.bfloat16),
        route_value,
        output,
        native_limit,
        route is not None,
    )
    return output


def route(logits, bias, bias_vl, image_mask):
    if not _native_available(logits):
        scores = F.softplus(logits.float()).sqrt()
        selected_bias = bias_vl if image_mask is not None else bias
        _, ids = (scores + selected_bias).topk(
            3 if logits.size(-1) == 128 else 6, dim=-1
        )
        weights = scores.gather(1, ids)
        return weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.5, ids
    topk = 3 if logits.size(-1) == 128 else 6
    ids = torch.empty((logits.size(0), topk), dtype=torch.int32, device=logits.device)
    coefficients = torch.empty(
        (logits.size(0), topk), dtype=torch.float32, device=logits.device
    )
    if image_mask is None:
        image_mask = torch.zeros(
            (logits.size(0),), dtype=torch.uint8, device=logits.device
        )
    else:
        image_mask = image_mask.to(torch.uint8)
    bias_vl = bias if bias_vl is None else bias_vl
    _routing_module().route(
        logits.float(), bias.float(), bias_vl.float(), image_mask, ids, coefficients
    )
    return coefficients, ids


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    indices = topk_idxs.clamp_min(0).long()
    gathered = torch.gather(
        kv.unsqueeze(1).expand(-1, q.size(1), -1, -1),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, kv.size(-1)),
    )
    valid = topk_idxs >= 0
    scores = (
        torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    )
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    scores = torch.cat(
        [scores, attn_sink.view(1, 1, -1, 1).expand(q.size(0), q.size(1), -1, 1)],
        dim=-1,
    )
    weights = torch.softmax(scores, dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", weights, gathered.float()).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    values = mixes.float()
    pre = torch.sigmoid(values[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        values[..., hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = values[..., 2 * hc_mult :].reshape(*values.shape[:-1], hc_mult, hc_mult)
    comb = torch.softmax(
        comb * hc_scale[2] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult), dim=-1
    )
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
