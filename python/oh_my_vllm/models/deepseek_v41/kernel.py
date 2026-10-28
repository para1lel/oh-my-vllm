"""PyTorch fallback kernels for the DeepSeek model seam.

The adapter uses native CUDA workspaces for production execution. These functions keep
model construction and small reference cases usable before a workspace is attached.
"""

import os

import torch
import torch.nn.functional as F

_PROJECTION_MODULE = None
_ROUTING_MODULE = None
_ACTIVATION_MODULE = None
_HYPERCONNECTION_MODULE = None
_TRANSFORM_MODULE = None
_SPARSE_ATTENTION_MODULE = None
_PACKED_WEIGHTS = {}
NATIVE_ROW_LIMIT = 384
_SPARSE_NATIVE_DISABLED = (
    os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "0") != "1"
)
_SPARSE_NATIVE_FAILED = False


def _native_available(x):
    return x.is_cuda


def _projection_module():
    global _PROJECTION_MODULE
    if _PROJECTION_MODULE is None:
        from ...operators.runtime.native import load

        _PROJECTION_MODULE, _, _ = load(
            "projection",
            (
                "pack_weights",
                "workspace_size",
                "project",
                "grouped_workspace_size",
                "grouped_project",
            ),
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


def _hyperconnection_module():
    global _HYPERCONNECTION_MODULE
    if _HYPERCONNECTION_MODULE is None:
        from ...operators.runtime.native import load

        _HYPERCONNECTION_MODULE, _, _ = load(
            "hyperconnection", ("hc_prepare", "hc_expand")
        )
    return _HYPERCONNECTION_MODULE


def _transform_module():
    global _TRANSFORM_MODULE
    if _TRANSFORM_MODULE is None:
        from ...operators.runtime.native import load

        _TRANSFORM_MODULE, _, _ = load("transform", ("rms", "rope"))
    return _TRANSFORM_MODULE


def _sparse_attention_module():
    global _SPARSE_ATTENTION_MODULE
    if _SPARSE_ATTENTION_MODULE is None:
        from ...operators.runtime.native import load

        _SPARSE_ATTENTION_MODULE, _, _ = load("sparse_attention", ("attend",))
    return _SPARSE_ATTENTION_MODULE


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


def grouped_fp4_gemm(values, counts, weights, *, counts_list=None):
    """Apply one FP4 projection to each consecutive routed expert segment."""
    if values.ndim != 2:
        raise ValueError("grouped FP4 input must be a matrix")
    if counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("grouped FP4 counts must be a one-dimensional integer tensor")
    if len(weights) != counts.numel():
        raise ValueError("grouped FP4 weights must match the expert count")
    if counts.device != values.device:
        raise ValueError("grouped FP4 counts must use the input device")
    counts_list = counts.tolist() if counts_list is None else counts_list
    if len(counts_list) != counts.numel():
        raise ValueError("grouped FP4 count metadata must match the count tensor")
    if any(type(count) is not int or count < 0 for count in counts_list):
        raise ValueError("grouped FP4 counts must be non-negative")
    if sum(counts_list) != values.size(0):
        raise ValueError("grouped FP4 counts must cover every input row")

    active = [
        (count, weight, scale)
        for count, (weight, scale) in zip(counts_list, weights, strict=True)
        if count
    ]
    if not active:
        return values.new_empty((0, weights[0][0].size(0)))
    if values.is_cuda and all(
        weight.is_cuda and weight.dtype == torch.float4_e2m1fn_x2
        for _, weight, _ in active
    ):
        encoded, input_scale = act_quant(
            values,
            32,
            "ue8m0",
            torch.float8_e8m0fnu,
        )
        packed = [_packed_weight(weight, scale, 4) for _, weight, scale in active]
        output_width = packed[0][0].size(0)
        width = values.size(-1)
        weight_ptrs = torch.tensor(
            [weight.data_ptr() for weight, _, _ in packed],
            dtype=torch.int64,
            device=values.device,
        )
        scale_ptrs = torch.tensor(
            [scale.data_ptr() for _, scale, _ in packed],
            dtype=torch.int64,
            device=values.device,
        )
        active_counts = torch.tensor(
            [count for count, _, _ in active], dtype=torch.int32, device=values.device
        )
        output = torch.empty(
            (values.size(0), output_width), dtype=torch.bfloat16, device=values.device
        )
        module = _projection_module()
        scratch = torch.empty(
            (
                module.grouped_workspace_size(
                    values.size(0),
                    len(active),
                    max(count for count, _, _ in active),
                    output_width,
                    width,
                ),
            ),
            dtype=torch.uint8,
            device=values.device,
        )
        module.grouped_project(
            encoded.view(torch.uint8),
            input_scale.view(torch.uint8),
            weight_ptrs,
            scale_ptrs,
            active_counts,
            output,
            scratch,
            max(count for count, _, _ in active),
        )
        return output

    outputs = []
    first = 0
    for count, (weight, scale) in zip(counts_list, weights, strict=True):
        if count:
            encoded, input_scale = act_quant(
                values[first : first + count],
                32,
                "ue8m0",
                torch.float8_e8m0fnu,
            )
            outputs.append(
                fp4_gemm(
                    encoded,
                    input_scale,
                    weight,
                    scale,
                    torch.float8_e8m0fnu,
                    act_block_size=32,
                )
            )
        first += count
    if outputs:
        return torch.cat(outputs, dim=0)
    return values.new_empty((0, weights[0][0].size(0)))


def rms(values, weight, epsilon):
    """Apply RMS normalization with the native full-width path when available."""
    if (
        values.is_cuda
        and values.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and values.size(-1) in (512, 1024, 1280, 5120)
        and values.is_contiguous()
        and weight.is_contiguous()
        and weight.device == values.device
    ):
        shape = values.shape
        rows = values.numel() // values.size(-1)
        inputs = values.reshape(rows, values.size(-1))
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        output = torch.empty_like(inputs)
        _transform_module().rms(inputs, weight, output, epsilon)
        return output.reshape(shape)
    normalized = values.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + epsilon
    )
    return (normalized * weight.float()).to(values.dtype)


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


def _hc_native_available(residual, *inputs):
    if not (
        residual.is_cuda
        and residual.dtype == torch.bfloat16
        and residual.ndim >= 3
        and tuple(residual.shape[-2:]) == (4, 5120)
        and residual.is_contiguous()
    ):
        return False
    for value in inputs:
        if (
            value is None
            or not value.is_cuda
            or value.device != residual.device
            or not value.is_contiguous()
        ):
            return False
    return True


def make_hc_workspace(device, max_rows=NATIVE_ROW_LIMIT):
    if type(max_rows) is not int or not 0 < max_rows <= NATIVE_ROW_LIMIT:
        raise ValueError("DeepSeek mHC workspace rows must be between one and 384")
    return {
        "mixed": torch.empty((max_rows, 24), dtype=torch.float32, device=device),
        "post": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "comb": torch.empty((max_rows, 4, 4), dtype=torch.float32, device=device),
        "collapsed": torch.empty((max_rows, 5120), dtype=torch.bfloat16, device=device),
        "next_pre_a": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "next_pre_b": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "expanded_a": torch.empty(
            (max_rows, 4, 5120), dtype=torch.bfloat16, device=device
        ),
        "expanded_b": torch.empty(
            (max_rows, 4, 5120), dtype=torch.bfloat16, device=device
        ),
    }


def hc_prepare(
    residual,
    incoming_pre,
    projection,
    scale,
    base,
    *,
    workspace=None,
    next_pre=None,
):
    """Collapse one mHC residual and prepare the coefficients for expansion.

    The native path uses the released DeepSeek shape.
    The reference path keeps CPU checks independent of the native module.
    """
    if not (
        _hc_native_available(residual, incoming_pre, projection, scale, base)
        and incoming_pre.dtype == torch.float32
        and projection.dtype == torch.float32
        and scale.dtype == torch.float32
        and base.dtype == torch.float32
    ):
        if residual.size(-2) != 4:
            raise ValueError("DeepSeek mHC residual must contain four copies")
        width = residual.size(-1)
        leading = residual.shape[:-2]
        flat_residual = residual.float().reshape(-1, 4, width)
        incoming = incoming_pre.float().reshape(-1, 4)
        flat = flat_residual.flatten(1)
        mixes = F.linear(flat, projection.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-20
        )
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale.float(), base.float(), hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        collapsed = torch.sum(incoming.unsqueeze(-1) * flat_residual, dim=1).to(
            residual.dtype
        )
        return (
            collapsed.reshape(*leading, width),
            pre.reshape(*leading, 4),
            post.reshape(*leading, 4),
            comb.reshape(*leading, 4, 4),
        )

    leading = residual.shape[:-2]
    rows = residual.numel() // (4 * 5120)
    residual_rows = residual.reshape(rows, 4, 5120)
    incoming_rows = incoming_pre.reshape(rows, 4)
    if not residual_rows.is_contiguous():
        residual_rows = residual_rows.contiguous()
    if not incoming_rows.is_contiguous():
        incoming_rows = incoming_rows.contiguous()
    if rows <= NATIVE_ROW_LIMIT:
        workspace = workspace or make_hc_workspace(residual.device)
        if any(
            value.device != residual.device or value.size(0) < rows
            for value in workspace.values()
        ):
            raise ValueError("DeepSeek mHC workspace is incompatible with the input")
        post = workspace["post"][:rows]
        comb = workspace["comb"][:rows]
        collapsed = workspace["collapsed"][:rows]
    else:
        workspace = None
        post = torch.empty((rows, 4), dtype=torch.float32, device=residual.device)
        comb = torch.empty((rows, 4, 4), dtype=torch.float32, device=residual.device)
        collapsed = torch.empty(
            (rows, 5120), dtype=torch.bfloat16, device=residual.device
        )
    if next_pre is None:
        if rows <= NATIVE_ROW_LIMIT:
            next_pre = workspace["next_pre_a"][:rows]
        else:
            next_pre = torch.empty(
                (rows, 4), dtype=torch.float32, device=residual.device
            )
    else:
        if (
            next_pre.device != residual.device
            or next_pre.dtype != torch.float32
            or next_pre.size(0) < rows
            or tuple(next_pre.shape[1:]) != (4,)
        ):
            raise ValueError("DeepSeek mHC pre-mix workspace is incompatible")
        next_pre = next_pre[:rows]
    module = _hyperconnection_module()
    if rows > NATIVE_ROW_LIMIT:
        mixed = torch.empty((rows, 24), dtype=torch.float32, device=residual.device)
        module.hc_prepare(
            residual_rows,
            incoming_rows,
            projection,
            scale,
            base,
            mixed,
            post,
            comb,
            collapsed,
            next_pre,
        )
    else:
        module.hc_prepare(
            residual_rows,
            incoming_rows,
            projection,
            scale,
            base,
            workspace["mixed"][:rows],
            post,
            comb,
            collapsed,
            next_pre,
        )
    return (
        collapsed.reshape(*leading, 5120),
        next_pre.reshape(*leading, 4),
        post.reshape(*leading, 4),
        comb.reshape(*leading, 4, 4),
    )


def hc_expand(residual, branch, post, comb, *, workspace=None, output=None):
    """Expand an mHC branch back into four residual copies."""
    if not (
        _hc_native_available(residual, branch, post, comb)
        and branch.dtype == torch.bfloat16
        and post.dtype == torch.float32
        and comb.dtype == torch.float32
    ):
        if residual.size(-2) != 4:
            raise ValueError("DeepSeek mHC residual must contain four copies")
        leading = residual.shape[:-2]
        residual_rows = residual.float().reshape(-1, 4, residual.size(-1))
        branch_rows = branch.float().reshape(-1, residual.size(-1))
        post_rows = post.float().reshape(-1, 4)
        comb_rows = comb.float().reshape(-1, 4, 4)
        expanded = post_rows.unsqueeze(-1) * branch_rows.unsqueeze(-2)
        expanded = expanded + torch.sum(
            comb_rows.unsqueeze(-1) * residual_rows.unsqueeze(-2), dim=1
        )
        return expanded.to(residual.dtype).reshape(*leading, 4, residual.size(-1))

    leading = residual.shape[:-2]
    rows = residual.numel() // (4 * 5120)
    residual_rows = residual.reshape(rows, 4, 5120)
    branch_rows = branch.reshape(rows, 5120)
    post_rows = post.reshape(rows, 4)
    comb_rows = comb.reshape(rows, 4, 4)
    if output is None:
        output = (
            None
            if rows > NATIVE_ROW_LIMIT or workspace is None
            else workspace.get("expanded_a")
        )
    if output is None:
        output_rows = torch.empty_like(residual_rows)
    else:
        if (
            output.device != residual.device
            or output.dtype != torch.bfloat16
            or output.size(0) < rows
            or tuple(output.shape[1:]) != (4, 5120)
        ):
            raise ValueError("DeepSeek mHC output workspace is incompatible")
        output_rows = output[:rows]
    module = _hyperconnection_module()
    if rows > NATIVE_ROW_LIMIT:
        module.hc_expand(
            residual_rows,
            branch_rows,
            post_rows,
            comb_rows,
            output_rows,
        )
    else:
        workspace = workspace or make_hc_workspace(residual.device)
        if any(
            value.device != residual.device or value.size(0) < rows
            for value in workspace.values()
        ):
            raise ValueError("DeepSeek mHC workspace is incompatible with the input")
        module.hc_expand(
            residual_rows,
            branch_rows,
            post_rows,
            comb_rows,
            output_rows,
        )
    return output_rows.reshape(*leading, 4, 5120)


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


def _pack_fp8_kv(values):
    scales = _block_scales(values, 32)
    scales = torch.ldexp(torch.ones_like(scales), torch.ceil(torch.log2(scales / 448)))
    groups = values.float().reshape(*values.shape[:-1], -1, 32)
    encoded = (
        (groups / scales.unsqueeze(-1))
        .clamp(-448, 448)
        .reshape_as(values)
        .to(torch.float8_e4m3fn)
    )
    scale_codes = scales.to(torch.float8_e8m0fnu).view(torch.uint8).clamp_max(254)
    return encoded.view(torch.uint8).reshape(*values.shape[:-1], 512), (
        scale_codes.reshape(*values.shape[:-1], 16)
    )


def _pack_fp4_kv(values):
    if values.size(-2) == 0:
        empty_shape = (*values.shape[:-1], 256)
        scales_shape = (*values.shape[:-1], 32)
        return (
            torch.empty(empty_shape, dtype=torch.uint8, device=values.device),
            torch.empty(scales_shape, dtype=torch.uint8, device=values.device),
        )
    scales = _block_scales(values, 16).clamp(0.001953125, 448)
    groups = values.float().reshape(*values.shape[:-1], -1, 16)
    normalized = (groups / scales.unsqueeze(-1)).clamp(-6, 6).reshape_as(values)
    absolute = normalized.abs()
    levels = torch.bucketize(
        absolute,
        torch.tensor(
            (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
            device=values.device,
            dtype=absolute.dtype,
        ),
    )
    codes = levels.to(torch.uint8) | (normalized.signbit().to(torch.uint8) << 3)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scale_codes = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return packed.reshape(*values.shape[:-1], 256), scale_codes.reshape(
        *values.shape[:-1], 32
    )


def _sparse_native_available(query, indices, window, compressed):
    return (
        query.is_cuda
        and query.dtype == torch.bfloat16
        and query.ndim == 4
        and query.size(-1) == 512
        and query.size(-2) in (16, 64)
        and query.is_contiguous()
        and indices.ndim == 3
        and indices.size(-1) <= 640
        and indices.is_contiguous()
        and indices.device == query.device
        and window.ndim == 3
        and tuple(window.shape[-1:]) == (512,)
        and window.dtype == torch.bfloat16
        and window.device == query.device
        and window.is_contiguous()
        and compressed.ndim == 3
        and tuple(compressed.shape[-1:]) == (512,)
        and compressed.dtype == torch.bfloat16
        and compressed.device == query.device
        and compressed.is_contiguous()
    )


def _sparse_attn_native(query, sink, indices, window, compressed):
    batch, sequence, heads, _ = query.shape
    window_length, compressed_length = window.size(1), compressed.size(1)
    packed_window, window_scales = _pack_fp8_kv(window)
    packed_compressed, compressed_scales = _pack_fp4_kv(compressed)
    window_rows = batch * window_length
    compressed_rows = batch * compressed_length
    ids = indices.to(torch.int32).clone()
    batch_offset = torch.arange(batch, device=query.device, dtype=torch.int32).view(
        batch, 1, 1
    )
    valid_ids = ids >= 0
    window_ids = valid_ids & (ids < window_length)
    ids = torch.where(
        window_ids,
        ids + batch_offset * window_length,
        torch.where(
            valid_ids,
            ids + window_rows + batch_offset * compressed_length - window_length,
            torch.full_like(ids, -1),
        ),
    )
    query_rows = query.reshape(batch * sequence, heads, 512)
    ids = ids.reshape(batch * sequence, ids.size(-1))
    output = torch.empty_like(query_rows)
    status = torch.empty(
        (query_rows.size(0), heads // 16), dtype=torch.uint8, device=query.device
    )
    if not sink.is_contiguous():
        sink = sink.contiguous()
    module = _sparse_attention_module()
    packed_window = packed_window.reshape(window_rows, 512)
    window_scales = window_scales.reshape(window_rows, 16)
    packed_compressed = packed_compressed.reshape(compressed_rows, 256)
    compressed_scales = compressed_scales.reshape(compressed_rows, 32)
    module.attend(
        query_rows,
        sink,
        ids,
        packed_window,
        window_scales,
        packed_compressed,
        compressed_scales,
        output,
        status,
    )
    if bool(status.any().item()):
        return None
    return output.reshape(batch, sequence, heads, 512)


def sparse_attn(
    q,
    kv,
    attn_sink,
    topk_idxs,
    softmax_scale,
    *,
    native_window=None,
    native_compressed=None,
):
    global _SPARSE_NATIVE_DISABLED, _SPARSE_NATIVE_FAILED
    long_query = q.numel() // (q.size(-1) * q.size(-2)) > NATIVE_ROW_LIMIT
    if (
        (not _SPARSE_NATIVE_DISABLED or (long_query and not _SPARSE_NATIVE_FAILED))
        and native_window is not None
        and native_compressed is not None
        and attn_sink.is_cuda
        and attn_sink.dtype == torch.float32
        and attn_sink.device == q.device
        and _sparse_native_available(q, topk_idxs, native_window, native_compressed)
    ):
        native = _sparse_attn_native(
            q, attn_sink, topk_idxs, native_window, native_compressed
        )
        if native is not None:
            return native
        _SPARSE_NATIVE_FAILED = True
        _SPARSE_NATIVE_DISABLED = True
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
