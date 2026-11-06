import torch
import torch.nn.functional as F

from .kernel_native import (
    _activation_module,
    _native_available,
    _routing_module,
    _transform_module,
)
from .kernel_projection import _logical_row_count, _source_steps


def ordered_expert_sum(values, sorted_routes, output, topk):
    """Add BF16 projection rows in expert order, with one unique route per slot."""
    route_rows = torch.full(
        (output.size(0), topk), -1, dtype=torch.int32, device=values.device
    )
    route_rows.view(-1).scatter_(
        0,
        sorted_routes,
        torch.arange(values.size(0), dtype=torch.int32, device=values.device),
    )
    if values.is_cuda:
        _routing_module().expert_sum(values, route_rows, output)
        return
    ordered = route_rows.masked_fill(route_rows < 0, values.size(0)).sort(dim=1).values
    output.zero_()
    for slot in range(topk):
        selected = ordered[:, slot].long()
        active = selected < values.size(0)
        output[active] += values[selected[active]].float()


def ordered_expert_sum_padded(
    values, sorted_routes, counts, output, topk, status, *, route_rows=None
):
    """Use device counts for the live prefix and zero tokens with no local routes."""
    if route_rows is None:
        route_rows = torch.empty(
            (output.size(0), topk), dtype=torch.int32, device=values.device
        )
    if route_rows.shape != (output.size(0), topk):
        raise ValueError("padded expert sum needs one map slot per token and route")
    _routing_module().expert_sum_padded(
        values, sorted_routes, counts, output, route_rows, status
    )
    return output


def require_padded_status(status):
    """Reject invalid device metadata at the step boundary before cache commit."""
    if status.item():
        raise RuntimeError("padded grouped operator rejected device metadata")


def rms(values, weight, epsilon, *, logical_rows=0):
    """Apply RMS normalization with the native full-width path when available."""
    rows = values.numel() // values.size(-1)
    source_rows = _logical_row_count(rows, logical_rows)
    if (
        values.is_cuda
        and values.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and values.size(-1) in (128, 512, 1024, 1280, 5120)
        and values.is_contiguous()
        and weight.is_contiguous()
        and weight.device == values.device
    ):
        shape = values.shape
        inputs = values.reshape(rows, values.size(-1))
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        output = torch.empty_like(inputs)
        if logical_rows:
            _transform_module().rms_grouped(
                inputs, weight, output, epsilon, logical_rows
            )
        else:
            _transform_module().rms(inputs, weight, output, epsilon)
        return output.reshape(shape)

    def normalize(x):
        value = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
        return value * weight.float()

    output = _source_steps(
        values.float().reshape(rows, values.size(-1)), source_rows, normalize
    )
    return output.to(values.dtype).reshape(values.shape)


def _engram_reference(
    residual, kv, q_weight, k_weight, token_mask, epsilon, clamp_value
):
    copies, width = residual.shape[-2:]
    key, value = kv.split([copies * width, width], dim=-1)
    key = key.float().unflatten(-1, (copies, width))
    weight = q_weight.float() * k_weight.float()
    hidden = residual.float()
    inverse = torch.rsqrt(hidden.square().mean(-1) + epsilon) * torch.rsqrt(
        key.square().mean(-1) + epsilon
    )
    dot = (hidden * weight * key).sum(-1) * inverse * width**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
        residual.dtype
    )


def engram_mix(
    residual,
    kv,
    q_weight,
    k_weight,
    token_mask=None,
    *,
    epsilon=1e-20,
    clamp_value=1e-6,
    logical_rows=0,
):
    copies, width = residual.shape[-2:]
    rows = residual.numel() // (copies * width)
    source_rows = _logical_row_count(rows, logical_rows)
    hidden = residual.reshape(rows, copies, width)
    projected = kv.reshape(rows, (copies + 1) * width)
    mask = None if token_mask is None else token_mask.reshape(rows)
    inputs = (residual, kv, q_weight, k_weight)
    if (
        logical_rows
        and copies == 4
        and width == 5120
        and epsilon == 1e-20
        and clamp_value == 1e-6
        and all(
            value.is_cuda
            and value.device == residual.device
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            for value in inputs
        )
    ):
        native_mask = (
            hidden.view(torch.uint8).flatten()[:0]
            if mask is None
            else mask.to(dtype=torch.uint8).contiguous()
        )
        output = torch.empty_like(hidden)
        _transform_module().engram(
            hidden, projected, q_weight, k_weight, native_mask, output, logical_rows
        )
        return output.reshape(residual.shape)
    if source_rows == rows:
        return _engram_reference(
            hidden, projected, q_weight, k_weight, mask, epsilon, clamp_value
        ).reshape(residual.shape)
    steps = rows // source_rows
    hidden = hidden.unflatten(0, (source_rows, steps))
    projected = projected.unflatten(0, (source_rows, steps))
    mask = None if mask is None else mask.unflatten(0, (source_rows, steps))
    output = [
        _engram_reference(
            hidden[:, step].contiguous(),
            projected[:, step].contiguous(),
            q_weight,
            k_weight,
            None if mask is None else mask[:, step],
            epsilon,
            clamp_value,
        )
        for step in range(steps)
    ]
    return torch.stack(output, dim=1).reshape(residual.shape)


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
    gate_value = gate.to(torch.bfloat16)
    if gate_value.data_ptr() % 16:
        gate_value = gate_value.clone()
    up_value = up.to(torch.bfloat16)
    if up_value.data_ptr() % 16:
        up_value = up_value.clone()
    route_value = (
        torch.empty((0,), dtype=torch.float32, device=gate.device)
        if route is None
        else route.float().reshape(-1)
    )
    if route_value.data_ptr() % 16:
        route_value = route_value.clone()
    native_limit = int(limit)
    if native_limit != limit:
        raise ValueError("native SwiGLU limit must be an integer")
    _activation_module().swiglu(
        gate_value,
        up_value,
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
