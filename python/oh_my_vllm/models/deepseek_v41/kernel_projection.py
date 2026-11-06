import torch
import torch.nn.functional as F

from .kernel_native import (
    _hyperconnection_module,
    _native_available,
    _projection_module,
)
from .kernel_weights import _packed_weight, _round128, _scale_bytes
from .quantization import fp4_values


def _scale_values(scales):
    if scales.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        return scales.float()
    return scales.float()


def _projection_output_dtype(output_dtype):
    if output_dtype not in (None, torch.bfloat16, torch.float32):
        raise ValueError("projection output dtype must be BF16 or FP32")
    return output_dtype


def fp8_gemm(
    a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128, *, output_dtype=None
):
    _projection_output_dtype(output_dtype)
    if (
        _native_available(a)
        and block_size == 32
        and a_s.dtype == torch.float8_e8m0fnu
        and b_s.dtype == torch.float8_e8m0fnu
    ):
        return _projection_gemm(
            a, a_s, b, b_s, bits=8, output_dtype=output_dtype or torch.bfloat16
        )
    a_value = a.float() * _scale_values(a_s).repeat_interleave(block_size, dim=-1)
    weight_scales = _scale_values(b_s).repeat_interleave(block_size, dim=0)[: b.size(0)]
    b_value = b.float() * weight_scales.repeat_interleave(block_size, dim=-1)
    return F.linear(a_value, b_value).to(output_dtype or torch.get_default_dtype())


def fp4_gemm(
    a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128, *, output_dtype=None
):
    _projection_output_dtype(output_dtype)
    if (
        _native_available(a)
        and act_block_size in (32, 128)
        and a_s.dtype == torch.float8_e8m0fnu
        and b_s.dtype == torch.float8_e8m0fnu
    ):
        if act_block_size == 128:
            a_s = a_s.view(torch.uint8).repeat_interleave(4, dim=-1).view(a_s.dtype)
        return _projection_gemm(
            a, a_s, b, b_s, bits=4, output_dtype=output_dtype or torch.bfloat16
        )
    if b.dtype == torch.float4_e2m1fn_x2:
        b = fp4_values(b)
    a_value = a.float() * _scale_values(a_s).repeat_interleave(act_block_size, dim=-1)
    b_value = b.float() * _scale_values(b_s).repeat_interleave(32, dim=-1)
    return F.linear(a_value, b_value).to(output_dtype or torch.get_default_dtype())


def _logical_row_count(rows, logical_rows):
    if (
        type(logical_rows) is not int
        or not 0 <= logical_rows <= rows
        or (logical_rows and rows % logical_rows)
    ):
        raise ValueError("logical row count must divide the physical row count")
    return logical_rows or rows


def _source_steps(values, logical_rows, operation):
    if logical_rows == values.size(0):
        return operation(values)
    grouped = values.unflatten(0, (logical_rows, -1))
    return torch.stack(
        [operation(grouped[:, step].contiguous()) for step in range(grouped.size(1))],
        dim=1,
    ).flatten(0, 1)


def fp32_project(values, weight, *, logical_rows=0, workspace=None):
    if values.ndim != 2 or weight.ndim != 2 or values.size(1) != weight.size(1):
        raise ValueError("FP32 projection inputs must be compatible matrices")
    if values.size(1) != 5120:
        raise ValueError("FP32 projection width must be 5120")
    rows = values.size(0)
    source_rows = _logical_row_count(rows, logical_rows)
    if (
        values.is_cuda
        and weight.device == values.device
        and values.dtype == weight.dtype == torch.float32
        and weight.size(0) >= 1
        and values.is_contiguous()
        and weight.is_contiguous()
    ):
        module = _hyperconnection_module()
        if workspace is None:
            workspace = torch.empty(
                module.workspace_size(rows), dtype=torch.uint8, device=values.device
            )
        output = torch.empty(
            (rows, weight.size(0)), dtype=torch.float32, device=values.device
        )
        module.fp32_project(values, weight, output, workspace, logical_rows)
        return output
    return _source_steps(
        values.float(), source_rows, lambda x: F.linear(x, weight.float())
    )


def bf16_linear(values, weight, *, logical_rows=0):
    """Apply one BF16 projection with the released cuBLAS configuration."""
    if weight.ndim != 2 or values.size(-1) != weight.size(1):
        raise ValueError("BF16 projection inputs must be compatible matrices")
    width = values.size(-1)
    rows = values.numel() // width
    outputs = weight.size(0)
    source_rows = _logical_row_count(rows, logical_rows)
    if not (
        values.is_cuda
        and weight.device == values.device
        and values.dtype == weight.dtype == torch.bfloat16
        and values.is_contiguous()
        and weight.is_contiguous()
    ):
        output = _source_steps(
            values.float().reshape(rows, width),
            source_rows,
            lambda x: F.linear(x, weight.float()),
        )
        return output.to(torch.bfloat16).reshape(*values.shape[:-1], outputs)
    module = _hyperconnection_module()
    output = torch.empty((rows, outputs), dtype=torch.bfloat16, device=values.device)
    if logical_rows == 1 and 1 < rows <= 32768 and width == 5120 and outputs == 512:
        module.bf16_linear_rows1(values.reshape(rows, width), weight, output)
    elif logical_rows and logical_rows < rows:
        steps = rows // logical_rows
        grouped = values.reshape(logical_rows, steps, width)
        out = output.view(logical_rows, steps, outputs)
        step_out = torch.empty(
            (logical_rows, outputs), dtype=torch.bfloat16, device=values.device
        )
        for step in range(steps):
            module.bf16_linear(grouped[:, step].contiguous(), weight, step_out)
            out[:, step] = step_out
    else:
        module.bf16_linear(values.reshape(rows, width), weight, output)
    return output.reshape(*values.shape[:-1], outputs)


def _native_fp32_linear(values, weight, function):
    width = values.size(-1)
    rows = values.numel() // width
    outputs = weight.size(0)
    if not (
        values.is_cuda
        and weight.device == values.device
        and values.dtype == weight.dtype == torch.float32
        and weight.is_contiguous()
    ):
        return F.linear(values.float(), weight.float())
    module = _hyperconnection_module()
    output = torch.empty((rows, outputs), dtype=torch.float32, device=values.device)
    function(
        module,
        values.reshape(rows, width).contiguous(),
        weight,
        output,
    )
    return output.reshape(*values.shape[:-1], outputs)


def markov_linear(values, weight):
    return _native_fp32_linear(
        values, weight, lambda module, x, w, out: module.markov_linear(x, w, out)
    )


def dot_linear(values, weight):
    return _native_fp32_linear(
        values, weight, lambda module, x, w, out: module.dot_linear(x, w, out)
    )


def _projection_gemm(a, a_s, b, b_s, bits, output_dtype=torch.bfloat16):
    _projection_output_dtype(output_dtype)
    module = _projection_module()
    packed_weight, packed_scales, width = _packed_weight(b, b_s, bits)
    values = a.reshape(-1, a.size(-1))
    encoded = values.view(torch.uint8)
    scales = a_s.reshape(-1, a_s.size(-1)).view(torch.uint8)
    output = torch.empty(
        (values.size(0), b.size(0)), dtype=output_dtype, device=a.device
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
