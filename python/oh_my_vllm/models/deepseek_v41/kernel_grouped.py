import torch

from .kernel_native import _projection_module
from .kernel_projection import _projection_output_dtype, fp4_gemm
from .kernel_weights import _grouped_pointer_arrays, _packed_weight
from .quantization import act_quant


def grouped_fp4_gemm(
    values,
    counts,
    weights,
    *,
    counts_list=None,
    pointers=None,
    max_rows=None,
    output_dtype=torch.bfloat16,
    quantized=None,
):
    """Apply one FP4 projection to each consecutive routed expert segment.

    The native operator compacts the nonempty groups on the device. Without
    `counts_list`, the caller keeps the counts on the device and `max_rows`
    defaults to every input row. With `counts_list`, the host list is validated
    and supplies `max_rows`. `pointers` supplies prepared device pointer arrays
    for the full expert order. `quantized` supplies a shared encoded FP8 input
    and UE8M0 K32 scales for native execution.
    """
    output_dtype = _projection_output_dtype(output_dtype) or torch.bfloat16
    if values.ndim != 2:
        raise ValueError("grouped FP4 input must be a matrix")
    if counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("grouped FP4 counts must be a one-dimensional integer tensor")
    if len(weights) != counts.numel():
        raise ValueError("grouped FP4 weights must match the expert count")
    if counts.device != values.device:
        raise ValueError("grouped FP4 counts must use the input device")
    if counts_list is None:
        if max_rows is None:
            max_rows = values.size(0)
    else:
        if len(counts_list) != counts.numel():
            raise ValueError("grouped FP4 count metadata must match the count tensor")
        if any(type(count) is not int or count < 0 for count in counts_list):
            raise ValueError("grouped FP4 counts must be non-negative")
        if sum(counts_list) != values.size(0):
            raise ValueError("grouped FP4 counts must cover every input row")
        if max(counts_list):
            max_rows = max(counts_list)
        else:
            return values.new_empty((0, weights[0][0].size(0)), dtype=output_dtype)
    if values.size(0) == 0:
        return values.new_empty((0, weights[0][0].size(0)), dtype=output_dtype)
    if values.is_cuda and all(
        weight.is_cuda and weight.dtype == torch.float4_e2m1fn_x2
        for weight, _ in weights
    ):
        return _grouped_native(
            values,
            counts,
            weights,
            counts_list,
            pointers,
            max_rows,
            output_dtype,
            quantized,
        )

    return _grouped_reference(
        values, counts, weights, counts_list, output_dtype, quantized
    )


def _grouped_native(
    values, counts, weights, counts_list, pointers, max_rows, output_dtype, quantized
):
    if quantized is None:
        encoded, input_scale = act_quant(
            values,
            32,
            "ue8m0",
            torch.float8_e8m0fnu,
        )
    else:
        encoded, input_scale = quantized
        if (
            encoded.shape != values.shape
            or encoded.device != values.device
            or encoded.dtype != torch.float8_e4m3fn
            or input_scale.shape != (values.size(0), values.size(1) // 32)
            or input_scale.device != values.device
            or input_scale.dtype != torch.float8_e8m0fnu
        ):
            raise ValueError(
                "shared grouped FP4 input must use matching FP8 and K32 scales"
            )
    if pointers is None:
        packed = [_packed_weight(weight, scale, 4) for weight, scale in weights]
        weight_ptrs, scale_ptrs = _grouped_pointer_arrays(packed, values.device)
        output_width = packed[0][0].size(0)
    else:
        weight_ptrs, scale_ptrs = pointers
        output_width = weights[0][0].size(0)
    width = values.size(-1)
    output = torch.empty(
        (values.size(0), output_width), dtype=output_dtype, device=values.device
    )
    module = _projection_module()
    scratch = torch.empty(
        (
            module.grouped_workspace_size(
                values.size(0),
                len(weights),
                max_rows,
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
        counts.to(torch.int32),
        output,
        scratch,
        max_rows,
    )
    return output


def _grouped_reference(values, counts, weights, counts_list, output_dtype, quantized):
    if quantized is not None:
        raise ValueError("shared grouped FP4 input requires native CUDA execution")
    counts_list = counts.tolist() if counts_list is None else counts_list
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
                    output_dtype=output_dtype,
                )
            )
        first += count
    if outputs:
        return torch.cat(outputs, dim=0)
    return values.new_empty((0, weights[0][0].size(0)), dtype=output_dtype)


def grouped_fp4_gemm_padded(
    encoded,
    input_scales,
    counts,
    weights,
    *,
    pointers,
    status,
    max_rows=None,
    output_dtype=torch.bfloat16,
    output=None,
    scratch=None,
):
    """Project the device-count prefix and zero every output row after the prefix.

    The caller retains packed weights and scales until completion or graph release.
    Invalid counts or a prior status error zero the complete output. The caller
    resets status at Execute start and rejects it before cache commit.
    The workspace query requires capacity <= len(weights) * max_rows.
    """
    if encoded.ndim != 2 or counts.ndim != 1:
        raise ValueError("padded FP4 inputs need a matrix and an expert count vector")
    if not encoded.is_cuda:
        raise ValueError("padded FP4 execution requires CUDA inputs")
    if encoded.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("padded FP4 activations must be encoded FP8 values")
    if input_scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("padded FP4 activations require UE8M0 scales")
    if not weights or len(weights) != counts.numel():
        raise ValueError("padded FP4 weights must match the expert count")
    output_dtype = _projection_output_dtype(output_dtype) or torch.bfloat16
    capacity, width = encoded.shape
    first_weight = weights[0][0]
    if first_weight.ndim != 2:
        raise ValueError("padded FP4 weights must be matrices")
    outputs = first_weight.size(0)
    for weight, scale in weights:
        if (
            weight.dtype != torch.float4_e2m1fn_x2
            or weight.shape != (outputs, width // 2)
            or scale.shape != (outputs, width // 32)
            or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or weight.device != encoded.device
            or scale.device != encoded.device
        ):
            raise ValueError("padded FP4 weight and scale shapes must match the input")
    max_rows = max(1, capacity) if max_rows is None else max_rows
    if output is None:
        output = torch.empty(
            (capacity, outputs), dtype=output_dtype, device=encoded.device
        )
    elif output.dtype != output_dtype or output.shape != (capacity, outputs):
        raise ValueError("padded FP4 output storage must match its shape and dtype")
    module = _projection_module()
    if scratch is None:
        scratch = torch.empty(
            module.grouped_workspace_size(
                capacity, len(weights), max_rows, outputs, width
            ),
            dtype=torch.uint8,
            device=encoded.device,
        )
    module.grouped_project_padded(
        encoded.view(torch.uint8),
        input_scales.view(torch.uint8),
        *pointers,
        counts,
        output,
        scratch,
        max_rows,
        status,
    )
    return output
