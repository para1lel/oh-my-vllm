"""PyTorch fallback kernels for the DeepSeek model seam.

The adapter uses native CUDA workspaces for production execution. These functions keep
model construction and small reference cases usable before a workspace is attached.
"""

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ...operators.hyperconnection.api import FUNCTIONS as HYPERCONNECTION_FUNCTIONS
from ...operators.sparse_attention.plan import scratch_bytes as sparse_scratch_bytes
from ...operators.transform.api import FUNCTIONS as TRANSFORM_FUNCTIONS
from .packed_cache import PackedView
from .quantization import act_quant, fp4_values
from .quantization import fp4_act_quant as fp4_act_quant

_CAPTURED_TENSORS = ContextVar("deepseek_graph_cached_tensors", default=None)


@contextmanager
def capture_cached_tensors():
    owners = {}
    token = _CAPTURED_TENSORS.set(owners)
    try:
        yield owners
    finally:
        _CAPTURED_TENSORS.reset(token)


def retain_graph_tensors(value):
    owners = _CAPTURED_TENSORS.get()
    if owners is not None:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, torch.Tensor):
                owners[id(item)] = item
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
    return value


def retain_cached_tensor(operation):
    @wraps(operation)
    def selected(*args, **kwargs):
        return retain_graph_tensors(operation(*args, **kwargs))

    for name in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(operation, name):
            setattr(selected, name, getattr(operation, name))
    return selected


class _PinnedMetadataUploader:
    """Reusable pinned hosts for stream-ordered metadata uploads."""

    def __init__(self, device, dtype=torch.int32, capacity=384):
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        self.slots = []
        self.cursor = 0

    def _new_slot(self):
        # Pin on the CPU explicitly: the caller can set a CUDA default device.
        host = torch.empty(
            (self.capacity,), dtype=self.dtype, device="cpu", pin_memory=True
        )
        device = torch.empty((self.capacity,), dtype=self.dtype, device=self.device)
        return [host, device, torch.cuda.Event()]

    def _slot(self):
        for offset in range(len(self.slots)):
            index = (self.cursor + offset) % len(self.slots)
            slot = self.slots[index]
            if slot[2].query():
                self.cursor = index + 1
                return slot
        if len(self.slots) < 64:
            slot = self._new_slot()
            self.slots.append(slot)
            self.cursor = len(self.slots)
            return slot
        slot = self.slots[self.cursor % len(self.slots)]
        slot[2].synchronize()
        self.cursor += 1
        return slot

    def upload(self, values):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("captured metadata needs a capture-owned host buffer")
        slot = self._slot()
        count = len(values)
        slot[0][:count].copy_(torch.tensor(values, dtype=self.dtype, device="cpu"))
        slot[1][:count].copy_(slot[0][:count], non_blocking=True)
        slot[2].record(torch.cuda.current_stream(self.device))
        return slot[1][:count]


_POINTER_UPLOADS = {}
_CAPTURE_POINTER_HOSTS = {}


@retain_cached_tensor
def _upload_pointer_array(values, role, device):
    """Upload one expert pointer array without a blocking host copy."""
    if torch.cuda.is_current_stream_capturing():
        # A captured copy must read pinned memory that outlives the graph.
        key = (role, tuple(values))
        host = _CAPTURE_POINTER_HOSTS.get(key)
        if host is None:
            host = torch.empty((len(values),), dtype=torch.int64, pin_memory=True)
            host.copy_(torch.tensor(values, dtype=torch.int64, device="cpu"))
            _CAPTURE_POINTER_HOSTS[key] = host
        retain_graph_tensors(host)
        result = torch.empty((len(values),), dtype=torch.int64, device=device)
        result.copy_(host, non_blocking=True)
        return result
    key = (role, str(device))
    uploader = _POINTER_UPLOADS.get(key)
    if uploader is None:
        uploader = _PinnedMetadataUploader(device, dtype=torch.int64)
        _POINTER_UPLOADS[key] = uploader
    return uploader.upload(values)


_PROJECTION_MODULE = None
_ROUTING_MODULE = None
_ACTIVATION_MODULE = None
_HYPERCONNECTION_MODULE = None
_TRANSFORM_MODULE = None
_SPARSE_ATTENTION_MODULE = None
_PACKED_WEIGHTS = {}
NATIVE_ROW_LIMIT = 384
_SPARSE_NATIVE_DISABLED = (
    os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "1") != "1"
)
# Set only when the native call rejected a step. The environment switch keeps
# long queries on the native path; a rejected native call does not.
_SPARSE_NATIVE_FAILED = False
_SPARSE_STATUS = {}


def _disable_sparse_native():
    global _SPARSE_NATIVE_DISABLED, _SPARSE_NATIVE_FAILED
    _SPARSE_NATIVE_FAILED = True
    _SPARSE_NATIVE_DISABLED = True


@retain_cached_tensor
def _sparse_status_accumulator(device):
    key = str(device)
    accumulator = _SPARSE_STATUS.get(key)
    if accumulator is None:
        accumulator = torch.zeros((), dtype=torch.bool, device=device)
        _SPARSE_STATUS[key] = accumulator
    return accumulator


def reset_sparse_status():
    """Clear the sparse attention status before a model step."""
    for accumulator in _SPARSE_STATUS.values():
        accumulator.zero_()


def require_sparse_status():
    """Stop the step when native sparse attention rejected a head tile.

    The native operator zeroes a rejected head tile and flags its status. The
    caller checks the accumulated status once per step, before the worker
    commits cache state. A rejected step is not recoverable in place because
    earlier layers already wrote their cache rows.
    """
    for accumulator in _SPARSE_STATUS.values():
        if accumulator.item():
            _disable_sparse_native()
            raise RuntimeError(
                "native sparse attention rejected the step; no state is committed"
            )


class _PackedWeightCacheEntry(NamedTuple):
    packed: tuple
    raw_weight: object
    raw_scales: object
    versions: tuple


def _native_available(x):
    return x.is_cuda


def _projection_module():
    global _PROJECTION_MODULE
    if _PROJECTION_MODULE is None:
        from ...operators.projection.api import FUNCTIONS
        from ...operators.runtime.native import load

        _PROJECTION_MODULE, _, _ = load("projection", FUNCTIONS)
    return _PROJECTION_MODULE


def _routing_module():
    global _ROUTING_MODULE
    if _ROUTING_MODULE is None:
        from ...operators.routing.api import FUNCTIONS
        from ...operators.runtime.native import load

        _ROUTING_MODULE, _, _ = load("routing", FUNCTIONS)
    return _ROUTING_MODULE


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
            "hyperconnection", HYPERCONNECTION_FUNCTIONS
        )
    return _HYPERCONNECTION_MODULE


def _transform_module():
    global _TRANSFORM_MODULE
    if _TRANSFORM_MODULE is None:
        from ...operators.runtime.native import load

        _TRANSFORM_MODULE, _, _ = load("transform", TRANSFORM_FUNCTIONS)
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


def invalidate_packed_weights(weight, scales):
    storages = {
        (value.device, value.untyped_storage().data_ptr())
        for value in (weight, scales)
        if value is not None
    }
    for key, entry in tuple(_PACKED_WEIGHTS.items()):
        if any(
            (value.device, value.untyped_storage().data_ptr()) in storages
            for value in (entry.raw_weight, entry.raw_scales)
        ):
            del _PACKED_WEIGHTS[key]


@retain_cached_tensor
def _packed_weight(weight, scales, bits):
    # Inference tensors have no version counter. Change them through Linear's
    # state-loading entry point, which invalidates entries for shared storage.
    versions = tuple(
        None if value.is_inference() else value._version for value in (weight, scales)
    )
    key = (
        weight.data_ptr(),
        scales.data_ptr(),
        bits,
        weight.device.index,
        (weight.dtype, weight.shape, weight.stride()),
        (scales.dtype, scales.shape, scales.stride()),
    )
    cached = _PACKED_WEIGHTS.get(key)
    if cached is not None and cached.versions == versions:
        return cached.packed
    module = _projection_module()
    raw_weight = weight.view(torch.uint8)
    raw_scales = scales.view(torch.uint8)
    outputs, stored_width = raw_weight.shape
    width = stored_width * (8 // bits)
    padded = _round128(width) if bits == 4 else width
    if (
        padded == width
        and raw_weight.is_contiguous()
        and raw_weight.data_ptr() % (64 if bits == 4 else 16) == 0
        and (
            raw_weight.data_ptr() + raw_weight.numel() <= raw_scales.data_ptr()
            or raw_scales.data_ptr() + raw_scales.numel() <= raw_weight.data_ptr()
        )
    ):
        packed_weight = raw_weight
    else:
        packed_weight = torch.empty(
            (outputs, padded * bits // 8), dtype=torch.uint8, device=weight.device
        )
    packed_scales = torch.empty(
        (_scale_bytes(outputs, width),), dtype=torch.uint8, device=weight.device
    )
    module.pack_weights(raw_weight, raw_scales, packed_weight, packed_scales, bits)
    packed = (packed_weight, packed_scales, width)
    _PACKED_WEIGHTS[key] = _PackedWeightCacheEntry(packed, weight, scales, versions)
    return packed


def _grouped_pointer_arrays(packed, device):
    weight_ptrs = _upload_pointer_array(
        [weight.data_ptr() for weight, _, _ in packed], "weight", device
    )
    scale_ptrs = _upload_pointer_array(
        [scales.data_ptr() for _, scales, _ in packed], "scale", device
    )
    return weight_ptrs, scale_ptrs


def _static_pointer_arrays(packed, device):
    """Device pointer arrays over every expert; prepared weights never move.

    The pinned host tensors stay with the result set, so the asynchronous copy
    never reads released storage.
    """
    packed = [_packed_weight(weight, scale, 4) for weight, scale in packed]
    weight_host = torch.tensor(
        [weight.data_ptr() for weight, _, _ in packed],
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    scale_host = torch.tensor(
        [scales.data_ptr() for _, scales, _ in packed],
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    weight_ptrs = weight_host.to(device, non_blocking=True)
    scale_ptrs = scale_host.to(device, non_blocking=True)
    weight_ptrs._packed_owners = scale_ptrs._packed_owners = tuple(packed)
    return weight_ptrs, scale_ptrs, weight_host, scale_host


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


_BF16_PROBE_DIGESTS = {
    "2": "f5a8ea40f4c74b2e",
    "64": "fb8f993fae7373f6",
    "1024": "4fdc4d03ceb8a9f9",
}
_BF16_PROBE_DONE = False


def _bf16_probe_digest(device, rows):
    width = 5120
    weight = torch.arange(512 * width, device=device, dtype=torch.int64)
    weight = ((weight.remainder(2048) - 1024).to(torch.float32) / 1024).bfloat16()
    values = torch.arange(rows * width, device=device, dtype=torch.int64)
    values = ((values.remainder(1024) - 512).to(torch.float32) / 512).bfloat16()
    output = bf16_linear(values.view(rows, width), weight.view(512, width))
    return hashlib.sha256(output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[
        :16
    ]


def verify_bf16_projection(device):
    """Fail when the limited cuBLAS handle stops matching the accepted layout."""
    global _BF16_PROBE_DONE
    if _BF16_PROBE_DONE:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("verify the BF16 projection outside Graph capture")
    actual = {str(rows): _bf16_probe_digest(device, rows) for rows in (2, 64, 1024)}
    different = {
        rows: (value, _BF16_PROBE_DIGESTS[rows])
        for rows, value in actual.items()
        if value != _BF16_PROBE_DIGESTS[rows]
    }
    if different:
        raise RuntimeError(
            "BF16 projection probe differs from the accepted cuBLAS layout: "
            + repr(different)
        )
    _BF16_PROBE_DONE = True


def make_hc_workspace(device, max_rows=NATIVE_ROW_LIMIT):
    if type(max_rows) is not int or not 0 < max_rows <= NATIVE_ROW_LIMIT:
        raise ValueError("DeepSeek mHC workspace rows must be between one and 384")
    module = _hyperconnection_module()
    verify_bf16_projection(device)
    workspace = {
        "mixed": torch.empty((max_rows, 24), dtype=torch.float32, device=device),
        "post": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "comb": torch.empty((max_rows, 4, 4), dtype=torch.float32, device=device),
        "collapsed": torch.empty((max_rows, 5120), dtype=torch.bfloat16, device=device),
        "next_pre_a": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "next_pre_b": torch.empty((max_rows, 4), dtype=torch.float32, device=device),
        "flat": torch.empty((max_rows, 20480), dtype=torch.float32, device=device),
        "statistics": torch.empty((max_rows, 2), dtype=torch.float32, device=device),
        "blas": torch.empty(
            (module.workspace_size(max_rows),), dtype=torch.uint8, device=device
        ),
        "expanded_a": torch.empty(
            (max_rows, 4, 5120), dtype=torch.bfloat16, device=device
        ),
        "expanded_b": torch.empty(
            (max_rows, 4, 5120), dtype=torch.bfloat16, device=device
        ),
    }
    module.hc_warmup(workspace["collapsed"])
    return workspace


def hc_prepare(
    residual,
    incoming_pre,
    projection,
    scale,
    base,
    *,
    workspace=None,
    next_pre=None,
    logical_rows=0,
):
    """Collapse one mHC residual and prepare the coefficients for expansion.

    The native path uses the released DeepSeek shape.
    The reference path keeps CPU checks independent of the native module.
    """
    if residual.size(-2) != 4 or residual.size(-1) < 1:
        raise ValueError("DeepSeek mHC residual must contain four nonempty copies")
    rows = residual.numel() // (4 * residual.size(-1))
    source_rows = _logical_row_count(rows, logical_rows)
    if not (
        _hc_native_available(residual, incoming_pre, projection, scale, base)
        and incoming_pre.dtype == torch.float32
        and projection.dtype == torch.float32
        and scale.dtype == torch.float32
        and base.dtype == torch.float32
    ):
        width = residual.size(-1)
        leading = residual.shape[:-2]
        flat_residual = residual.float().reshape(-1, 4, width)
        incoming = incoming_pre.float().reshape(-1, 4)
        flat = flat_residual.flatten(1)
        mixes = _source_steps(
            flat,
            source_rows,
            lambda x: (
                F.linear(x, projection.float())
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
            ),
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
    prepare = module.hc_prepare_grouped if logical_rows else module.hc_prepare
    grouping = (logical_rows,) if logical_rows else ()
    if rows > NATIVE_ROW_LIMIT:
        mixed = torch.empty((rows, 24), dtype=torch.float32, device=residual.device)
        flat = torch.empty((rows, 20480), dtype=torch.float32, device=residual.device)
        statistics = torch.empty((rows, 2), dtype=torch.float32, device=residual.device)
        blas = torch.empty(
            (module.workspace_size(rows),), dtype=torch.uint8, device=residual.device
        )
        module.hc_warmup(collapsed)
        prepare(
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
            flat,
            statistics,
            blas,
            *grouping,
        )
    else:
        prepare(
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
            workspace["flat"][:rows],
            workspace["statistics"][:rows],
            workspace["blas"][: module.workspace_size(rows)],
            *grouping,
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
        and (
            (
                isinstance(compressed, PackedView)
                and compressed.size(0) == query.size(0)
                and compressed.size(2) == 512
                and compressed.history.dtype == torch.bfloat16
                and compressed.device == query.device
            )
            or (
                isinstance(compressed, torch.Tensor)
                and compressed.ndim == 3
                and tuple(compressed.shape[-1:]) == (512,)
                and compressed.dtype == torch.bfloat16
                and compressed.device == query.device
                and compressed.is_contiguous()
            )
        )
    )


def _sparse_attn_native(query, sink, indices, window, compressed):
    batch, sequence, heads, _ = query.shape
    window_length, compressed_length = window.size(1), compressed.size(1)
    window_rows = batch * window_length
    packed = isinstance(compressed, PackedView)
    compressed_rows = (
        compressed.history.codes.size(1) if packed else batch * compressed_length
    )
    window_scales = torch.empty(
        (window_rows, 0), dtype=torch.uint8, device=query.device
    )
    compressed_scales = (
        compressed.history.scales.view(compressed_rows, 32)
        if packed
        else torch.empty((compressed_rows, 0), dtype=torch.uint8, device=query.device)
    )
    compressed_table = (
        compressed.history.codes.view(compressed_rows, 256)
        if packed
        else compressed.reshape(compressed_rows, 512)
    )
    ids = indices.to(torch.int32).clone()
    batch_offset = torch.arange(batch, device=query.device, dtype=torch.int32).view(
        batch, 1, 1
    )
    valid_ids = ids >= 0
    window_ids = valid_ids & (ids < window_length)
    if packed:
        logical = ids - window_length
        physical = compressed.native_rows(
            logical.clamp(0, max(0, compressed_length - 1))
        )
        compressed_ids = torch.where(
            logical < compressed_length,
            physical + window_rows,
            torch.full_like(physical, window_rows + compressed_rows),
        ).to(torch.int32)
    else:
        compressed_ids = (
            ids + window_rows + batch_offset * compressed_length - window_length
        )
    ids = torch.where(
        window_ids,
        ids + batch_offset * window_length,
        torch.where(valid_ids, compressed_ids, ids),
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
    scratch = torch.empty(
        (sparse_scratch_bytes(query_rows.size(0), heads, ids.size(-1)),),
        dtype=torch.uint8,
        device=query.device,
    )
    module.attend(
        query_rows,
        sink,
        ids,
        window.reshape(window_rows, 512),
        window_scales,
        compressed_table,
        compressed_scales,
        scratch,
        output,
        status,
    )
    _sparse_status_accumulator(query.device).logical_or_(status.any())
    return output.reshape(batch, sequence, heads, 512)


def native_sparse_graph_enabled():
    return not _SPARSE_NATIVE_DISABLED and not _SPARSE_NATIVE_FAILED


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
        return _sparse_attn_native(
            q, attn_sink, topk_idxs, native_window, native_compressed
        )
    if _CAPTURED_TENSORS.get() is not None or (
        q.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "DeepSeek reference sparse attention requires Eager execution"
        )
    if kv is None:
        if isinstance(native_compressed, PackedView):
            native_compressed = native_compressed.materialize(shared=False)
        kv = torch.cat((native_window, native_compressed), dim=1)
    maximum = torch.full(q.shape[:-1], -1e30, dtype=torch.float32, device=q.device)
    denominator = torch.zeros_like(maximum)
    accumulated = torch.zeros_like(q, dtype=torch.float32)
    use_bf16_mma = q.is_cuda and q.dtype == kv.dtype == torch.bfloat16
    query = q.flatten(0, 1) if use_bf16_mma else q.float()
    for start in range(0, topk_idxs.size(-1), 64):
        selected = topk_idxs[..., start : start + 64]
        valid = selected >= 0
        gathered = torch.gather(
            kv.unsqueeze(1).expand(-1, q.size(1), -1, -1),
            2,
            selected.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, -1, kv.size(-1)),
        )
        gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
        if use_bf16_mma:
            scores = torch.bmm(
                query, gathered.flatten(0, 1).transpose(1, 2), out_dtype=torch.float32
            ).view(*q.shape[:-1], -1)
        else:
            gathered = gathered.float()
            scores = torch.einsum("bshd,bskd->bshk", query, gathered)
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        previous_maximum = maximum
        maximum = torch.maximum(maximum, scores.amax(-1))
        rescale = (previous_maximum - maximum).exp()
        probabilities = (scores - maximum.unsqueeze(-1)).exp()
        denominator = denominator * rescale + probabilities.sum(-1)
        # Released attention rounds each unnormalized 64-slot block before V.
        rounded = probabilities.to(torch.bfloat16)
        accumulated = accumulated * rescale.unsqueeze(-1)
        if use_bf16_mma:
            accumulated = torch.baddbmm(
                accumulated.flatten(0, 1),
                rounded.flatten(0, 1),
                gathered.flatten(0, 1),
                out_dtype=torch.float32,
            ).view_as(accumulated)
        else:
            accumulated = accumulated + torch.einsum(
                "bshk,bskd->bshd", rounded.float(), gathered
            )
    denominator = denominator + (attn_sink.view(1, 1, -1) - maximum).exp()
    safe_denominator = torch.where(denominator == 0, 1, denominator)
    return (accumulated / safe_denominator.unsqueeze(-1)).to(q.dtype)


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
    comb = comb + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
