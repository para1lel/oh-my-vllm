import os

import torch

from ...operators.sparse_attention.plan import scratch_bytes as sparse_scratch_bytes
from .kernel_graph import _CAPTURED_TENSORS, retain_cached_tensor
from .kernel_graph import capture_cached_tensors as capture_cached_tensors
from .kernel_graph import retain_graph_tensors as retain_graph_tensors
from .kernel_grouped import grouped_fp4_gemm as grouped_fp4_gemm
from .kernel_grouped import grouped_fp4_gemm_padded as grouped_fp4_gemm_padded
from .kernel_hc import hc_expand as hc_expand
from .kernel_hc import hc_prepare as hc_prepare
from .kernel_hc import hc_split_sinkhorn as hc_split_sinkhorn
from .kernel_hc import make_hc_workspace as make_hc_workspace
from .kernel_hc import verify_bf16_projection as verify_bf16_projection
from .kernel_native import NATIVE_ROW_LIMIT
from .kernel_projection import bf16_linear as bf16_linear
from .kernel_projection import dot_linear as dot_linear
from .kernel_projection import fp4_gemm as fp4_gemm
from .kernel_projection import fp8_gemm as fp8_gemm
from .kernel_projection import fp32_project as fp32_project
from .kernel_projection import markov_linear as markov_linear
from .kernel_transform import engram_mix as engram_mix
from .kernel_transform import ordered_expert_sum as ordered_expert_sum
from .kernel_transform import ordered_expert_sum_padded as ordered_expert_sum_padded
from .kernel_transform import require_padded_status as require_padded_status
from .kernel_transform import rms as rms
from .kernel_transform import route as route
from .kernel_transform import swiglu as swiglu
from .kernel_weights import invalidate_packed_weights as invalidate_packed_weights
from .packed_cache import PackedView
from .quantization import act_quant as act_quant
from .quantization import fp4_act_quant as fp4_act_quant

_SPARSE_ATTENTION_MODULE = None


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
        # Shared status is mutable across inference and ordinary Eager calls.
        with torch.inference_mode(False):
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


def _sparse_attention_module():
    global _SPARSE_ATTENTION_MODULE
    if _SPARSE_ATTENTION_MODULE is None:
        from ...operators.runtime.native import load

        _SPARSE_ATTENTION_MODULE, _, _ = load("sparse_attention", ("attend",))
    return _SPARSE_ATTENTION_MODULE


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
