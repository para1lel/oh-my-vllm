import hashlib

import torch
import torch.nn.functional as F

from .kernel_native import NATIVE_ROW_LIMIT, _hyperconnection_module
from .kernel_projection import _logical_row_count, _source_steps, bf16_linear


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
        return _hc_reference(
            residual, incoming_pre, projection, scale, base, source_rows
        )

    return _hc_prepare_native(
        residual,
        incoming_pre,
        projection,
        scale,
        base,
        rows,
        workspace,
        next_pre,
        logical_rows,
    )


def _hc_reference(residual, incoming_pre, projection, scale, base, source_rows):
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


def _hc_buffers(residual, rows, workspace, next_pre):
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

    return workspace, post, comb, collapsed, next_pre


def _hc_prepare_native(
    residual,
    incoming_pre,
    projection,
    scale,
    base,
    rows,
    workspace,
    next_pre,
    logical_rows,
):
    leading = residual.shape[:-2]
    residual_rows = residual.reshape(rows, 4, 5120)
    incoming_rows = incoming_pre.reshape(rows, 4)
    if not residual_rows.is_contiguous():
        residual_rows = residual_rows.contiguous()
    if not incoming_rows.is_contiguous():
        incoming_rows = incoming_rows.contiguous()
    workspace, post, comb, collapsed, next_pre = _hc_buffers(
        residual, rows, workspace, next_pre
    )
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
