from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .engram import EngramLayout
from .kernel import (
    NATIVE_ROW_LIMIT,
    hc_expand,
    hc_prepare,
    hc_split_sinkhorn,
    make_hc_workspace,
)
from .model_args import ModelArgs
from .model_attention import Attention
from .model_engram import Engram
from .model_linear import RMSNorm, set_dtype
from .model_moe import MoE


class Block(nn.Module):
    """
    A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer
    input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes`
    derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    attention_cls = Attention

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        self._hc_workspace = None

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        logical_rows=0,
    ):
        """
        x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns
        the
        pre / post / comb coefficients, split out of one projection of the flattened
        stream.
        """
        if logical_rows:
            rows = x.size(0) * x.size(1)
            if not 0 < logical_rows <= rows or rows % logical_rows:
                raise ValueError("HC logical rows must divide the physical rows")
            grouped = x.reshape(logical_rows, rows // logical_rows, *x.shape[2:])
            values = [
                self.hc_mixes(
                    grouped[:, step : step + 1].contiguous(), hc_fn, hc_scale, hc_base
                )
                for step in range(grouped.size(1))
            ]
            return tuple(
                torch.cat([value[index] for value in values], dim=1).reshape(
                    *x.shape[:2], *values[0][index].shape[2:]
                )
                for index in range(3)
            )
        # normalized over the whole flattened hc*d stream, one statistic per token
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """
        Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] ->
        [b,s,d]
        """
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        """
        Expand the sublayer output back to hc copies and mix the residual in through
        `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] ->
        [b,s,hc,d]
        """
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        *attn_args,
        logical_rows=0,
        position: torch.Tensor | None = None,
        plan=None,
        valid_rows=None,
        dispatch_combine=None,
        prefill_eager=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `pre_mix` collapses the hc_mult copies down to one input for this block's
        attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention
        uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias).
        """
        options = {"logical_rows": logical_rows} if logical_rows else {}
        ffn_options = dict(options)
        if prefill_eager:
            ffn_options["prefill_eager"] = True
        if plan is not None or valid_rows is not None or dispatch_combine is not None:
            ffn_options.update(
                plan=plan, valid_rows=valid_rows, dispatch_combine=dispatch_combine
            )
        attention_options = (
            {**options, "position": position} if position is not None else options
        )
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.size(-2) == 4
            and x.size(-1) == 5120
        ):
            return self._forward_native(
                x,
                start_pos,
                pre_mix,
                image_mask,
                attn_args,
                options,
                attention_options,
                ffn_options,
                logical_rows,
            )

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, **options
        )
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x, **options)
        x = self.attn(x, start_pos, *attn_args, **attention_options)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, **options
        )
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x, **options)
        x = self.ffn(x, image_mask, **ffn_options)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _native_workspace(self, x):
        rows = x.numel() // (4 * 5120)
        workspace = None
        next_pre_a = None
        next_pre_b = None
        expanded_a = None
        expanded_b = None
        if rows <= NATIVE_ROW_LIMIT:
            current_rows = (
                0
                if self._hc_workspace is None
                else self._hc_workspace["collapsed"].size(0)
            )
            if (
                current_rows < rows
                or self._hc_workspace["collapsed"].device != x.device
            ):
                self._hc_workspace = make_hc_workspace(x.device, max_rows=rows)
            workspace = self._hc_workspace
            next_pre_a = workspace["next_pre_a"]
            next_pre_b = workspace["next_pre_b"]
            expanded_a = workspace["expanded_a"]
            expanded_b = workspace["expanded_b"]

        return workspace, next_pre_a, next_pre_b, expanded_a, expanded_b

    def _forward_native(
        self,
        x,
        start_pos,
        pre_mix,
        image_mask,
        attn_args,
        options,
        attention_options,
        ffn_options,
        logical_rows,
    ):
        workspace, next_pre_a, next_pre_b, expanded_a, expanded_b = (
            self._native_workspace(x)
        )
        residual = x
        x, attn_pre, attn_post, attn_comb = hc_prepare(
            residual,
            pre_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            workspace=workspace,
            next_pre=next_pre_a,
            logical_rows=logical_rows,
        )
        x = self.attn_norm(x, **options)
        x = self.attn(x, start_pos, *attn_args, **attention_options)
        x = hc_expand(
            residual,
            x,
            attn_post,
            attn_comb,
            workspace=workspace,
            output=expanded_a,
        )

        residual = x
        x, ffn_pre, ffn_post, ffn_comb = hc_prepare(
            residual,
            attn_pre,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            workspace=workspace,
            next_pre=next_pre_b,
            logical_rows=logical_rows,
        )
        x = self.ffn_norm(x, **options)
        if logical_rows:
            ffn_options["blas_workspace"] = (
                None if workspace is None else workspace["blas"]
            )
        x = self.ffn(x, image_mask, **ffn_options)
        x = hc_expand(
            residual,
            x,
            ffn_post,
            ffn_comb,
            workspace=workspace,
            output=expanded_b,
        )
        return x, ffn_pre


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix
