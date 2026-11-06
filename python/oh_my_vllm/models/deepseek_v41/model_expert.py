from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from . import model_context as context
from .kernel import act_quant, fp8_gemm, fp32_project
from .model_args import ModelArgs
from .model_linear import ColumnParallelLinear, Linear, _plain_fp8_linear, linear
from .routed_state import _invalidate_routed_owners


class Gate(nn.Module):
    """
    MoE gating. The correction bias steers expert selection only; the routing weights
    come from the
    unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`).
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.dim = args.dim
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = (
            nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
            if args.vision_enabled
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        *,
        logical_rows=0,
        blas_workspace=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        # The released router projects in FP32. Single source rows use the
        # released gemv order; other steps use split chunks.
        if x.is_cuda and self.dim == 5120 and x.ndim == 2 and x.size(-1) == self.dim:
            logits = fp32_project(
                x.float(),
                self.weight.float(),
                logical_rows=logical_rows,
                workspace=blas_workspace,
            )
        else:
            logits = linear(x.float(), self.weight.float())
        logits = logits / self.gate_temp
        if (
            x.is_cuda
            and self.score_func == "sqrtsoftplus"
            and self.gate_temp == 1
            and self.topk in (3, 6)
            and self.route_scale == 1.5
        ):
            from .kernel import route

            bias_vl = self.bias if self.bias_vl is None else self.bias_vl
            return route(logits, self.bias, bias_vl, image_mask)

        scores = logits.float()
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw
        # scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= (
                weights.sum(dim=-1, keepdim=True) + 1e-20
            )  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


def _gather_channels(value: torch.Tensor) -> torch.Tensor:
    if context.world_size == 1:
        return value
    shards = torch.empty(
        (context.world_size, *value.shape), dtype=value.dtype, device=value.device
    )
    dist.all_gather_into_tensor(shards.flatten(0, 1), value.contiguous())
    return shards.movedim(0, -2).flatten(-2)


class Expert(nn.Module):
    """
    One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4
    activations in
    range: the up branch is clamped on both sides, the gate branch only from above.
    """

    def __setattr__(self, name, value):
        if name in ("w1", "w2", "w3"):
            _invalidate_routed_owners(self)
        super().__setattr__(name, value)

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype=None,
        swiglu_limit=0.0,
        tensor_parallel=False,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel
        linear_cls = ColumnParallelLinear if tensor_parallel else Linear
        self.w1 = linear_cls(dim, inter_dim, dtype=dtype)
        self.w2 = linear_cls(inter_dim, dim, dtype=dtype)
        self.w3 = linear_cls(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        from .kernel import swiglu

        dtype = x.dtype
        if _plain_fp8_linear(self.w1) and _plain_fp8_linear(self.w3):
            assert self.w1.bias is None and self.w3.bias is None
            encoded, scales = act_quant(
                x, context.fp8_block_size, context.scale_fmt, context.scale_dtype
            )
            gate, up = (
                fp8_gemm(
                    encoded,
                    scales,
                    projection.weight,
                    projection.weight.scale,
                    context.scale_dtype,
                    block_size=context.fp8_block_size,
                )
                for projection in (self.w1, self.w3)
            )
        else:
            gate = self.w1(x)
            up = self.w3(x)
        if x.is_cuda and gate.size(-1) in (576, 2304):
            product = swiglu(gate, up, weights, self.swiglu_limit)
        else:
            gate = gate.float()
            up = up.float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            product = F.silu(gate) * up
            if weights is not None:
                product = weights * product
            product = product.to(dtype)
        if self.tensor_parallel:
            product = _gather_channels(product)
        return self.w2(product)
