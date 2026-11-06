from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from . import model_context as context
from .kernel import act_quant, fp4_gemm, fp8_gemm, invalidate_packed_weights, rms
from .routed_state import _invalidate_routed_owners


@contextmanager
def set_dtype(dtype):
    """
    Temporarily override torch's default dtype, restoring it even if the body raises.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class ParallelEmbedding(nn.Module):
    """
    Embedding sharded along the vocab dimension. Each rank holds vocab_size //
    world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial
    embeddings.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % context.world_size == 0, (
            "Vocabulary size must be divisible by world size "
            f"(world_size={context.world_size})"
        )
        self.part_vocab_size = vocab_size // context.world_size
        self.vocab_start_idx = (
            context.rank if context.world_size > 1 else 0
        ) * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if context.world_size > 1:
            # ids off this rank read row 0 then get zeroed, so the all_reduce sums one
            # real row
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if context.world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype=None,
) -> torch.Tensor:
    """
    Pick a GEMM from the weight dtype. Quantized weights need a quantized activation,
    and both
    fp4 and fp8 weights take an fp8 one -- for fp4 the kernel handles the mixed
    precision.
    """
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(
            x, context.fp8_block_size, context.scale_fmt, context.scale_dtype
        )
        return fp4_gemm(
            x,
            s,
            weight,
            weight.scale,
            context.scale_dtype,
            act_block_size=context.fp8_block_size,
            output_dtype=output_dtype,
        )
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(
            x, context.fp8_block_size, context.scale_fmt, context.scale_dtype
        )
        return fp8_gemm(
            x,
            s,
            weight,
            weight.scale,
            context.scale_dtype,
            block_size=context.fp8_block_size,
            output_dtype=output_dtype,
        )
    else:
        if output_dtype is not None:
            return F.linear(x.float(), weight.float()).to(output_dtype)
        return F.linear(x, weight)


class Linear(nn.Module):
    """
    bf16, fp8 or fp4 weights. Quantized ones get a `scale`, also attached to `.weight`
    so that
    `linear()` can reach it.
    """

    def __setattr__(self, name, value):
        if name in ("weight", "scale"):
            _invalidate_routed_owners(self)
        super().__setattr__(name, value)

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or context.default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # two values per byte: [out, in] logically, [out, in//2] stored
            self.weight = nn.Parameter(
                torch.empty(
                    out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2
                )
            )
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(
                    out_features,
                    in_features // context.fp4_block_size,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(
                    (out_features + context.fp8_block_size - 1)
                    // context.fp8_block_size,
                    (in_features + context.fp8_block_size - 1)
                    // context.fp8_block_size,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)

    def _invalidate_routed_bindings(self):
        _invalidate_routed_owners(self)

    def _apply(self, fn, recurse=True):
        self._invalidate_routed_bindings()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_routed_bindings()
        if self.scale is not None:
            invalidate_packed_weights(self.weight, self.scale)
        return super()._load_from_state_dict(*args, **kwargs)


_PLAIN_LINEAR_FORWARD = Linear.forward


def _plain_fp8_linear(layer):
    return (
        type(layer) is Linear
        and getattr(layer.forward, "__func__", None) is _PLAIN_LINEAR_FORWARD
        and getattr(layer.forward, "__self__", None) is layer
        and layer.weight.dtype == torch.float8_e4m3fn
        and not layer._forward_hooks
        and not layer._forward_pre_hooks
        and layer._compiled_call_impl is None
        and not nn.modules.module._global_forward_hooks
        and not nn.modules.module._global_forward_pre_hooks
    )


def separate_request_math(values, logical_rows):
    return values.is_cuda and logical_rows == 1 and context.world_size > 1


def request_linear(layer, values, *, separate_requests=False):
    if separate_requests and values.size(0) > 1:
        return torch.cat(
            [layer(values[index : index + 1]) for index in range(values.size(0))]
        )
    return layer(values)


def request_einsum(equation, values, weight, *, separate_requests=False):
    if separate_requests and values.size(0) > 1:
        return torch.cat(
            [
                torch.einsum(equation, values[index : index + 1], weight)
                for index in range(values.size(0))
            ]
        )
    return torch.einsum(equation, values, weight)


class ColumnParallelLinear(Linear):
    """
    Splits the output dim across ranks; each rank's slice of the output is already
    complete.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert out_features % context.world_size == 0, (
            "Output features must be divisible by world size "
            f"(world_size={context.world_size})"
        )
        self.part_out_features = out_features // context.world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """
    Splits the reduction dim, so each rank holds a partial sum: hence the fp32
    all_reduce, with
    the bias added only after it.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert in_features % context.world_size == 0, (
            "Input features must be divisible by world size "
            f"(world_size={context.world_size})"
        )
        self.part_in_features = in_features // context.world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)
        self._local_projection = None
        if context.world_size == 1 and context.expert_world_size == 4:
            from .local_tp import LocalRowProjection

            self._local_projection = LocalRowProjection(4)

    def _apply(self, fn, recurse=True):
        if self._local_projection is not None:
            self._local_projection.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        if self._local_projection is not None:
            self._local_projection.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (
            linear(x, self.weight)
            if self._local_projection is None
            else self._local_projection(x, self.weight, self.scale, linear)
        )
        if context.world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, *, logical_rows=0):
        dtype = x.dtype
        return rms(x, self.weight, self.eps, logical_rows=logical_rows).to(dtype)
