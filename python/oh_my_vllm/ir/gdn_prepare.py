"""A semantic GDN fork/join: convolution and independent decay/beta gates.

Only the convolution branch writes persistent state. The other branch uses a
BF16 projection, so it does not share the FP8 provider's writable workspace.
All side-stream work joins before this operation returns to its consumers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import Operation, TensorSpec
from .execution import branch_stream, execution_policy
from .fp8 import _LINEAR, linear
from .pointwise import _GATES, delta_gates
from .recurrent import _CONV, causal_conv


def _fake_prepare(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    reads: torch.Tensor,
    writes: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch._check(x.ndim == 2)
    torch._check(qkvz_weight.shape == (16384, x.shape[1]))
    torch._check(qkvz_scale.shape == (128, x.shape[1] // 128))
    torch._check(ba_weight.shape == (96, x.shape[1]))
    torch._check(conv_weight.shape == (10240, 4))
    torch._check(pool.ndim == 3 and pool.shape[1:] == (10240, 3))
    torch._check(sequence_ids.shape == writes.shape == (x.shape[0],))
    torch._check(starts.numel() == reads.numel() + 1)
    torch._check(a_log.shape == bias.shape == (48,))
    return (
        torch.empty((x.shape[0], 10240), device=x.device, dtype=x.dtype),
        torch.empty((x.shape[0], 16384), device=x.device, dtype=x.dtype)[:, 10240:],
        torch.empty((x.shape[0], 48), device=x.device, dtype=torch.float32),
        torch.empty((x.shape[0], 48), device=x.device, dtype=torch.float32),
    )


@torch.library.custom_op("oh_my_vllm_ir::gdn_prepare", mutates_args=("pool",))
def _ir_prepare(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    reads: torch.Tensor,
    writes: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    args = (
        x,
        qkvz_weight,
        qkvz_scale,
        ba_weight,
        conv_weight,
        pool,
        sequence_ids,
        starts,
        reads,
        writes,
        a_log,
        bias,
    )
    return _PREPARE.select(*args).op(*args)


_ir_prepare.register_fake(_fake_prepare)


@torch.library.custom_op("oh_my_vllm_native::gdn_prepare", mutates_args=("pool",))
def _native_prepare(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    reads: torch.Tensor,
    writes: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = _LINEAR.reference(x, qkvz_weight, qkvz_scale, False)
    mixed, z = packed.split((10240, 6144), dim=-1)
    mixed = _CONV.reference(
        mixed, conv_weight, pool, sequence_ids, starts, reads, writes
    )
    decay, beta = _GATES.reference(F.linear(x, ba_weight), a_log, bias)
    return mixed, z, decay, beta


_native_prepare.register_fake(_fake_prepare)


@torch.library.custom_op("oh_my_vllm_kernel::gdn_prepare", mutates_args=("pool",))
def _kernel_prepare(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    reads: torch.Tensor,
    writes: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def gates():
        return delta_gates(F.linear(x, ba_weight), a_log, bias)

    def convolution():
        packed = linear(x, qkvz_weight, qkvz_scale)
        mixed, z = packed.split((10240, 6144), dim=-1)
        return causal_conv(
            mixed, conv_weight, pool, sequence_ids, starts, reads, writes
        ), z

    if not execution_policy().multi_stream:
        mixed, z = convolution()
        decay, beta = gates()
        return mixed, z, decay, beta

    origin = torch.cuda.current_stream(x.device)
    side = branch_stream(x.device.index)
    # wait_stream records events inside capture, yielding a real graph fork.
    side.wait_stream(origin)
    try:
        with torch.cuda.stream(side):
            for tensor in (x, ba_weight, a_log, bias):
                tensor.record_stream(side)
            decay, beta = gates()
        mixed, z = convolution()
    finally:
        # Join on success and failure before a caller restores or releases state.
        origin.wait_stream(side)
    decay.record_stream(origin)
    beta.record_stream(origin)
    return mixed, z, decay, beta


_kernel_prepare.register_fake(_fake_prepare)


def _supported(*args):
    if not all(isinstance(arg, TensorSpec) for arg in args):
        return False
    x, qkvz, scale, ba, conv, pool, ids, starts, reads, writes, log, bias = args
    if x.device.type != "cuda" or any(arg.device != x.device for arg in args):
        return False
    if any(arg.dtype != torch.bfloat16 for arg in (x, ba, conv, pool)):
        return False
    if qkvz.dtype != torch.float8_e4m3fn:
        return False
    if any(arg.dtype != torch.float32 for arg in (scale, log, bias)):
        return False
    if any(
        arg.dtype not in (torch.int32, torch.int64)
        for arg in (ids, starts, reads, writes)
    ):
        return False
    return all(
        arg.stride[-1] == 1 and (arg.ndim != 2 or arg.stride[0] == arg.shape[1])
        for arg in args
    )


_PREPARE = Operation(
    "gdn_prepare", _ir_prepare, _native_prepare, default_priority=("cuda",)
)
_PREPARE.register_impl(
    "cuda",
    _kernel_prepare,
    supports=_supported,
)


def gdn_prepare(
    x,
    qkvz_weight,
    qkvz_scale,
    ba_weight,
    conv_weight,
    pool,
    sequence_ids,
    starts,
    reads,
    writes,
    a_log,
    bias,
):
    return _PREPARE(
        x,
        qkvz_weight,
        qkvz_scale,
        ba_weight,
        conv_weight,
        pool,
        sequence_ids,
        starts,
        reads,
        writes,
        a_log,
        bias,
    )
