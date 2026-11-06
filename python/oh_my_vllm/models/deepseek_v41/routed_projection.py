import torch
import torch.nn.functional as F

from .kernel import act_quant
from .model_expert import _gather_channels


def project_routed(
    moe,
    local_x,
    local_tokens,
    local_weights,
    local_counts,
    local_count_tensor,
    state,
    prepared,
):
    pointers = state.pointers
    if pointers is None and local_x.is_cuda and prepared.gate[0][0].is_cuda:
        from .kernel_weights import _static_pointer_arrays

        pointers = tuple(
            _static_pointer_arrays(projection, local_x.device)
            for projection in (prepared.gate, prepared.up, prepared.down)
        )
        state.pointers = pointers
    if pointers is None:
        pointers = (None, None, None)
    from .kernel import grouped_fp4_gemm, swiglu

    quantized = None
    if pointers[0] is not None:
        quantized = act_quant(local_x, 32, "ue8m0", torch.float8_e8m0fnu)
    gate = grouped_fp4_gemm(
        local_x,
        local_count_tensor,
        prepared.gate,
        counts_list=local_counts,
        pointers=None if pointers[0] is None else pointers[0][:2],
        quantized=quantized,
    )
    up = grouped_fp4_gemm(
        local_x,
        local_count_tensor,
        prepared.up,
        counts_list=local_counts,
        pointers=None if pointers[1] is None else pointers[1][:2],
        quantized=quantized,
    )
    limit = moe.experts[moe.experts_start_idx].swiglu_limit
    if local_x.is_cuda and gate.size(-1) in (576, 2304):
        product = swiglu(gate, up, local_weights, limit)
    else:
        gate_value = gate.float().clamp_max(limit) if limit else gate.float()
        up_value = up.float().clamp(-limit, limit) if limit else up.float()
        product = (F.silu(gate_value) * up_value * local_weights).to(gate.dtype)
    if moe.tensor_parallel:
        # Full K32 accumulation must precede each expert's BF16 rounding.
        product = _gather_channels(product)
    routed = grouped_fp4_gemm(
        product,
        local_count_tensor,
        prepared.down,
        counts_list=local_counts,
        pointers=None if pointers[2] is None else pointers[2][:2],
    )
    return local_tokens, routed
