from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41 import kernel_native, kernel_weights, model


def fp4_literal(rows, width=64):
    encoded = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
    encoded[:, 8] = -1
    encoded[:, 32] = 64
    encoded[:, 48] = 32
    encoded[:, 63] = 3
    encoded = encoded.to(torch.float8_e4m3fn)
    raw = torch.zeros((128, width // 2), dtype=torch.uint8, device="cuda")
    raw[:, 4], raw[:, 16], raw[:, 24], raw[:, 31] = 2, 6, 4, 32
    row_shifts = torch.arange(rows, device="cuda") % 3 - 1
    column_shifts = torch.arange(128, device="cuda") % 5 - 2
    input_scales = (
        (
            torch.tensor([113, 127] + [127] * (width // 32 - 2), device="cuda")
            + row_shifts[:, None]
        )
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    weight_scales = (
        (
            torch.tensor([126, 128] + [127] * (width // 32 - 2), device="cuda")
            + column_shifts[:, None]
        )
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    factors = torch.exp2((row_shifts[:, None] + column_shifts[None, :]).float())
    return (
        encoded,
        input_scales,
        raw.view(torch.float4_e2m1fn_x2),
        weight_scales,
        factors,
    )


def grouped_literal(counts=(1, 0, 33, 2), max_rows=33, dtype=torch.bfloat16, width=64):
    rows = sum(counts)
    encoded, scales, weight, weight_scales, factors = fp4_literal(rows, width)
    weights = [
        (
            weight.clone(),
            (weight_scales.view(torch.uint8) + shift).view(weight_scales.dtype),
        )
        for shift in (-1, 0, 1, 2)
    ]
    packed = [kernel_weights._packed_weight(w, s, 4) for w, s in weights]
    pointers = kernel_weights._grouped_pointer_arrays(packed, encoded.device)
    module = kernel_native._projection_module()
    return SimpleNamespace(
        encoded=encoded.view(torch.uint8),
        scales=scales.view(torch.uint8),
        pointers=pointers,
        counts=torch.tensor(counts, dtype=torch.int32, device="cuda"),
        output=torch.full((rows, 128), -7, dtype=dtype, device="cuda"),
        scratch=torch.empty(
            module.grouped_workspace_size(rows, 4, max_rows, 128, width),
            dtype=torch.uint8,
            device="cuda",
        ),
        factors=factors,
        max_rows=max_rows,
        weights=weights,
        packed=packed,
    )


def run_grouped(module, case, *, output=None):
    module.grouped_project(
        case.encoded,
        case.scales,
        *case.pointers,
        case.counts,
        case.output if output is None else output,
        case.scratch,
        case.max_rows,
    )


def prepared_moe_literal(device, strategy="TP4", experts=4):
    args = model.ModelArgs(
        dim=32,
        moe_inter_dim=64,
        n_routed_experts=experts,
        n_activated_experts=2,
        expert_dtype="fp4",
        parallel_strategy=strategy,
    )
    moe = model.MoE(0, args).to(device)

    class ZeroShared(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)

    moe.shared_experts = ZeroShared()
    with torch.no_grad():
        for expert in moe.experts:
            if expert is not None:
                for layer in (expert.w1, expert.w2, expert.w3):
                    raw = layer.weight.view(torch.uint8)
                    raw.zero_()
                    raw[0, 0] = 2
                    layer.scale.fill_(1)
    values = torch.zeros((1, 32), dtype=torch.bfloat16, device=device)
    values[:, 0] = 1
    routing = (torch.ones((1, 2), device=device), torch.tensor([[0, 1]], device=device))
    return moe, values, routing


def pack_weight_bytes(raw_weight, raw_scales, packed_weight, packed_scales, bits):
    if packed_weight.data_ptr() != raw_weight.data_ptr():
        packed_weight.zero_()
        packed_weight[:, : raw_weight.size(1)].copy_(raw_weight)
    packed_scales.zero_()
    packed_scales[: raw_scales.numel()].copy_(raw_scales.flatten())


class _RoutedDecodeModel(torch.nn.Module):
    def __init__(self, moe, values, routing):
        super().__init__()
        layer = torch.nn.Module()
        layer.ffn = moe
        layer.engram = None
        layer.attn = SimpleNamespace(compress_ratio=0, window_size=128)
        self.layers = torch.nn.ModuleList([layer])
        self.values, self.routing = values, routing

    def forward(self, _tokens, **_options):
        value = self.layers[0].ffn._forward_rows(self.values, routing=self.routing)
        return None, value, None
