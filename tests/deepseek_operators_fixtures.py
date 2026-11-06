import torch
import torch.nn.functional as F


def projection_weight(bits, outputs, width, *, offset=0, seed=43):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    columns = width * bits // 8
    storage = torch.empty(outputs * columns + offset, dtype=torch.uint8, device="cuda")
    raw = storage[offset:].view(outputs, columns)
    if bits == 4:
        raw.copy_(
            torch.randint(
                0, 256, raw.shape, generator=generator, device="cuda", dtype=torch.uint8
            )
        )
        weight = raw.view(torch.float4_e2m1fn_x2)
    else:
        values = torch.randn(raw.shape, generator=generator, device="cuda")
        raw.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
        weight = raw.view(torch.float8_e4m3fn)
    shape = (outputs if bits == 4 else outputs // 32, width // 32)
    scales = torch.randint(
        125, 129, shape, generator=generator, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    return weight, scales


def projection_reference(encoded, input_scales, weight, scales, bits):
    if bits == 4:
        raw = weight.view(torch.uint8)
        levels = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=weight.device
        )
        low, high = raw & 15, raw >> 4
        values = torch.stack(
            (
                levels[(low & 7).long()] * torch.where((low & 8).bool(), -1.0, 1.0),
                levels[(high & 7).long()] * torch.where((high & 8).bool(), -1.0, 1.0),
            ),
            dim=-1,
        ).flatten(-2)
        weight_scales = scales.float()
    else:
        values = weight.float()
        weight_scales = scales.float().repeat_interleave(32, dim=0)
    values = values * weight_scales.repeat_interleave(32, dim=-1)
    inputs = encoded.float() * input_scales.float().repeat_interleave(32, dim=-1)
    return F.linear(inputs, values).to(torch.bfloat16)


def rms_rounding_case(rows):
    values = torch.zeros((rows, 5120), dtype=torch.bfloat16, device="cuda")
    expected = torch.zeros_like(values)
    for column, value, normalized in (
        (4864, 2.609375, 0.0908203125),
        (3137, 20.75, 0.71875),
        (3789, 21.125, 0.734375),
        (1459, 21.0, 0.73046875),
        (4219, 50.25, 1.7421875),
        (3520, 3.640625, 0.1259765625),
        (3919, 0.291015625, 0.0101318359375),
        (2109, 0.019775390625, 0.0006866455078125),
    ):
        values[:, column] = value
        expected[:, column] = normalized
    weight = torch.full((5120,), 0.0301513671875, dtype=torch.bfloat16, device="cuda")
    return values, weight, expected
