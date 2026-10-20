import torch
import torch.nn.functional as functional

from .common import linear


def order(height, width, device):
    return (
        torch.arange(height * width, device=device)
        .reshape(height // 2, 2, width // 2, 2)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )


def patch(pixels, shape, weights):
    _, frames, height, width, _ = pixels.shape
    if frames % 2:
        pixels = torch.cat((pixels, pixels[:, -1:]), dim=1)
    normalized = pixels.to(torch.float64) * (2 / 255) - 1
    gh, gw = shape["height"], shape["width"]
    if (height, width) != (gh * 16, gw * 16):
        raise ValueError("fixed vision fixtures must use the declared source grid")
    indices = order(gh, gw, pixels.device)
    patches = (
        normalized.reshape(1, (frames + 1) // 2, 2, gh, 16, gw, 16, 3)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(-1, gh * gw, 1536)
    )
    patches = patches[:, indices].reshape(-1, 1536)
    projected = (
        patches
        @ weights.tensor("model.visual.patch_embed.proj.weight").reshape(1152, 1536).T
    )
    projected += weights.tensor("model.visual.patch_embed.proj.bias")
    position = (
        weights.tensor("model.visual.pos_embed.weight")
        .reshape(48, 48, 1152)
        .permute(2, 0, 1)[None]
    )
    position = functional.interpolate(
        position, size=(gh, gw), mode="bilinear", align_corners=True
    )
    position = position[0].permute(1, 2, 0).reshape(-1, 1152)[indices]
    return projected + position.repeat((frames + 1) // 2, 1)


def norm(value, weights, prefix):
    return functional.layer_norm(
        value,
        (1152,),
        weights.tensor(prefix + ".weight"),
        weights.tensor(prefix + ".bias"),
        eps=1e-6,
    )


def rotary(value, height, width):
    indices = order(height, width, value.device)
    coordinates = torch.stack((indices // width, indices % width), dim=-1)
    frequency = 10000.0 ** (
        -torch.arange(18, device=value.device, dtype=torch.float64) / 18
    )
    angles = (coordinates[:, :, None] * frequency).reshape(height * width, 36)
    angles = angles.repeat(value.shape[0] // len(angles), 1)[:, None, :]
    first, second = value.split(36, dim=-1)
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        dim=-1,
    )


def block(value, shape, weights, layer):
    prefix = f"model.visual.blocks.{layer}."
    normalized = norm(value, weights, prefix + "norm1")
    query, key, values = (
        linear(normalized, weights, prefix + "attn.qkv", bias=True)
        .reshape(-1, 3, 16, 72)
        .unbind(dim=1)
    )
    query, key = (
        rotary(item, shape["height"], shape["width"]) for item in (query, key)
    )
    tokens = shape["height"] * shape["width"]
    query, key, values = (
        item.reshape(-1, tokens, 16, 72).transpose(1, 2)
        for item in (query, key, values)
    )
    scores = query @ key.transpose(-1, -2) / 72**0.5
    attended = (
        (torch.softmax(scores, dim=-1) @ values).transpose(1, 2).reshape(-1, 1152)
    )
    value = value + linear(attended, weights, prefix + "attn.proj", bias=True)
    normalized = norm(value, weights, prefix + "norm2")
    activated = functional.gelu(
        linear(normalized, weights, prefix + "mlp.linear_fc1", bias=True),
        approximate="tanh",
    )
    return value + linear(activated, weights, prefix + "mlp.linear_fc2", bias=True)


def reference(case, inputs, state, weights):
    value = (
        patch(inputs["pixels"], case.shape, weights)
        if case.boundary == "vision.input"
        else inputs["hidden"]
    )
    value = block(value, case.shape, weights, case.weights[0])
    if case.boundary == "vision.merge":
        value = norm(value, weights, "model.visual.merger.norm").reshape(-1, 4608)
        value = functional.gelu(
            linear(value, weights, "model.visual.merger.linear_fc1", bias=True),
            approximate="none",
        )
        value = linear(value, weights, "model.visual.merger.linear_fc2", bias=True)
    return {"hidden": value}
