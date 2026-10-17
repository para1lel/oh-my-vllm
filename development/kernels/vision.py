import math

import numpy as np
from oh_my_vllm.kernels.vision import FUNCTIONS, VisionBlock, Weights

from .build import build
from .reference import bf16, bf16_bits, gelu, layer_norm, linear, numerical_error

BLOCK_NAMES = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "norm2.weight",
    "norm2.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias",
)
MERGER_NAMES = (
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
)


def download_bf16(device, value):
    return (device.download(value, np.uint16).astype(np.uint32) << 16).view(np.float32)


def spatial_positions(height, width):
    y, x = np.indices((height, width))
    return tuple(
        value.reshape(height // 2, 2, width // 2, 2).transpose(0, 2, 1, 3).reshape(-1)
        for value in (y, x)
    )


def resize_axis(source, target):
    scale = source / target
    width = max(scale, 1)
    centers = (np.arange(target, dtype=np.float64) + 0.5) * scale - 0.5
    distance = np.abs((np.arange(source)[None, :] - centers[:, None]) / width)
    weights = np.where(
        distance < 1,
        1.5 * distance**3 - 2.5 * distance**2 + 1,
        np.where(
            distance < 2, -0.5 * distance**3 + 2.5 * distance**2 - 4 * distance + 2, 0
        ),
    )
    return weights / weights.sum(axis=-1, keepdims=True)


def resize_image(value, height, width):
    if value.shape[:2] == (height, width):
        return value
    vertical = resize_axis(value.shape[0], height)
    horizontal = resize_axis(value.shape[1], width)
    resized = np.einsum(
        "yh,hwc,xw->yxc", vertical, value.astype(np.float64), horizontal, optimize=True
    )
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def reference_patches(images, grid):
    height, width = grid
    frames = [resize_image(image, height * 16, width * 16) for image in images]
    if len(frames) % 2:
        frames.append(frames[-1])
    normalized = bf16(np.asarray(frames, dtype=np.float64) * (2 / 255) - 1)
    patches = normalized.reshape(
        len(frames) // 2, 2, height // 2, 2, 16, width // 2, 2, 16, 3
    )
    return patches.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7).reshape(-1, 1536)


def reference_positions(position, grid):
    height, width = grid
    py, px = spatial_positions(height, width)
    y, x = py * (47 / (height - 1)), px * (47 / (width - 1))
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, 47), np.minimum(x0 + 1, 47)
    dy, dx = y - y0, x - x0
    position = np.asarray(position, dtype=np.float64)
    return bf16(
        position[y0 * 48 + x0] * ((1 - dy) * (1 - dx))[:, None]
        + position[y0 * 48 + x1] * ((1 - dy) * dx)[:, None]
        + position[y1 * 48 + x0] * (dy * (1 - dx))[:, None]
        + position[y1 * 48 + x1] * (dy * dx)[:, None]
    )


def reference_rope(value, grid):
    py, px = spatial_positions(*grid)
    frequency = 10000 ** (-np.arange(18, dtype=np.float64) / 18)
    angle = np.concatenate((py[:, None] * frequency, px[:, None] * frequency), axis=-1)
    angle = np.concatenate((angle, angle), axis=-1)[:, None, :]
    rotated = np.concatenate((-value[..., 36:], value[..., :36]), axis=-1)
    return bf16(value * bf16(np.cos(angle)) + rotated * bf16(np.sin(angle)))


def reference_block(value, weights, indices, grid):
    n1w, n1b, qkvw, qkvb, pw, pb, n2w, n2b, uw, ub, dw, db = weights
    normalized = layer_norm(value, n1w, n1b)
    qkv = linear(normalized, qkvw, qkvb).reshape(len(value), 3, 16, 72)
    query, key, values = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    query, key = reference_rope(query, grid), reference_rope(key, grid)
    scores = np.einsum("qhd,khd->hqk", query[indices], key) / np.sqrt(72)
    scores -= scores.max(axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability = np.asarray(
        probability / probability.sum(axis=-1, keepdims=True), dtype=np.float32
    )
    attended = bf16(np.einsum("hqk,khd->qhd", probability, values)).reshape(
        len(indices), 1152
    )
    residual = bf16(value[indices] + linear(attended, pw, pb))
    up = linear(layer_norm(residual, n2w, n2b), uw, ub)
    return bf16(residual + linear(bf16(gelu(up, approximate=True)), dw, db))


class VisionWeights:
    def __init__(self, device, checkpoint):
        self.blocks = tuple(
            tuple(
                checkpoint.tensor(f"model.visual.blocks.{layer}.{name}")
                for name in BLOCK_NAMES
            )
            for layer in range(27)
        )
        self.patch = tuple(
            checkpoint.tensor("model.visual." + name)
            for name in (
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
                "pos_embed.weight",
            )
        )
        self.patch = (self.patch[0].reshape(1152, 1536), *self.patch[1:])
        self.merger = tuple(
            checkpoint.tensor("model.visual.merger." + name) for name in MERGER_NAMES
        )

        def upload(values):
            return tuple(
                device.upload(bf16_bits(value), "bfloat16") for value in values
            )

        self.device_weights = Weights(
            tuple(upload(block) for block in self.blocks),
            upload(self.patch),
            upload(self.merger),
        )
        self.patch_boundary_checks = None


class VisionCase:
    def __init__(self, device, weights, shape):
        build("vision", FUNCTIONS)
        self.device, self.weights, self.shape = device, weights, shape
        side = math.isqrt(shape.frame_tokens)
        if side * side != shape.frame_tokens:
            raise ValueError("modeled vision cases require a square spatial grid")
        self.grid = (side, side)
        groups = shape.rows // (shape.batch * shape.frame_tokens)
        self.frames = 1 if groups == 1 else groups * 2
        rng = np.random.default_rng(57 + shape.rows)
        if shape.variant == "input":
            self.input = rng.integers(
                0,
                256,
                (shape.batch, self.frames, side * 16, side * 16, 3),
                dtype=np.uint8,
            )
            self.value = device.upload(self.input)
        else:
            self.input = bf16(rng.normal(0, 0.3, (shape.rows, 1152)))
            self.value = device.upload(bf16_bits(self.input), "bfloat16")
        self.operation = VisionBlock(
            weights.device_weights,
            shape.rows,
            shape.frame_tokens,
            device.empty,
            variant=shape.variant,
            grid=self.grid,
            frames=self.frames,
        )
        self.reference_cache = {}

    def close(self):
        self.operation.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def reset(self):
        pass

    def run(self):
        return self.operation(self.value)

    def frame_input(self, group):
        tokens = self.shape.frame_tokens
        if self.shape.variant != "input":
            return self.input[group * tokens : (group + 1) * tokens]
        temporal_groups = (self.frames + 1) // 2
        item, temporal = divmod(group, temporal_groups)
        frames = self.input[item, temporal * 2 : min(temporal * 2 + 2, self.frames)]
        patches = reference_patches(frames, self.grid)
        weight, bias, position = self.weights.patch
        return bf16(
            linear(patches, weight, bias) + reference_positions(position, self.grid)
        )

    def expected_rows(self, indices):
        tokens = self.shape.frame_tokens
        output = np.empty((len(indices), 1152), dtype=np.float64)
        for group in np.unique(indices // tokens):
            selected = np.flatnonzero(indices // tokens == group)
            local = indices[selected] % tokens
            key = (int(group), tuple(local.tolist()))
            if key not in self.reference_cache:
                self.reference_cache[key] = reference_block(
                    self.frame_input(int(group)),
                    self.weights.blocks[self.operation.layer],
                    local,
                    self.grid,
                )
            output[selected] = self.reference_cache[key]
        return output

    def output_reference(self, indices):
        if self.shape.variant != "merge":
            return self.expected_rows(indices)
        rows = (indices[:, None] * 4 + np.arange(4)).reshape(-1)
        nweight, nbias, upweight, upbias, downweight, downbias = self.weights.merger
        normalized = layer_norm(self.expected_rows(rows), nweight, nbias)
        up = linear(normalized.reshape(len(indices), 4608), upweight, upbias)
        return linear(bf16(gelu(up)), downweight, downbias)

    def check_patch_order(self):
        if self.shape.variant != "input":
            return {
                "passed": True,
                "scope": "block-major spatial RoPE and merger rows in FP64 reference",
            }
        actual = download_bf16(self.device, self.operation.patches)
        tokens = self.shape.frame_tokens
        indices = np.unique([0, 1, 2, 3, tokens // 2, tokens - 1])
        expected = reference_patches(self.input[0, : min(self.frames, 2)], self.grid)
        result = numerical_error(actual[indices], expected[indices])
        result["passed"] = bool(np.array_equal(actual[indices], expected[indices]))
        result["rows_checked"] = indices.tolist()
        return result

    def check_frame_isolation(self, actual):
        groups = self.shape.rows // self.shape.frame_tokens
        if groups == 1:
            return {
                "passed": True,
                "scope": "one frame group; all attention keys in FP64 reference",
            }
        changed = self.input.copy()
        if self.shape.variant == "input":
            changed[0, : min(self.frames, 2)] = 255 - changed[0, : min(self.frames, 2)]
            value = self.device.upload(changed)
        else:
            changed[: self.shape.frame_tokens] *= -1
            value = self.device.upload(bf16_bits(changed), "bfloat16")
        try:
            perturbed = download_bf16(self.device, self.operation(value))
            boundary = self.shape.frame_tokens // (
                4 if self.shape.variant == "merge" else 1
            )
            isolated = bool(np.array_equal(actual[boundary:], perturbed[boundary:]))
            changed_group = bool(np.any(actual[:boundary] != perturbed[:boundary]))
            return {
                "passed": isolated and changed_group,
                "other_groups_unchanged": isolated,
                "first_group_changed": changed_group,
            }
        finally:
            self.run()

    def check_patch_input(self, images):
        expected = reference_patches(images[0], self.grid)
        rows = len(expected)
        patches = self.device.empty((rows, 1536), "bfloat16")
        projected = self.device.empty((rows * 4304,), "float32")
        output = self.device.empty((rows, 1152), "bfloat16")
        self.operation.module.vision_patch(
            self.device.upload(images),
            *self.weights.device_weights.patch,
            patches,
            projected,
            output,
            *self.grid,
        )
        tokens = self.shape.frame_tokens
        indices = np.unique([0, 1, 3, tokens - 1, rows - 1])
        patch_check = numerical_error(
            download_bf16(self.device, patches)[indices], expected[indices]
        )
        weight, bias, position = self.weights.patch
        features = bf16(
            linear(expected[indices], weight, bias)
            + reference_positions(position, self.grid)[indices % tokens]
        )
        actual = download_bf16(self.device, output)
        feature_check = numerical_error(actual[indices], features)
        finite = bool(np.all(np.isfinite(actual)))
        return {
            "passed": patch_check["passed"] and feature_check["passed"] and finite,
            "source_shape": list(images.shape),
            "grid": list(self.grid),
            "rows_checked": indices.tolist(),
            "patches": patch_check,
            "projected_features": feature_check,
            "all_rows_finite": finite,
        }

    def check_patch_boundaries(self):
        if self.shape.variant != "input":
            return {"passed": True, "scope": "patch input is outside this variant"}
        if self.weights.patch_boundary_checks is None:
            image = self.input[0, 0]
            resized = image[None, None, :-1, :-7].copy()
            odd_frames = np.stack((image, image[:, ::-1], 255 - image))[None]
            resize = self.check_patch_input(resized)
            temporal = self.check_patch_input(odd_frames)
            self.weights.patch_boundary_checks = {
                "passed": resize["passed"] and temporal["passed"],
                "nonidentity_bicubic_resize": resize,
                "three_frames_repeat_last": temporal,
            }
        return self.weights.patch_boundary_checks

    def check(self, *, execute=True):
        actual = download_bf16(
            self.device, self.run() if execute else self.operation.output
        )
        finite = bool(np.all(np.isfinite(actual)))
        tokens = self.shape.frame_tokens // (4 if self.shape.variant == "merge" else 1)
        indices = np.unique([0, 1, 2, 3, tokens - 1, len(actual) // 2, len(actual) - 1])
        output = numerical_error(
            actual[indices], self.output_reference(indices), tolerance="vision"
        )
        patch_order = self.check_patch_order()
        patch_boundaries = self.check_patch_boundaries()
        isolation = self.check_frame_isolation(actual)
        return {
            "passed": finite
            and output["passed"]
            and patch_order["passed"]
            and patch_boundaries["passed"]
            and isolation["passed"],
            "all_rows_finite": finite,
            "output_tolerance_scope": (
                "sampled rows, all columns, complete frame-local keys and values"
            ),
            "rows_checked": indices.tolist(),
            "output": output,
            "patch_order": patch_order,
            "patch_boundaries": patch_boundaries,
            "frame_isolation": isolation,
            "merger_norm": "each 1152-element patch before the 2-by-2 gather"
            if self.shape.variant == "merge"
            else "not applicable",
        }
