"""CPU FP64 vision equations from LOCAL-V41 inference/vision.py.

Packing and learned spans follow inference/image_processor.py and model.py.
Inputs are normalized pixels or source-order patches. This reference omits
preprocessing, activation quantization, and source BF16/FP32 rounding.
"""

import math

import torch
import torch.nn.functional as functional

from .reference_math import require_fp64, rms
from .reference_model import TorchWeights


def _cpu(*values):
    require_fp64(*values)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("vision reference requires CPU FP64 tensors")


def _grid(n_h, n_w):
    if any(type(size) is not int or size < 1 for size in (n_h, n_w)):
        raise ValueError("vision grid dimensions must be positive integers")


def _tile(query_tile):
    if type(query_tile) is not int or not 1 <= query_tile <= 256:
        raise ValueError("vision query tile must be between one and 256")


def _cpu_weights(weights):
    if weights.device.type != "cpu":
        raise ValueError("vision reference requires CPU weights")


def _tensor(weights, name):
    _cpu_weights(weights)
    value = weights.tensor(name)
    _cpu(value)
    return value


def _linear(weights, value, prefix, *, bias=True):
    _cpu_weights(weights)
    _cpu(value)
    output = weights.linear(value, prefix)
    _cpu(output)
    if bias:
        output += _tensor(weights, prefix + ".bias")
    return output


@torch.inference_mode()
def pack_patches(pixels, n_h, n_w, *, patch_size=14):
    """Pack normalized [3,H,W] pixels as [n_h*n_w,3,p,p], without rounding."""
    _cpu(pixels)
    _grid(n_h, n_w)
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("vision patch size must be a positive integer")
    p = patch_size
    if pixels.shape != (3, n_h * p, n_w * p):
        raise ValueError("normalized pixels differ from the explicit patch grid")
    return (
        pixels.reshape(3, n_h, p, n_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_h * n_w, 3, p, p)
    )


@torch.inference_mode()
def rotary_2d(value, n_h, n_w, *, theta=10000):
    """Rotate split channel halves with height frequencies before width."""
    _cpu(value)
    _grid(n_h, n_w)
    if (
        value.ndim != 3
        or value.shape[0] != n_h * n_w
        or value.shape[-1] == 0
        or value.shape[-1] % 4
        or not math.isfinite(theta)
        or theta <= 1
    ):
        raise ValueError("invalid vision rotary shape or frequency base")
    half = value.shape[-1] // 2
    frequency = theta ** (
        -torch.arange(0, half, 2, dtype=torch.float64, device="cpu") / half
    )
    position = torch.arange(n_h * n_w, device="cpu")
    angle = torch.cat(
        (
            (position // n_w)[:, None] * frequency,
            (position % n_w)[:, None] * frequency,
        ),
        dim=-1,
    )[:, None, :]
    real, imaginary = value.chunk(2, dim=-1)
    cosine, sine = angle.cos(), angle.sin()
    return torch.cat(
        (real * cosine - imaginary * sine, imaginary * cosine + real * sine),
        dim=-1,
    )


@torch.inference_mode()
def attention(weights, prefix, value, n_h, n_w, *, heads, theta=10000, query_tile=32):
    """Tile queries over all image keys, with no mask or attention sink."""
    _cpu(value)
    _grid(n_h, n_w)
    _tile(query_tile)
    if (
        value.ndim != 2
        or value.shape[0] != n_h * n_w
        or type(heads) is not int
        or heads < 1
        or value.shape[-1] % heads
    ):
        raise ValueError("invalid vision attention grid or head count")
    count, width = value.shape
    head_width = width // heads
    q, k, v = (
        part.reshape(count, heads, head_width)
        for part in _linear(weights, value, prefix + ".wqkv").chunk(3, dim=-1)
    )
    q = rotary_2d(q, n_h, n_w, theta=theta).transpose(0, 1)
    k = rotary_2d(k, n_h, n_w, theta=theta).permute(1, 2, 0)
    v = v.transpose(0, 1)
    output = torch.empty_like(value)
    for begin in range(0, count, query_tile):
        end = min(begin + query_tile, count)
        scores = (q[:, begin:end] @ k) / math.sqrt(head_width)
        probabilities = scores.softmax(dim=-1)
        output[begin:end] = (probabilities @ v).transpose(0, 1).reshape(-1, width)
        del scores, probabilities
    return _linear(weights, output, prefix + ".wo")


@torch.inference_mode()
def align(weights, value, n_h, n_w, *, ratio=3):
    """Pad right and bottom; gather cells in the source F.unfold order."""
    _cpu(value)
    _grid(n_h, n_w)
    if (
        value.ndim != 2
        or value.shape[0] != n_h * n_w
        or type(ratio) is not int
        or ratio < 1
    ):
        raise ValueError("invalid vision aligner grid or downsample ratio")
    height, width = (n_h + ratio - 1) // ratio, (n_w + ratio - 1) // ratio
    channels = value.shape[-1]
    padded = value.new_zeros((height * ratio, width * ratio, channels))
    padded[:n_h, :n_w] = value.reshape(n_h, n_w, channels)
    cells = (
        padded.reshape(height, ratio, width, ratio, channels)
        .permute(0, 2, 4, 1, 3)
        .reshape(height * width, channels * ratio**2)
    )
    projected = _linear(weights, cells, "aligner.w1")
    return _linear(
        weights, functional.gelu(projected, approximate="none"), "aligner.w2"
    )


@torch.inference_mode()
def image_span(weights, features, n_h, n_w):
    """Emit start, each aligned row plus newline, then end."""
    _cpu(features)
    _grid(n_h, n_w)
    if features.ndim != 2 or features.shape[0] != n_h * n_w:
        raise ValueError("aligned features differ from the explicit image grid")
    start, newline, end = (
        _tensor(weights, name) for name in ("image_start", "image_newline", "image_end")
    )
    if any(token.shape != (features.shape[-1],) for token in (start, newline, end)):
        raise ValueError("learned image token width differs from aligned features")
    rows = torch.cat(
        (features.reshape(n_h, n_w, -1), newline.expand(n_h, 1, -1)), dim=1
    )
    return torch.cat((start[None], rows.flatten(0, 1), end[None]), dim=0)


class Vision:
    """Stream TorchWeights projections through all configured vision layers.

    The checkpoint configuration has 32 layers, width 1024, 16 heads, MLP
    width 2816, patch size 14, ratio 3, and output width 5120.
    Smaller explicit configurations permit independent synthetic checks.
    """

    def __init__(self, weights: TorchWeights, config, *, query_tile=32):
        _cpu_weights(weights)
        _tile(query_tile)
        vision = config["vision_config"]
        self.weights = weights
        self.query_tile = query_tile
        self.width = vision["hidden_size"]
        self.heads = vision["num_attention_heads"]
        self.intermediate = vision["intermediate_size"]
        self.layers = vision["num_hidden_layers"]
        self.patch_size = vision["patch_size"]
        self.ratio = vision["downsample_ratio"]
        self.max_tokens = vision["max_image_tokens"]
        self.theta = vision["rope_theta"]
        self.output_width = config["text_config"]["hidden_size"]
        sizes = (
            self.width,
            self.heads,
            self.intermediate,
            self.layers,
            self.patch_size,
            self.ratio,
            self.max_tokens,
            self.output_width,
        )
        if (
            any(type(size) is not int or size < 1 for size in sizes)
            or self.width % self.heads
            or (self.width // self.heads) % 4
            or not math.isfinite(self.theta)
            or self.theta <= 1
        ):
            raise ValueError("invalid vision reference configuration")
        self._check_layout()

    def _shape(self, name, *shape):
        record = self.weights.source.layout.get(name)
        if record is None or tuple(record["shape"]) != shape:
            raise ValueError(f"vision checkpoint shape differs: {name}")

    def _projection_shape(self, prefix, rows, columns, *, bias=True):
        self._shape(prefix + ".weight", rows, columns)
        if bias:
            self._shape(prefix + ".bias", rows)

    def _check_layout(self):
        width = self.width
        self._projection_shape("vision.patch_embed.proj", width, 3 * self.patch_size**2)
        for layer in range(self.layers):
            prefix = f"vision.blocks.{layer}"
            for norm in ("norm1", "norm2"):
                self._shape(prefix + f".{norm}.weight", width)
            self._projection_shape(prefix + ".attn.wqkv", 3 * width, width)
            self._projection_shape(prefix + ".attn.wo", width, width)
            self._projection_shape(
                prefix + ".mlp.w1", 2 * self.intermediate, width, bias=False
            )
            self._projection_shape(
                prefix + ".mlp.w2", width, self.intermediate, bias=False
            )
        self._shape("vision.norm.weight", width)
        self._projection_shape("aligner.w1", self.output_width, width * self.ratio**2)
        self._projection_shape("aligner.w2", self.output_width, self.output_width)
        for name in ("image_start", "image_end", "image_newline"):
            self._shape(name, self.output_width)

    def _check_grid(self, n_h, n_w):
        _grid(n_h, n_w)
        height = (n_h + self.ratio - 1) // self.ratio
        width = (n_w + self.ratio - 1) // self.ratio
        if height * (width + 1) + 2 > self.max_tokens:
            raise ValueError("full image span exceeds the configured token limit")
        return height, width

    @torch.inference_mode()
    def encode(self, patches, n_h, n_w):
        """Return aligned image features before learned token insertion."""
        self._check_grid(n_h, n_w)
        _cpu(patches)
        if patches.shape != (n_h * n_w, 3, self.patch_size, self.patch_size):
            raise ValueError("source patches must have shape [n_h*n_w,3,p,p]")
        weights = self.weights
        value = _linear(weights, patches.flatten(1), "vision.patch_embed.proj")
        for layer in range(self.layers):
            prefix = f"vision.blocks.{layer}"
            normalized = rms(
                value, _tensor(weights, prefix + ".norm1.weight"), eps=1e-6
            )
            value = value + attention(
                weights,
                prefix + ".attn",
                normalized,
                n_h,
                n_w,
                heads=self.heads,
                theta=self.theta,
                query_tile=self.query_tile,
            )
            normalized = rms(
                value, _tensor(weights, prefix + ".norm2.weight"), eps=1e-6
            )
            gate, up = _linear(
                weights, normalized, prefix + ".mlp.w1", bias=False
            ).chunk(2, dim=-1)
            value = value + _linear(
                weights, functional.silu(gate) * up, prefix + ".mlp.w2", bias=False
            )
        value = rms(value, _tensor(weights, "vision.norm.weight"), eps=1e-6)
        return align(weights, value, n_h, n_w, ratio=self.ratio)

    @torch.inference_mode()
    def forward(self, patches, n_h, n_w):
        """Return the complete source-order image span in CPU FP64."""
        height, width = self._check_grid(n_h, n_w)
        features = self.encode(patches, n_h, n_w)
        return image_span(self.weights, features, height, width)

    @torch.inference_mode()
    def forward_pixels(self, pixels, n_h, n_w):
        """Accept normalized CPU FP64 [3,H,W] pixels with no preprocessing."""
        self._check_grid(n_h, n_w)
        patches = pack_patches(pixels, n_h, n_w, patch_size=self.patch_size)
        return self.forward(patches, n_h, n_w)
