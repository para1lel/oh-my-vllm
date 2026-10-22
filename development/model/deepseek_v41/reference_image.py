"""CPU image preparation from the local source, with FP64 normalized pixels.

Pillow performs the source resize/pad operation. No EXIF rotation, network fetch,
source BF16 conversion, or inference-runtime import occurs here.
"""

import hashlib
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL import __version__ as PILLOW_VERSION

MAX_ENCODED_BYTES = 64 << 20
MAX_DECODED_PIXELS = 64 << 20
MAX_PREPARED_BYTES = 512 << 20


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    patch_h: int
    patch_w: int
    aligned_h: int
    aligned_w: int

    @property
    def tokens(self):
        return self.aligned_h * (self.aligned_w + 1) + 2


@dataclass(frozen=True)
class PreparedImage:
    plan: Grid
    patches: np.ndarray
    token_types: np.ndarray
    input_sha256: str


def configuration(config):
    names = ("patch_size", "downsample_ratio", "max_image_tokens", "min_pixels")
    if any(type(config[name]) is not int or config[name] < 0 for name in names):
        raise ValueError("image sizes must be nonnegative integers")
    if (
        config["patch_size"] < 1
        or config["downsample_ratio"] < 1
        or config["max_image_tokens"] < 4
    ):
        raise ValueError("patch, aligner, and image-token capacities are too small")
    ratio = config["max_wh_ratio"]
    if ratio is not None and (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio <= 0
    ):
        raise ValueError("maximum width/height ratio must be positive and finite")


def _grid(height, width, patch, downsample):
    return Grid(
        height,
        width,
        height // patch,
        width // patch,
        math.ceil((height // patch) / downsample),
        math.ceil((width // patch) / downsample),
    )


def plan(width, height, config):
    configuration(config)
    if any(type(size) is not int or size < 1 for size in (width, height)):
        raise ValueError("image dimensions must be positive integers")
    patch, downsample = config["patch_size"], config["downsample_ratio"]
    maximum, minimum = config["max_image_tokens"], config["min_pixels"]
    max_ratio = config["max_wh_ratio"]
    if max_ratio is not None and width > height * max_ratio:
        width = height * max_ratio
    if width * height < minimum:
        scale = math.sqrt(minimum / (width * height))
        width, height = int(width * scale), int(height * scale)
    result = _grid(
        math.ceil(height / patch) * patch,
        math.ceil(width / patch) * patch,
        patch,
        downsample,
    )
    if result.tokens <= maximum:
        return result
    ratio = height / width
    max_width = math.sqrt((maximum - 2) / ratio + 0.25) - 0.5
    max_height = max_width * ratio
    cell = patch * downsample
    if max_width < 1:
        best_height, best_width = (maximum - 2) // 2 * cell, cell
    elif max_height < 1:
        best_height, best_width = cell, (maximum - 3) * cell
    else:
        scale = min(
            math.floor(max_width) * cell / width,
            math.floor(max_height) * cell / height,
        )
        best_height = math.floor(height * scale / patch) * patch
        best_width = math.floor(width * scale / patch) * patch
    result = _grid(best_height, best_width, patch, downsample)
    if min(result.patch_h, result.patch_w) < 1 or result.tokens > maximum:
        raise ValueError("source image grid does not fit its token bound")
    return result


def prepare(raw, config):
    if PILLOW_VERSION != "12.3.0":
        raise ValueError("image reference requires pinned Pillow 12.3.0")
    if not isinstance(raw, bytes):
        raise TypeError("image reference input must be encoded bytes, not a path")
    if not raw or len(raw) > MAX_ENCODED_BYTES:
        raise ValueError("encoded image is empty or exceeds 64 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > MAX_DECODED_PIXELS:
                raise ValueError("decoded image exceeds the pixel limit")
            grid = plan(source.width, source.height, config)
            if grid.height * grid.width * 3 * 8 > MAX_PREPARED_BYTES:
                raise ValueError("prepared FP64 pixels exceed their byte bound")
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid encoded image") from error
    max_ratio = config["max_wh_ratio"]
    if max_ratio is not None and image.width >= max_ratio * image.height:
        image = image.resize((grid.width, grid.height))
    else:
        image = ImageOps.pad(image, (grid.width, grid.height), color=(127, 127, 127))
    pixels = (np.asarray(image, dtype=np.float64).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    patch = config["patch_size"]
    patches = (
        pixels.reshape(3, grid.patch_h, patch, grid.patch_w, patch)
        .transpose(1, 3, 0, 2, 4)
        .reshape(grid.patch_h * grid.patch_w, 3, patch, patch)
        .copy()
    )
    types = [0] + ([1] * grid.aligned_w + [2]) * grid.aligned_h + [3]
    return PreparedImage(
        grid,
        patches,
        np.asarray(types, dtype=np.int64),
        hashlib.sha256(raw).hexdigest(),
    )
