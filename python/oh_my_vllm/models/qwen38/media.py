import base64
import copy
import io
import math
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass, field

import av
import numpy as np
import tvm_ffi
from PIL import Image, ImageOps

from ...operators.vision import api as vision
from ...operators.vision.plan import make_plan
from ...operators.vision.workspace import prepare


@dataclass
class Media:
    pixels: object
    grid: tuple
    timestamps: list
    spans: list = field(default_factory=list)
    embedding: object = None


def read_media(url):
    limit = 64 * 1024 * 1024
    if url.startswith("data:"):
        raw = url.split(",", 1)[1]
        if len(raw) > (limit + 2) // 3 * 4:
            raise ValueError("media exceeds 64 MiB")
        result = base64.b64decode(raw, validate=True)
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            result = response.read(limit + 1)
    if len(result) > limit:
        raise ValueError("media exceeds 64 MiB")
    return result


def resize_grid(height, width, minimum, maximum):
    if (
        min(height, width) <= 0
        or max(height, width) > 65536
        or max(height, width) / min(height, width) > 200
    ):
        raise ValueError("invalid media aspect ratio")
    resized = [max(32, round(value / 32) * 32) for value in (height, width)]
    if math.prod(resized) > maximum:
        scale = math.sqrt(height * width / maximum)
        resized = [
            max(32, math.floor(value / scale / 32) * 32) for value in (height, width)
        ]
    elif math.prod(resized) < minimum:
        scale = math.sqrt(minimum / (height * width))
        resized = [
            max(32, math.ceil(value * scale / 32) * 32) for value in (height, width)
        ]
    return tuple(value // 16 for value in resized)


def image_media(raw, detail):
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 64 * 1024 * 1024:
            raise ValueError("image exceeds the decoded pixel limit")
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = np.asarray(image).copy()[None]
    grid = resize_grid(
        *pixels.shape[1:3], 65536, 65536 if detail == "low" else 16777216
    )
    return Media(pixels, grid, [])


def video_media(raw):
    with av.open(io.BytesIO(raw)) as container:
        if not container.streams.video:
            raise ValueError("video has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        total = stream.frames
        if not total:
            total = sum(1 for _ in container.decode(stream))
    if not total or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video has no valid frame rate or frames")
    count = min(total, max(4, min(768, int(total / fps * 2))))
    selected = np.linspace(0, total - 1, count).round().astype(int).tolist()
    needed = set(selected)
    frames = {}
    size = 0
    with av.open(io.BytesIO(raw)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in needed:
                pixels = frame.to_ndarray(format="rgb24")
                size += pixels.nbytes
                if size > 512 * 1024 * 1024:
                    raise ValueError("sampled video exceeds 512 MiB")
                frames[index] = pixels
            if index >= selected[-1]:
                break
    if set(frames) != needed:
        raise ValueError("video ended before its declared frame count")
    pixels = np.stack([frames[index] for index in selected])
    grid = resize_grid(
        *pixels.shape[1:3], 4096, max(4096, 25165824 // ((count + 1) // 2 * 2))
    )
    padded = selected + ([selected[-1]] if count % 2 else [])
    timestamps = [
        (padded[index] + padded[index + 1]) / (2 * fps)
        for index in range(0, len(padded), 2)
    ]
    return Media(pixels, grid, timestamps)


def rope_positions(length, media):
    positions = np.empty((3, length), np.int64)
    cursor, base = 0, 0
    for item in media:
        height, width = (value // 2 for value in item.grid)
        for first, count, _offset in item.spans:
            text = first - cursor
            positions[:, cursor:first] = np.arange(text)[None] + base
            base += text
            positions[:, first : first + count] = (
                np.indices((1, height, width)).reshape(3, -1) + base
            )
            base += max(height, width)
            cursor = first + count
    positions[:, cursor:] = np.arange(length - cursor)[None] + base
    return positions, int(positions.max()) + 1 - length


class MediaProcessor:
    def __init__(self, device, weights, tokenizer):
        self.device, self.weights, self.tokenizer = device, weights, tokenizer
        self.markers = {
            name: tokenizer.tokenizer.token_to_id(f"<|{name}|>")
            for name in ("image_pad", "video_pad", "vision_start", "vision_end")
        }

    def prepare(self, options):
        options = copy.deepcopy(options)
        items = []
        for message in options["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                kind = part["type"]
                if kind == "image_url":
                    reference = part["image_url"]
                    items.append(
                        image_media(
                            read_media(reference["url"]),
                            reference.get("detail", "auto"),
                        )
                    )
                    part["type"] = "image"
                elif kind == "video_url":
                    items.append(video_media(read_media(part["video_url"]["url"])))
                    part["type"] = "video"
        tokens = self.tokenizer.prepare(options)
        if not items:
            return tokens, items, None, 0
        expanded, index, media_index = [], 0, 0
        while index < len(tokens):
            if (
                tokens[index] != self.markers["vision_start"]
                or index + 2 >= len(tokens)
                or tokens[index + 1]
                not in (self.markers["image_pad"], self.markers["video_pad"])
            ):
                expanded.append(tokens[index])
                index += 1
                continue
            if (
                media_index >= len(items)
                or tokens[index + 2] != self.markers["vision_end"]
            ):
                raise ValueError("media markers do not match input media")
            item = items[media_index]
            video = tokens[index + 1] == self.markers["video_pad"]
            if video != bool(item.timestamps):
                raise ValueError("media marker order differs from input media")
            count = math.prod(item.grid) // 4
            for frame, timestamp in enumerate(item.timestamps if video else [None]):
                if video:
                    expanded.extend(
                        self.tokenizer.tokenizer.encode(
                            f"<{timestamp:.1f} seconds>", add_special_tokens=False
                        ).ids
                    )
                expanded.append(self.markers["vision_start"])
                item.spans.append((len(expanded), count, frame * count))
                expanded.extend([tokens[index + 1]] * count)
                expanded.append(self.markers["vision_end"])
            media_index += 1
            index += 3
        if media_index != len(items):
            raise ValueError("chat template omitted input media")
        positions, delta = rope_positions(len(expanded), items)
        return expanded, items, positions, delta

    def encode(self, item):
        if isinstance(item.embedding, np.ndarray):
            source = np.require(item.embedding, requirements=["C", "W"])
            value = self.device.scratch(source.shape, "bfloat16")
            self.device.module.copy_bytes(value, tvm_ffi.from_dlpack(source))
            self.device.synchronize_compute()
            item.embedding = value
        if item.embedding is not None:
            return item.embedding
        d, weights = self.device, self.weights.vision()
        frames, height, width, _ = item.pixels.shape
        tokens = math.prod(item.grid)
        groups = (frames + 1) // 2
        result = d.scratch((groups * tokens // 4, 5120), "bfloat16")
        group_limit = max(1, 65536 // tokens)
        for first in range(0, groups, group_limit):
            count = min(group_limit, groups - first)
            pixels = d.upload(
                item.pixels[None, first * 2 : min((first + count) * 2, frames)]
            )
            rows = count * tokens
            with ExitStack() as resources:
                bindings = tuple(
                    vision.layer_weights(weights, layer) for layer in range(27)
                )
                operations = []
                for variant in ("input", "default", "default", "merge"):
                    plan = make_plan(
                        rows=rows,
                        variant=variant,
                        grid=item.grid,
                        source=(height, width),
                        frames=min(count * 2, frames - first * 2),
                    )
                    operation = prepare(plan, d, bindings)
                    resources.callback(operation.close)
                    operations.append(operation)
                value = operations[0].run(bindings[0], vision.Inputs(pixels)).hidden
                for layer in range(1, 26):
                    operation = operations[1 + (layer % 2)]
                    value = operation.run(bindings[layer], vision.Inputs(value)).hidden
                value = operations[3].run(bindings[26], vision.Inputs(value)).hidden
                destination = d.view(result, value.shape, first * tokens // 4 * 5120)
                d.module.copy_bytes(destination, value)
                d.synchronize_compute()
        item.embedding, item.pixels = result, None
        return result

    def offload(self, request):
        for item in request.media:
            if item.embedding is not None and not isinstance(
                item.embedding, np.ndarray
            ):
                item.embedding = self.device.download(item.embedding, np.uint16)

    def rows(self, entries):
        selected = []
        for request, position, row in entries:
            for item in request.media:
                for first, count, offset in item.spans:
                    if first <= position < first + count:
                        selected.append((row, item, offset + position - first))
        if not selected:
            return None, None
        output = self.device.empty((len(selected), 5120), "bfloat16")
        indices = []
        cursor = 0
        while cursor < len(selected):
            row, item, first = selected[cursor]
            last = cursor + 1
            while (
                last < len(selected)
                and selected[last][1] is item
                and selected[last][2] == first + last - cursor
            ):
                last += 1
            source = self.device.view(
                self.encode(item), (last - cursor, 5120), first * 5120
            )
            destination = self.device.view(output, (last - cursor, 5120), cursor * 5120)
            self.device.module.copy_bytes(destination, source)
            indices.extend(value[0] for value in selected[cursor:last])
            cursor = last
        return output, self.device.upload(np.asarray(indices, np.int32))
