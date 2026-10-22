"""Source image-placeholder expansion for the independent CPU reference."""

from dataclasses import dataclass

import numpy as np

from .reference_image import MAX_PREPARED_BYTES, PreparedImage, prepare
from .weights import select_rows


@dataclass(frozen=True)
class ImageSpan:
    start: int
    image: PreparedImage


@dataclass(frozen=True)
class MediaPrompt:
    tokens: np.ndarray
    token_types: np.ndarray
    spans: tuple[ImageSpan, ...]


def expand(tokens, encoded_images, config):
    text = config["text_config"]
    tokens = select_rows(tokens, text["vocab_size"])
    capacity = text["max_position_embeddings"]
    image_id = config["image_token_id"]
    if type(capacity) is not int or not 1 <= capacity <= 1048576:
        raise ValueError("invalid reference context capacity")
    select_rows([image_id], text["vocab_size"])
    if not 0 < len(tokens) <= capacity:
        raise ValueError("reference prompt is empty or exceeds context capacity")
    if not isinstance(encoded_images, (tuple, list)):
        raise ValueError("reference images must be an ordered sequence of bytes")
    count = int(np.count_nonzero(tokens == image_id))
    if count != len(encoded_images):
        raise ValueError("image placeholders differ from the supplied image count")
    if count and config["vision_config"]["num_hidden_layers"] < 1:
        raise ValueError("reference configuration has no vision tower")
    if len(tokens) + 3 * count > capacity:
        raise ValueError("minimum image spans exceed context capacity")
    expanded, types, spans = [], [], []
    prepared_bytes = 0
    length = len(tokens)
    for token in tokens:
        if token != image_id:
            expanded.append(int(token))
            types.append(-1)
            continue
        image = prepare(encoded_images[len(spans)], config["vision_config"])
        length += image.plan.tokens - 1
        if length > capacity:
            raise ValueError("expanded media prompt exceeds context capacity")
        prepared_bytes += image.patches.nbytes
        if prepared_bytes > MAX_PREPARED_BYTES:
            raise ValueError("combined reference patches exceed their byte bound")
        image.patches.setflags(write=False)
        image.token_types.setflags(write=False)
        spans.append(ImageSpan(len(expanded), image))
        expanded.extend([image_id] * image.plan.tokens)
        types.extend(image.token_types)
    output = np.asarray(expanded, dtype=np.int64)
    token_types = np.asarray(types, dtype=np.int64)
    output.setflags(write=False)
    token_types.setflags(write=False)
    return MediaPrompt(output, token_types, tuple(spans))
