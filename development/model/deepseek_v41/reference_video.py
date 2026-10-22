"""CPU video selection and ordered image blocks for the local V4.1 encoder.

The caller decodes frames and enforces byte limits before this pure codec step.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MAX_FRAMES = 768
MAX_FRAME_COUNT = 1 << 53


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float

    def __post_init__(self):
        if (
            type(self.frame_count) is not int
            or not 1 <= self.frame_count <= MAX_FRAME_COUNT
        ):
            raise ValueError("frame count must be an integer from 1 to 2**53")
        if type(self.fps) not in (int, float):
            raise ValueError("fps must be positive and finite")
        try:
            fps = float(self.fps)
        except OverflowError as error:
            raise ValueError("fps must be positive and finite") from error
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        object.__setattr__(self, "fps", fps)


def sample_plan(metadata: VideoMetadata) -> tuple[int, ...]:
    """Select source indices at approximately two frames per second."""
    if not isinstance(metadata, VideoMetadata):
        raise ValueError("metadata must be validated VideoMetadata")
    count = min(
        metadata.frame_count,
        max(4, int(min(MAX_FRAMES, metadata.frame_count / metadata.fps * 2))),
    )
    return tuple(
        int(index) for index in np.linspace(0, metadata.frame_count - 1, count).round()
    )


def video_records(
    metadata: VideoMetadata, frames: Sequence[tuple[int, dict]]
) -> tuple[dict, ...]:
    """Expand one video's selected (source index, image block) records in order."""
    selected = sample_plan(metadata)
    if not isinstance(frames, Sequence) or len(frames) != len(selected):
        raise ValueError("frame record count must match the sample plan")
    blocks = []
    for expected, frame in zip(selected, frames, strict=True):
        if not isinstance(frame, (tuple, list)) or len(frame) != 2:
            raise ValueError("frame record must contain an index and an image block")
        index, image = frame
        if type(index) is not int or index != expected:
            raise ValueError("frame indices must match the sample plan in order")
        if (
            not isinstance(image, dict)
            or image.get("type") != "image"
            or not any(image.get(key) for key in ("source", "url", "data"))
        ):
            raise ValueError("frame must contain a local encoder image block")
        timestamp = index / metadata.fps
        if not math.isfinite(timestamp):
            raise ValueError("frame timestamp must be finite")
        blocks.append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
        blocks.append(dict(image))
    return tuple(blocks)
