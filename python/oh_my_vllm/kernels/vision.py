from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from .native import load

FUNCTIONS = (
    "vision_patch",
    "vision_block",
    "vision_merge",
    "vision_classic_prepare",
    "vision_classic_close",
    "vision_classic_images",
    "vision_classic_state",
    "vision_classic_graph_lease",
)


@dataclass(frozen=True)
class Weights:
    blocks: tuple
    patch: tuple
    merger: tuple


class VisionBlock:
    def __init__(
        self,
        weights,
        rows,
        frame_tokens,
        allocate,
        *,
        variant="default",
        grid=None,
        source=None,
        frames=1,
        layer=None,
    ):
        if variant not in ("input", "default", "merge"):
            raise ValueError("unknown vision block variant")
        if grid is None or len(grid) != 2:
            raise ValueError("vision grid must contain height and width")
        height, width = grid
        if min(height, width, rows, frames) <= 0 or height % 2 or width % 2:
            raise ValueError("vision grid and row count are not valid")
        if height * width != frame_tokens or rows % frame_tokens:
            raise ValueError("vision rows do not match the frame grid")
        if rows > 65536 or not 4 <= frame_tokens <= 65536:
            raise ValueError("vision dimensions exceed the native operator limits")
        if rows % (frame_tokens * ((frames + 1) // 2)):
            raise ValueError("vision rows do not match the temporal groups")
        source = source or (height * 16, width * 16)
        if min(source) <= 0:
            raise ValueError("source image dimensions must be positive")
        self.module, self.source_id, self.library = load("vision", FUNCTIONS)
        self.weights, self.variant = weights, variant
        self.grid, self.source, self.frames = grid, source, frames
        self.layer = (
            (0 if variant == "input" else 26 if variant == "merge" else 1)
            if layer is None
            else layer
        )
        if not 0 <= self.layer < 27:
            raise ValueError("vision layer must be between zero and 26")
        groups = rows // frame_tokens * 16
        self.hidden = allocate((rows, 1152), "bfloat16")
        self.output = allocate(
            (rows // 4, 5120) if variant == "merge" else (rows, 1152), "bfloat16"
        )
        self.normalized = allocate((rows, 1152), "bfloat16")
        self.projected = allocate((rows * 4304,), "float32")
        self.qkv = tuple(
            allocate((groups, frame_tokens, 128), "bfloat16") for _ in range(3)
        )
        self.padded_attention = allocate((groups, frame_tokens, 128), "bfloat16")
        self.normalizer = allocate((groups, frame_tokens), "float32")
        self.attended = allocate((rows, 1152), "bfloat16")
        self.activated = allocate((rows * 4304,), "bfloat16")
        self.classic = None
        self.closed = False
        if rows > 32:
            self._prepare_classic(allocate)
        self.patches = (
            allocate((rows, 1536), "bfloat16") if variant == "input" else None
        )

    def _prepare_classic(self, allocate):
        workspace = allocate((32 * 1024 * 1024,), "uint8")
        self.classic = self.module.vision_classic_prepare(workspace)

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("VisionBlock is closed")
        module = self.module
        lease = (
            module.vision_classic_graph_lease(self.classic)
            if self.classic is not None
            else None
        )
        try:
            yield
        finally:
            del lease

    def graph(self, graph_factory, value):
        operation = partial(self, value)
        operation.graph_lease = self.graph_lease
        return graph_factory(operation)

    def close(self):
        if not self.closed:
            if self.classic is not None:
                self.module.vision_classic_close(self.classic)
                self.classic = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __call__(self, value):
        if self.closed:
            raise RuntimeError("VisionBlock is closed")
        if self.variant == "input":
            self.module.vision_patch(
                value,
                *self.weights.patch,
                self.patches,
                self.projected,
                self.hidden,
                *self.grid,
            )
            value = self.hidden
        block_output = self.hidden if self.variant == "merge" else self.output
        self.module.vision_block(
            value,
            *self.weights.blocks[self.layer],
            block_output,
            self.normalized,
            self.projected,
            *self.qkv,
            self.padded_attention,
            self.normalizer,
            self.attended,
            self.activated,
            *self.grid,
            self.classic,
        )
        if self.variant == "merge":
            self.module.vision_merge(
                block_output,
                *self.weights.merger,
                self.output,
                self.normalized,
                self.projected,
                self.activated,
            )
        return self.output
