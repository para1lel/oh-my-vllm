from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    frame_tokens: int
    variant: str
    grid: tuple
    source: tuple
    frames: int
    storage: tuple

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.frame_tokens,
            self.variant,
            self.grid,
            self.source,
            self.frames,
        )


def make_plan(*, rows, grid, frames=1, source=None, variant="default"):
    if variant not in ("input", "default", "merge"):
        raise ValueError("unknown vision block variant")
    if grid is None or len(grid) != 2:
        raise ValueError("vision grid must contain height and width")
    height, width = grid
    tokens = height * width
    if min(height, width, rows, frames) <= 0 or height % 2 or width % 2:
        raise ValueError("vision grid and row count are not valid")
    if rows % tokens or rows > 65536 or not 4 <= tokens <= 65536:
        raise ValueError("vision dimensions exceed the native operator limits")
    if rows % (tokens * ((frames + 1) // 2)):
        raise ValueError("vision rows do not match the temporal groups")
    source = source or (height * 16, width * 16)
    if len(source) != 2 or min(source) <= 0:
        raise ValueError("source image dimensions must be positive")
    groups = rows // tokens * 16
    storage = (
        ((rows, 1152), "bfloat16"),
        ((rows // 4, 5120) if variant == "merge" else (rows, 1152), "bfloat16"),
        ((rows, 1152), "bfloat16"),
        ((rows * 4304,), "float32"),
        *(((groups, tokens, 128), "bfloat16"),) * 4,
        ((groups, tokens), "float32"),
        ((rows, 1152), "bfloat16"),
        ((rows * 4304,), "bfloat16"),
    )
    if rows > 32:
        storage += (((32 * 1024 * 1024,), "uint8"),)
    if variant == "input":
        storage += (((rows, 1536), "bfloat16"),)
    return Plan(rows, tokens, variant, tuple(grid), tuple(source), frames, storage)
