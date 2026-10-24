from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    kind: str
    rows: int
    width: int
    heads: int
    epsilon: float | None
    inverse: bool

    @property
    def reuse_key(self):
        return self.kind, self.rows, self.width, self.heads, self.epsilon, self.inverse

    @property
    def storage(self):
        if self.kind == "widen":
            return (
                ((self.rows, self.width), "float32"),
                ((self.rows,), "uint8"),
            )
        shape = (
            (self.rows, self.heads, self.width)
            if self.kind == "rope"
            else (self.rows, self.width)
        )
        return ((shape, "bfloat16"),)


def make_plan(*, kind, rows, width, heads=1, epsilon=None, inverse=False):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("transform row count must be between zero and 384")
    if type(width) is not int or type(heads) is not int or type(inverse) is not bool:
        raise ValueError("transform width or inverse flag has an invalid type")
    if kind == "rms":
        if (
            width not in (512, 1024, 1280, 5120)
            or heads != 1
            or inverse
            or epsilon not in (1e-20, 1e-6)
        ):
            raise ValueError("unsupported RMS width, epsilon, or inverse flag")
    elif kind == "rope":
        if (
            width not in (128, 512)
            or heads not in (1, 8, 16, 32, 64)
            or epsilon is not None
        ):
            raise ValueError("unsupported RoPE width or epsilon")
    elif kind == "widen":
        if width != 5120 or heads != 1 or epsilon is not None or inverse:
            raise ValueError("unsupported widening width or parameters")
    else:
        raise ValueError("unknown transform kind")
    return Plan(kind, rows, width, heads, epsilon, inverse)
