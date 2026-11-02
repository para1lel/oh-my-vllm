from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    width: int
    limit: int
    routed: bool

    @property
    def reuse_key(self):
        return self.rows, self.width, self.limit, self.routed

    @property
    def storage(self):
        return (((self.rows, self.width), "bfloat16"),)


def make_plan(*, rows, width, limit=10, routed=False):
    if type(rows) is not int or not 0 <= rows <= 2304:
        raise ValueError("SwiGLU row count must be between zero and 2304")
    if type(width) is not int or type(limit) is not int or type(routed) is not bool:
        raise ValueError("SwiGLU width, limit, or routing flag has an invalid type")
    if (width in (576, 2304) and limit == 10) or (
        width == 2816 and limit == 0 and not routed
    ):
        return Plan(rows, width, limit, routed)
    raise ValueError("unsupported SwiGLU width, limit, or routing flag")
