from dataclasses import dataclass

WORKSPACE_BYTES = 32 << 20


@dataclass(frozen=True)
class Plan:
    rows: int
    outputs: int
    width: int
    input_dtype: str
    output_dtype: str
    groups: int

    @property
    def input_shape(self):
        return (
            (self.rows, self.width)
            if self.groups == 1
            else (self.rows, self.groups, self.width)
        )

    @property
    def weight_shape(self):
        return (
            (self.outputs, self.width)
            if self.groups == 1
            else (self.groups, self.outputs, self.width)
        )

    @property
    def output_shape(self):
        return (
            (self.rows, self.outputs)
            if self.groups == 1
            else (self.rows, self.groups, self.outputs)
        )


def make_plan(
    *,
    rows,
    outputs,
    width,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    groups=1,
):
    if (
        any(type(value) is not int for value in (rows, outputs, width, groups))
        or not 0 <= rows <= 384
        or not 1 <= outputs <= 129280
        or not 1 <= width <= 8192
        or type(input_dtype) is not str
        or type(output_dtype) is not str
        or (input_dtype, output_dtype)
        not in {
            ("bfloat16", "bfloat16"),
            ("bfloat16", "float32"),
            ("float32", "float32"),
        }
        or groups not in (1, 2, 8)
        or (
            groups != 1
            and (input_dtype, output_dtype, outputs, width)
            != ("bfloat16", "bfloat16", 1024, 4096)
        )
    ):
        raise ValueError("invalid unquantized projection dimensions or types")
    return Plan(rows, outputs, width, input_dtype, output_dtype, groups)
