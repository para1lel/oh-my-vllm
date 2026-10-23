"""BF16 input; encoded values, scales, BF16 roundtrip and per-group status.

All status entries must be zero before a result can be accepted. Nonzero
entries reject that group and clear its payload: 1 means nonfinite input,
2 means a scale outside finite normal UE8M0, 3 means FP32/BF16 overflow.
Status is written on every invocation, including CUDA Graph replay.

Arithmetic follows development/model/deepseek_v41/reference_quantization.py:
FP32 reciprocal multiplication for UE8M0 scale selection, FP32 division for
E4M3 scale selection and normalization, RN-even conversion, and low nibble
channel 2j. This is source precision, not the mathematical FP64 reference.
"""

from dataclasses import dataclass
from typing import NamedTuple

FUNCTIONS = ("quantize",)


@dataclass(frozen=True)
class Inputs:
    value: object


class Outputs(NamedTuple):
    encoded: object
    scales: object
    dequantized: object
    status: object


def run(workspace, inputs):
    return workspace.run(inputs)
