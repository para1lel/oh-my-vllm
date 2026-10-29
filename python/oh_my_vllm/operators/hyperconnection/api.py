"""Inputs stay alive through Eager completion or graph release.

Preparation uses incoming_pre to collapse the residual. Its next_pre belongs
to the following sublayer. Expansion consumes the current sublayer's output.
Sinkhorn mixing adds epsilon after the initial row softmax and before the
first column normalization.
Preparation retains an FP32 residual copy, variance/inverse statistics, and
a 32 MiB cuBLAS workspace. Prepare the thread-local handle before capture.
The complete logical row count selects the variance reduction layout.
hc_prepare_grouped uses source logical rows and batch-major step storage.
fp32_project accepts FP32 (rows, 5120) input and (128|384, 5120) weights.
It uses the same caller-owned cuBLAS workspace. Zero logical rows selects all
physical rows; otherwise, logical rows must divide physical rows.
"""

from typing import NamedTuple

BLAS_WORKSPACE_BYTES = 32 * 1024 * 1024
FUNCTIONS = (
    "workspace_size",
    "hc_warmup",
    "hc_images",
    "hc_prepare",
    "hc_prepare_grouped",
    "fp32_project",
    "hc_expand",
)


class Weights(NamedTuple):
    projection: object
    scale: object
    base: object


class Inputs(NamedTuple):
    residual: object
    incoming_pre: object
    branch: object


class Outputs(NamedTuple):
    collapsed: object
    expanded: object
    next_pre: object
