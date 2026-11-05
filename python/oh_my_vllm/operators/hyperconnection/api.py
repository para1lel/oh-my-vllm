"""Inputs stay alive through Eager completion or graph release.

Preparation uses incoming_pre to collapse the residual. Its next_pre belongs
to the following sublayer. Expansion consumes the current sublayer's output.
Sinkhorn mixing adds epsilon after the initial row softmax and before the
first column normalization.
Preparation retains an FP32 residual copy, variance/inverse statistics, and a
32 MiB caller-owned workspace for the split and chunk partials. Preparation
is capture-safe. The complete logical row count selects the variance reduction
layout. hc_prepare_grouped uses source logical rows and batch-major step
storage. fp32_project accepts FP32 (rows, 5120) input and (outputs, 5120)
weights. Its workspace can be empty. Zero logical rows selects all physical
rows; otherwise, logical rows must divide physical rows. The native HC
projection writes the released reduction order: 32 interleaved split dots for
one source row and a measured split chunk for a multi-row step. The measured
router (384 outputs), DSpark router (128 outputs), and compressor (512 outputs)
tables cover 2..32768 rows.
The router and compressor single-row projections use a 32-lane warp reduction.
The DSpark markov head uses an eight-lane warp reduction, and its confidence
head uses the single-row split-dot order. hc_chunked_project applies the
measured split chunk to a two-dimensional HC input. Its workspace can be
empty. An empty workspace selects the sequential reference order. Prepare HC
workspaces before capture.
bf16_linear_rows1 accepts contiguous BF16 (rows, 5120) input and (512, 5120)
weights. It writes contiguous BF16 (rows, 512) output with separate storage.
It uses the caller stream and the existing single-row cuBLAS handle and call
for each row. Prepare the handle before capture. Rows must be in 0..32768.
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
    "bf16_linear",
    "bf16_linear_rows1",
    "markov_linear",
    "dot_linear",
    "hc_chunk",
    "project_chunk",
    "hc_chunked_project",
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
