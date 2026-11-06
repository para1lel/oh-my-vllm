from itertools import pairwise

import torch


def _assert_released_sequential_projection(module, output, values, weight, block=8192):
    """Check the released single-chunk order through the HC sequential kernel.

    One 20480-wide chunk is a plain sequence of fused multiply-adds. Zero
    padding after width 5120 keeps that sequence for the FP32 shapes. Rows and
    outputs are compared in blocks to bound the temporary storage. The block
    stays in the 5187..32768 row range, where the HC chunk is 20480.
    """
    rows, outputs = values.size(0), weight.size(0)
    edges = [*list(range(0, rows, block)), rows]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    workspace = torch.empty(
        (module.workspace_size(block),), dtype=torch.uint8, device=values.device
    )
    for first, stop in pairwise(edges):
        padded = torch.zeros((stop - first, 20480), device=values.device)
        padded[:, :5120] = values[first:stop]
        for start in range(0, outputs, 24):
            count = min(24, outputs - start)
            padded_weight = torch.zeros((24, 20480), device=values.device)
            padded_weight[:count, :5120] = weight[start : start + count]
            part = torch.empty((stop - first, 24), device=values.device)
            module.hc_chunked_project(padded, padded_weight, part, workspace)
            assert torch.equal(
                output[first:stop, start : start + count], part[:, :count]
            )
