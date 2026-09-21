"""Stage private host metadata without an explicit wait after every CUDA copy."""

from collections.abc import Sequence

import torch


def device_tensor(
    values: Sequence,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # The source is private and never mutated after submission. CUDA consumers
    # use the same stream; pageable staging may still synchronize internally.
    return torch.tensor(values, device="cpu", dtype=dtype).to(device, non_blocking=True)
