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


def device_vectors(
    values: Sequence[Sequence[int]], *, device: str | torch.device = "cuda"
) -> tuple[torch.Tensor, ...]:
    """Copy integer batch metadata once, then expose disjoint device views."""
    lengths = [len(row) for row in values]
    packed = device_tensor(
        [value for row in values for value in row], device=device, dtype=torch.int64
    )
    return packed.split(lengths)
