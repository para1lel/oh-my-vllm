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


def device_page_tables(
    tables: Sequence[Sequence[int]],
    width: int,
    *,
    counts: Sequence[int] | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Upload one padded page row per request; expand token rows on the device."""
    if width <= 0 or not tables:
        raise ValueError("page tables require a positive width and request count")
    if counts is not None and (
        len(counts) != len(tables) or any(count < 0 for count in counts)
    ):
        raise ValueError("page-table token counts must match requests")
    if counts is not None and not any(counts):
        raise ValueError("page tables require at least one token row")
    compact_rows = []
    for table in tables:
        row = list(table[:width])
        compact_rows.append(row + [0] * (width - len(row)))
    compact = device_tensor(compact_rows, device=device, dtype=dtype)
    if counts is None or all(count == 1 for count in counts):
        return compact
    rows = [request for request, count in enumerate(counts) for _ in range(count)]
    indices = device_tensor(rows, device=device, dtype=torch.int64)
    return compact.index_select(0, indices)
