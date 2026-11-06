from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class ContextMetadata:
    sequence_rows: tuple
    hidden_rows: tuple
    boundary_rows: tuple
    endpoint_rows: tuple


def make_context_metadata(
    *,
    token_positions,
    target_positions,
    boundary_positions,
    boundary_slots,
    endpoint_positions,
):
    batch = len(token_positions)
    if not 1 <= batch <= 32768 or any(
        len(values) != batch
        for values in (
            target_positions,
            boundary_positions,
            boundary_slots,
            endpoint_positions,
        )
    ):
        raise ValueError("MTP metadata batch dimensions differ")
    starts, hidden_rows, boundaries, endpoints = [0], [], [], []
    target_offset = 0
    for tokens, hidden, boundary, slot, endpoint in zip(
        token_positions,
        target_positions,
        boundary_positions,
        boundary_slots,
        endpoint_positions,
        strict=True,
    ):
        tokens, hidden = tuple(tokens), tuple(hidden)
        if not tokens or tokens[0] < 1 or slot < 0:
            raise ValueError(
                "MTP context requires positive token positions and a valid boundary"
            )
        if any(b != a + 1 for a, b in pairwise(tokens)) or any(
            b != a + 1 for a, b in pairwise(hidden)
        ):
            raise ValueError("MTP token and target positions must be contiguous")
        if hidden and hidden[0] < 0:
            raise ValueError("MTP target hidden position is negative")
        if endpoint not in tokens:
            raise ValueError("MTP endpoint is outside this context sequence")
        sources = {position: target_offset + row for row, position in enumerate(hidden)}
        for position in tokens:
            previous = position - 1
            if previous in sources:
                hidden_rows.append(sources[previous])
            elif previous == boundary:
                hidden_rows.append(-1)
            else:
                raise ValueError("MTP context has no target hidden for token p-1")
            boundaries.append(slot)
        endpoints.append(starts[-1] + endpoint - tokens[0])
        starts.append(starts[-1] + len(tokens))
        target_offset += len(hidden)
    if starts[-1] > 32768:
        raise ValueError("MTP context exceeds the row limit")
    return ContextMetadata(
        tuple(starts), tuple(hidden_rows), tuple(boundaries), tuple(endpoints)
    )
