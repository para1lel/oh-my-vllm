import numpy as np


def context_rows(batch, final):
    tokens, hidden_rows, boundaries, positions, slots, lengths, starts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [0],
    )
    active, writes, media_entries = [], [], []
    for index, (row, request) in enumerate(
        zip(batch.scheduled, batch.requests, strict=True)
    ):
        start = row["forward_start"]
        last = (
            request.state_position
            - 1
            + int(
                index in final
                and not request.finish
                and request.generated + 1 < request.limit
            )
        )
        begin = max(1, start, request.mtp_position + 1)
        if begin > last:
            continue
        active.append(index)
        for position in range(begin, last + 1):
            plan = batch.plans[index]
            write = position >= plan.end or plan.write_mask[position - start]
            page = plan.pages[position // 128]
            if page is None or (write and page["access"] != "write"):
                raise ValueError("MTP context requires its reserved private KV page")
            writes.append(write)
            media_entries.append((request, position, len(tokens)))
            tokens.append(request.history[position])
            hidden_rows.append(
                batch.schedule.sequence_rows[index] + position - start - 1
                if position > start
                else -1
            )
            boundaries.append(batch.schedule.read_slots[index])
            positions.append(request.position(position))
            slots.append(
                int(batch.metadata[2][index, position // 128]) * 128 + position % 128
            )
            lengths.append(position + 1)
        starts.append(len(tokens))
        request.mtp_position = last
    if not tokens:
        return None
    arrays = (
        np.asarray(tokens, np.int64),
        np.asarray(hidden_rows, np.int32),
        np.asarray(boundaries, np.int32),
        np.asarray(positions, np.int64).T.copy(),
        np.asarray(slots, np.int32),
        batch.metadata[2][active].copy(),
        np.asarray(starts, np.int32),
        np.asarray(lengths, np.int32),
        np.asarray(starts[1:], np.int32) - 1,
    )
    return tokens, active, media_entries, arrays, starts, lengths, writes
