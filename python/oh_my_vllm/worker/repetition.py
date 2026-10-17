import numpy as np

MAX_PATTERN_TOKENS = 256
MIN_PATTERN_REPEATS = 3
MIN_REPEATED_TOKENS = 64
HISTORY_WINDOW = MAX_PATTERN_TOKENS * MIN_PATTERN_REPEATS


def cycle_continuations(history):
    if len(history) < MIN_REPEATED_TOKENS:
        return set()
    tail = np.asarray(history[-HISTORY_WINDOW:], dtype=np.int64)[::-1]
    size = len(tail)
    limit = min(MAX_PATTERN_TOKENS, size // MIN_PATTERN_REPEATS)
    matches = tail[1 : limit + 1] == tail[0]
    # A valid cycle has at least 43 matching tokens, so this filter cannot reject it.
    for index in range(1, 4):
        matches &= tail[1 + index : limit + 1 + index] == tail[index]
    tail_bytes = tail.tobytes()
    token_bytes = tail.itemsize
    blocked = set()
    for offset in np.flatnonzero(matches):
        period = int(offset) + 1
        continuation = int(tail[period - 1])
        if continuation in blocked:
            continue
        span = max(MIN_PATTERN_REPEATS * period, MIN_REPEATED_TOKENS)
        if span <= size and tail_bytes.startswith(
            tail_bytes[: (span - period) * token_bytes],
            period * token_bytes,
            span * token_bytes,
        ):
            blocked.add(continuation)
    return blocked
