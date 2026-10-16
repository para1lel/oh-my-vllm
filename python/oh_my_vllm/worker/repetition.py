MAX_PATTERN_TOKENS = 256
MIN_PATTERN_REPEATS = 3
MIN_REPEATED_TOKENS = 64
HISTORY_WINDOW = MAX_PATTERN_TOKENS * MIN_PATTERN_REPEATS


def cycle_continuations(history):
    if len(history) < MIN_REPEATED_TOKENS:
        return set()
    tail = list(reversed(history[-HISTORY_WINDOW:]))
    size = len(tail)
    matches = [0] * size
    left = right = 0
    for index in range(1, size):
        if index <= right:
            matches[index] = min(right - index + 1, matches[index - left])
        while (
            index + matches[index] < size
            and tail[matches[index]] == tail[index + matches[index]]
        ):
            matches[index] += 1
        if index + matches[index] - 1 > right:
            left, right = index, index + matches[index] - 1
    blocked = set()
    for period in range(1, min(MAX_PATTERN_TOKENS, size // MIN_PATTERN_REPEATS) + 1):
        span = max(MIN_PATTERN_REPEATS * period, MIN_REPEATED_TOKENS)
        if span <= size and matches[period] >= span - period:
            blocked.add(tail[period - 1])
    return blocked
