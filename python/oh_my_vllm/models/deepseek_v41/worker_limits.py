DEFAULT_CACHE_CONTEXT = 65536

MAX_NATIVE_ROWS = 384

MAX_PREFILL_ROWS = 32768

DEFAULT_PREFILL_ROWS = 8192

MAX_PREFILL_SUBBATCH = 16

DEFAULT_DECODE_GRAPHS = 32


def prefill_row_limit(value=None):
    rows = DEFAULT_PREFILL_ROWS if value is None else int(value)
    if not 1 <= rows <= MAX_PREFILL_ROWS:
        raise ValueError("DeepSeek prefill row limit must be between one and 32768")
    return rows


def prefill_subbatch_limit(rows):
    return min(MAX_PREFILL_SUBBATCH, max(4, rows // 2048))
