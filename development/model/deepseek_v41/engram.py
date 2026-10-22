from dataclasses import dataclass
from math import isqrt

import numpy as np

MAX_INT64 = (1 << 63) - 1


@dataclass(frozen=True)
class HashLayout:
    max_ngram_size: int
    n_heads: int
    vocabulary_size: int
    primes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    multipliers: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        positive_integer(self.max_ngram_size, "ngram size")
        positive_integer(self.n_heads, "head count")
        positive_integer(self.vocabulary_size, "vocabulary size")
        if self.max_ngram_size < 2 or not self.primes:
            raise ValueError("hash layout needs layers and ngrams of size at least two")
        if len(self.primes) != len(self.offsets) or len(self.primes) != len(
            self.multipliers
        ):
            raise ValueError("hash layout layer counts differ")
        width = (self.max_ngram_size - 1) * self.n_heads
        for primes, offsets, multipliers in zip(
            self.primes, self.offsets, self.multipliers, strict=True
        ):
            if len(primes) != width or len(offsets) != width:
                raise ValueError("hash layout bucket counts differ")
            offset = 0
            for prime, actual in zip(primes, offsets, strict=True):
                positive_integer(prime, "bucket size")
                if actual != offset or prime < 2:
                    raise ValueError("hash layout buckets are not contiguous")
                offset += prime
            if offset > MAX_INT64 or len(multipliers) != self.max_ngram_size:
                raise ValueError("hash layout size is invalid")
            for multiplier in multipliers:
                positive_integer(multiplier, "multiplier")
                if (
                    multiplier % 2 != 1
                    or multiplier * (self.vocabulary_size - 1) > MAX_INT64
                ):
                    raise ValueError("hash multiplier can overflow int64")


def positive_integer(value, label):
    if type(value) is not int or not 0 < value <= MAX_INT64:
        raise ValueError(f"{label} must be a positive int64")


def compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    keys = {}
    mapping = []
    for token in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        raw = tokenizer.id_to_token(token)
        if raw is None:
            raise ValueError("tokenizer vocabulary has missing IDs")
        text = tokenizer.decode([token], skip_special_tokens=False)
        key = raw if "\ufffd" in text else normalize.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        mapping.append(keys[key])
    return np.asarray(mapping, dtype=np.int64), len(keys)


def is_prime(candidate):
    if candidate < 2:
        return False
    if candidate % 2 == 0:
        return candidate == 2
    return all(candidate % divisor for divisor in range(3, isqrt(candidate) + 1, 2))


def build_layout(
    *,
    layer_ids,
    max_ngram_size,
    n_heads,
    bucket_size,
    compressed_vocab_size,
    table_sizes,
):
    for value, label in (
        (max_ngram_size, "ngram size"),
        (n_heads, "head count"),
        (bucket_size, "bucket size"),
        (compressed_vocab_size, "vocabulary size"),
    ):
        positive_integer(value, label)
    if max_ngram_size < 2 or not layer_ids or len(layer_ids) != len(table_sizes):
        raise ValueError("invalid hash layer or ngram counts")
    if any(type(layer) is not int or layer < 0 for layer in layer_ids) or len(
        set(layer_ids)
    ) != len(layer_ids):
        raise ValueError("hash layer IDs must be distinct nonnegative integers")
    seen = set()
    primes, offsets, multipliers = [], [], []
    bound = max(1, (MAX_INT64 // compressed_vocab_size) // 2)
    for layer, table_size in zip(layer_ids, table_sizes, strict=True):
        positive_integer(table_size, "table size")
        sizes = []
        for _ in range(max_ngram_size - 1):
            candidate = bucket_size
            for _ in range(n_heads):
                while candidate in seen or not is_prime(candidate):
                    candidate += 1
                sizes.append(candidate)
                seen.add(candidate)
                candidate += 1
        if sum(sizes) != table_size:
            raise ValueError("prime buckets differ from checkpoint table size")
        primes.append(tuple(sizes))
        offsets.append(tuple(int(value) for value in np.cumsum([0, *sizes[:-1]])))
        generator = np.random.default_rng(10007 * layer)
        draws = generator.integers(0, bound, size=max_ngram_size, dtype=np.int64)
        multipliers.append(tuple(int(value) * 2 + 1 for value in draws))
    return HashLayout(
        max_ngram_size,
        n_heads,
        compressed_vocab_size,
        tuple(primes),
        tuple(offsets),
        tuple(multipliers),
    )


def token_sequence(tokens, vocabulary_size):
    if not isinstance(tokens, (tuple, list, np.ndarray)) or (
        isinstance(tokens, np.ndarray) and tokens.ndim != 1
    ):
        raise ValueError("hash tokens must be a one-dimensional integer sequence")
    if any(
        not isinstance(token, (int, np.integer))
        or isinstance(token, (bool, np.bool_))
        or not -1 <= token < vocabulary_size
        for token in tokens
    ):
        raise ValueError("hash tokens must be vocabulary IDs or dead-token markers")
    return np.asarray(tokens, dtype=np.int64)


def hash_rows(tokens, layout, *, pad_id, history=()):
    """Hash one sequence and return its last n-1 IDs, including media markers.

    Mathematical source: LOCAL-V41 inference/engram.py. A dead token (-1)
    blocks all further lookback; absent or blocked positions use compressed pad.
    """
    if (
        not isinstance(pad_id, (int, np.integer))
        or isinstance(pad_id, (bool, np.bool_))
        or not 0 <= pad_id < layout.vocabulary_size
    ):
        raise ValueError("hash padding must be a compressed vocabulary ID")
    pad_id = int(pad_id)
    tokens = token_sequence(tokens, layout.vocabulary_size)
    history = token_sequence(history, layout.vocabulary_size)
    if len(history) >= layout.max_ngram_size:
        raise ValueError("hash history exceeds the ngram lookback")
    joined = np.concatenate((history, tokens))
    positions = len(history) + np.arange(len(tokens))
    blocked = np.zeros(len(tokens), dtype=bool)
    lookback = []
    for shift in range(layout.max_ngram_size):
        valid = positions >= shift
        source = np.full(len(tokens), -1, dtype=np.int64)
        source[valid] = joined[positions[valid] - shift]
        blocked |= ~valid | (source == -1)
        lookback.append(np.where(blocked, pad_id, source))
    products = np.stack(lookback, axis=-1)[:, None, :] * np.asarray(layout.multipliers)
    running = products[:, :, 0]
    primes = np.asarray(layout.primes)
    parts = []
    for shift in range(1, layout.max_ngram_size):
        running = np.bitwise_xor(running, products[:, :, shift])
        begin = (shift - 1) * layout.n_heads
        parts.append(running[:, :, None] % primes[:, begin : begin + layout.n_heads])
    hashed = np.concatenate(parts, axis=-1) + np.asarray(layout.offsets)
    return hashed, tuple(int(token) for token in joined[-(layout.max_ngram_size - 1) :])
