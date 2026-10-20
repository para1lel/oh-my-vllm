import numpy as np

MASK = (1 << 32) - 1


def philox(seed, offset):
    block, lane = divmod(offset, 4)
    counter = [block & MASK, (block >> 32) & MASK, 0, 0]
    key = [seed & MASK, (seed >> 32) & MASK]
    for _ in range(10):
        first, second = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [
            (second >> 32) ^ counter[1] ^ key[0],
            second & MASK,
            (first >> 32) ^ counter[3] ^ key[1],
            first & MASK,
        ]
        key = [(key[0] + 0x9E3779B9) & MASK, (key[1] + 0xBB67AE85) & MASK]
    return counter[lane]


def uniform(seed, offset):
    return float(
        np.float32(
            np.float32(philox(seed, offset)) * np.float32(2.0**-32)
            + np.float32(2.0**-33)
        )
    )


def accept_uniform(seed, offset):
    bits = philox(seed, offset) ^ (philox(seed, offset + 1) << 21)
    return bits * 2.0**-53 + 2.0**-54
