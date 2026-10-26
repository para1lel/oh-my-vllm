from dataclasses import dataclass

from ...operators.runtime.storage import storage_bytes


@dataclass(frozen=True)
class CacheLayout:
    mamba_blocks: int
    parallel: int
    token_budget: int
    fa_layers: int

    @property
    def private_start(self):
        return self.mamba_blocks

    @property
    def scratch_start(self):
        return self.private_start + self.parallel * 6

    @property
    def slots(self):
        return self.scratch_start + (self.token_budget + 783) // 784 + self.parallel

    @property
    def fa_block_shape(self):
        return (2, 784, 4, 256)

    @property
    def fa_block_bytes(self):
        return self.fa_layers * storage_bytes(((self.fa_block_shape, "bfloat16"),))

    @property
    def state_shape(self):
        return (self.slots, 48, 128, 128)

    @property
    def convolution_shape(self):
        return (self.slots, 10240, 3)

    @property
    def hidden_shape(self):
        return (self.slots, 5120)

    @property
    def gdn_bytes(self):
        return storage_bytes(
            ((self.state_shape, "float32"),) * 48
            + ((self.convolution_shape, "bfloat16"),) * 48
            + ((self.hidden_shape, "bfloat16"),)
        )


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def gpu_blocks(config):
    blocks = config["num_gpu_blocks"]
    if blocks is None:
        blocks = 1608 if config["num_speculative_tokens"] else 1824
    if type(blocks) is not int or not 2 <= blocks <= 0xFFFFFFFF:
        raise ValueError("num_gpu_blocks must be an integer in [2, 4294967295]")
    return blocks


def cache_layout(config):
    parallel = positive_integer(config["max_num_seqs"], "max_num_seqs")
    tokens = positive_integer(
        config["max_num_batched_tokens"], "max_num_batched_tokens"
    )
    mamba_blocks = config["mamba_blocks"]
    if mamba_blocks is None:
        mamba_blocks = 128
    if positive_integer(mamba_blocks, "mamba_blocks") < 2:
        raise ValueError("GDN cache needs a null slot and at least one usable slot")
    layout = CacheLayout(
        mamba_blocks,
        parallel,
        min(32768, tokens),
        16 + bool(config["num_speculative_tokens"]),
    )
    if layout.slots > 65536:
        raise ValueError("GDN cache exceeds the state slot limit")
    return layout


def cache_budget(config, *, free_bytes, total_bytes, overhead_bytes):
    """Subtract only future overhead; the memory snapshot includes loaded weights."""
    positive_integer(total_bytes, "total_bytes")
    if type(free_bytes) is not int or not 0 <= free_bytes <= total_bytes:
        raise ValueError("free_bytes must be within the device memory capacity")
    if type(overhead_bytes) is not int or overhead_bytes < 0:
        raise ValueError("overhead_bytes must be a nonnegative integer")
    layout = cache_layout(config)
    limit = total_bytes * 88 // 100
    used = total_bytes - free_bytes
    available = limit - used - overhead_bytes - layout.gdn_bytes
    maximum = min(available // layout.fa_block_bytes, 0xFFFFFFFF)
    blocks = gpu_blocks(config)
    if blocks > maximum:
        raise ValueError(
            f"cache does not fit the device memory budget: requested={blocks}, "
            f"maximum={maximum}, limit_bytes={limit}, used_bytes={used}, "
            f"overhead_bytes={overhead_bytes}, gdn_bytes={layout.gdn_bytes}, "
            f"fa_block_bytes={layout.fa_block_bytes}"
        )
    fa_bytes = blocks * layout.fa_block_bytes
    return {
        "policy": "static_blocks",
        "requested_num_gpu_blocks": config["num_gpu_blocks"],
        "num_gpu_blocks": blocks,
        "mamba_blocks": layout.mamba_blocks,
        "gdn_slots": layout.slots,
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
        "used_bytes": used,
        "limit_bytes": limit,
        "overhead_bytes": overhead_bytes,
        "gdn_bytes": layout.gdn_bytes,
        "fa_block_bytes": layout.fa_block_bytes,
        "fa_bytes": fa_bytes,
        "planned_bytes": used + overhead_bytes + layout.gdn_bytes + fa_bytes,
    }
