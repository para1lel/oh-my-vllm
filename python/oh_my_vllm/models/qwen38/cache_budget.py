from dataclasses import dataclass, replace

from ...operators.runtime.storage import storage_bytes

PAGE_TOKENS = 128
CHECKPOINT_TOKENS = 8192
PAGES_PER_CHECKPOINT = 64
LAYOUT_ID = "qwen_radix_gdn8192_fp8kv128_v1"
FORMAT_FINGERPRINT = "qwen_fp8_e4m3fn_rne_fp32scale_v1"
CHECKPOINT_BYTES = 153954304


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CacheLayout:
    checkpoints: int
    parallel: int
    token_budget: int
    fa_layers: int
    max_context: int = 262144

    @property
    def pages(self):
        return self.checkpoints * PAGES_PER_CHECKPOINT

    @property
    def table_width(self):
        return (self.max_context + PAGE_TOKENS - 1) // PAGE_TOKENS

    @property
    def private_start(self):
        return self.checkpoints + 1

    @property
    def private_slots(self):
        return self.parallel * 6

    @property
    def scratch_start(self):
        return self.private_start + self.private_slots

    @property
    def scratch_slots(self):
        return (
            self.token_budget + CHECKPOINT_TOKENS - 1
        ) // CHECKPOINT_TOKENS + self.parallel

    @property
    def slots(self):
        return self.scratch_start + self.scratch_slots

    @property
    def codes_shape(self):
        return (self.pages + 1, 2, PAGE_TOKENS, 4, 256)

    @property
    def scales_shape(self):
        return (self.pages + 1, 2, PAGE_TOKENS, 4)

    @property
    def page_bytes(self):
        return self.fa_layers * storage_bytes(
            (((2, PAGE_TOKENS, 4, 256), "uint8"), ((2, PAGE_TOKENS, 4), "float32"))
        )

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
    def checkpoint_bytes(self):
        return CHECKPOINT_BYTES + (10240 if self.fa_layers == 17 else 0)

    @property
    def pool_bytes(self):
        return {
            "kv_codes": self.fa_layers * storage_bytes(((self.codes_shape, "uint8"),)),
            "kv_scales": self.fa_layers
            * storage_bytes(((self.scales_shape, "float32"),)),
            "persistent_checkpoints": self.checkpoints * self.checkpoint_bytes,
            "null_checkpoint": self.checkpoint_bytes,
            "private_checkpoints": self.private_slots * self.checkpoint_bytes,
            "scratch_checkpoints": self.scratch_slots * self.checkpoint_bytes,
        }

    @property
    def bytes(self):
        return sum(self.pool_bytes.values())

    def wire(self):
        return {
            "kind": "qwen_radix_v1",
            "layout_id": LAYOUT_ID,
            "gdn_period_tokens": CHECKPOINT_TOKENS,
            "kv_page_tokens": PAGE_TOKENS,
            "usable_checkpoints": self.checkpoints,
            "usable_pages": self.pages,
            "null_checkpoints": 1,
            "null_pages": 1,
            "private_checkpoints": self.private_slots,
            "scratch_checkpoints": self.scratch_slots,
            "pool_bytes": self.pool_bytes,
            "format_fingerprint": FORMAT_FINGERPRINT,
        }


def layout_template(config, checkpoints=0):
    tokens = positive_integer(
        config["max_num_batched_tokens"], "max_num_batched_tokens"
    )
    context = positive_integer(config["max_model_len"], "max_model_len")
    if (
        tokens > 32768
        or context > 262144
        or config["num_speculative_tokens"] not in (0, 4)
    ):
        raise ValueError("Qwen batch, context, or speculation exceeds its contract")
    layout = CacheLayout(
        checkpoints,
        positive_integer(config["max_num_seqs"], "max_num_seqs"),
        tokens,
        16 + bool(config["num_speculative_tokens"]),
        context,
    )
    if layout.slots > 65536:
        raise ValueError("Qwen checkpoint capacity exceeds the state slot limit")
    return layout


def target_token_limit(config):
    boundary = max(4, config["max_num_seqs"]) if config["num_speculative_tokens"] else 0
    return min(config["max_num_batched_tokens"], 32768 - boundary)


def cache_layout(config):
    layout = layout_template(
        config, positive_integer(config["mamba_blocks"], "mamba_blocks")
    )
    if config["num_gpu_blocks"] != layout.pages:
        raise ValueError("usable KV pages must equal 64 times usable checkpoints")
    return layout


def capacity_plan(config, *, byte_limit, workspace_bytes=0):
    if (
        type(byte_limit) is not int
        or type(workspace_bytes) is not int
        or workspace_bytes < 0
    ):
        raise ValueError("invalid physical cache byte budget")
    template = layout_template(config)
    increment = template.checkpoint_bytes + PAGES_PER_CHECKPOINT * template.page_bytes
    maximum = min(
        (byte_limit - workspace_bytes - template.bytes) // increment,
        65536 - template.slots,
    )
    checkpoints = config["mamba_blocks"]
    pages = config["num_gpu_blocks"]
    if pages is not None:
        positive_integer(pages, "num_gpu_blocks")
        if pages % PAGES_PER_CHECKPOINT:
            raise ValueError("num_gpu_blocks must be a multiple of 64 usable pages")
        if checkpoints is not None and checkpoints != pages // PAGES_PER_CHECKPOINT:
            raise ValueError("conflicting checkpoint and page capacities")
        checkpoints = pages // PAGES_PER_CHECKPOINT
    if checkpoints is None:
        checkpoints = maximum
    if positive_integer(checkpoints, "mamba_blocks") > maximum:
        raise ValueError(
            f"physical cache budget exhausted: requested={checkpoints}, "
            f"maximum={maximum}"
        )
    layout = replace(template, checkpoints=checkpoints)
    return layout, {
        "policy": "radix_fixed_slabs",
        "capacity_verified": False,
        "num_gpu_blocks": layout.pages,
        "mamba_blocks": layout.checkpoints,
        "gdn_slots": layout.slots,
        "table_width": layout.table_width,
        "pool_bytes": layout.pool_bytes,
        "workspace_bytes": workspace_bytes,
        "planned_bytes": layout.bytes + workspace_bytes,
        "limit_bytes": byte_limit,
    }


def legacy_cache_ceiling(config):
    parallel = positive_integer(config["max_num_seqs"], "max_num_seqs")
    tokens = min(
        32768,
        positive_integer(config["max_num_batched_tokens"], "max_num_batched_tokens"),
    )
    mtp = bool(config["num_speculative_tokens"])
    states = 128 + 6 * parallel + (tokens + 783) // 784 + parallel
    pages = 1608 if mtp else 1824
    return states * CHECKPOINT_BYTES + pages * (16 + mtp) * 2 * 784 * 4 * 256 * 2


def cache_budget(config, *, free_bytes, total_bytes, overhead_bytes):
    positive_integer(total_bytes, "total_bytes")
    if type(free_bytes) is not int or not 0 <= free_bytes <= total_bytes:
        raise ValueError("free_bytes must be within the device memory capacity")
    used = total_bytes - free_bytes
    limit = min(total_bytes * 88 // 100 - used, legacy_cache_ceiling(config))
    _, plan = capacity_plan(config, byte_limit=limit, workspace_bytes=overhead_bytes)
    plan.update(total_bytes=total_bytes, free_bytes=free_bytes, used_bytes=used)
    return plan
