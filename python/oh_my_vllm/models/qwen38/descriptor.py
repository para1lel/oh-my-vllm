"""Pure Qwen descriptor construction; this module does not load a backend.

Both cache layouts use the existing paired FA/GDN reservation and commit.
They describe owned storage, not a shared-producer relationship:

* qwen38_paired_fa_kv_bf16 v1: each layer owns BF16
  [capacity_blocks, 2, 784, 4, 256] K/V storage. Target layer IDs are 0..63.
  Logical layer 64 names the single MTP predictor; its FA storage is present
  only when four-draft MTP is configured, after the 16 target FA arrays.
* qwen38_paired_gdn_f32_conv_bf16_hidden_bf16 v1: each of the 48 GDN
  layers owns FP32 [slots, 48, 128, 128] recurrent state and BF16
  [slots, 10240, 3] convolution history. One BF16 [slots, 5120] target
  boundary hidden-state array belongs to the same checkpoint bundle.
  capacity_blocks is the resolved mamba_blocks pool, including null slot 0.
  Private and scratch slots are additional workspace, not scheduler blocks:
  slots = mamba_blocks + 6*parallel + ceil(min(32768, token_budget)/784)
          + parallel.

Both block units are 784 tokens, including partial endpoint checkpoints.
FA capacity also includes null block 0. Neither group can be admitted or
committed independently by a future consumer of these paired layouts.
"""

import hashlib
import json

from ...operators.specification import MODEL
from ..descriptor import identity_digest, validate_descriptor
from .adapter import SPEC

MAX_CONFIG_BYTES = 1 << 20
TOKENIZER_ASSETS = frozenset(
    ("tokenizer.json", "generation_config.json", "chat_template.jinja")
)


def _same_config(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(
            _same_config(value[key], item) for key, item in expected.items()
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _same_config(a, b) for a, b in zip(value, expected, strict=True)
        )
    return value == expected


def _verified_config(config, raw):
    if (
        type(raw) is not bytes
        or not 1 <= len(raw) <= MAX_CONFIG_BYTES
        or hashlib.sha256(raw).hexdigest() != MODEL["config_sha256"]
    ):
        raise ValueError("checkpoint configuration differs from the model contract")
    parsed = json.loads(raw)
    if not _same_config(config, parsed):
        raise ValueError("configuration differs from the verified original bytes")
    if (
        config.get("architectures") != [SPEC.architecture]
        or config.get("model_type") != SPEC.model_type
    ):
        raise ValueError("unsupported Qwen configuration")
    SPEC.validate_config(config)
    text = config.get("text_config")
    expected = {
        "vocab_size": MODEL["vocabulary"],
        "num_hidden_layers": MODEL["layers"],
        "max_position_embeddings": MODEL["context_tokens"],
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    expected["layer_types"] *= MODEL["fa_layers"]
    if type(text) is not dict or any(
        not _same_config(text.get(key), value) for key, value in expected.items()
    ):
        raise ValueError("unsupported Qwen text configuration")
    return text


def _capacity(value, maximum, name):
    if type(value) is not int or not 2 <= value <= maximum:
        raise ValueError(f"{name} must include a null block and usable storage")
    return value


def _group(name, kind, layout, layers, capacity):
    return {
        "name": name,
        "kind": kind,
        "block_tokens": MODEL["block_size"],
        "capacity_blocks": capacity,
        "storage": {"type": "layout", "id": layout, "version": 1},
        "layers": layers,
        "source": {"type": "owned"},
    }


def build_descriptor(
    config,
    *,
    raw_config,
    checkpoint_identity,
    tokenizer_identity,
    fa_capacity_blocks,
    gdn_capacity_blocks,
    num_speculative_tokens=0,
):
    """Build a fresh named map from validated config and loaded identities.

    Pass Checkpoint.identity (or verify_identity()) and Tokenization.identity
    after loading. Every loaded shard and tokenizer asset is hashed into the
    descriptor, even when not used by the selected speculation mode.
    The caller owns file verification and must freeze these maps at readiness.
    Capacities are resolved pool counts from Cache/Ready, not byte budgets or
    maximum context settings. This helper does not check available GPU memory.
    Speculation lists supported MTP limits; cache layers report configured use.
    """
    text = _verified_config(config, raw_config)
    if type(num_speculative_tokens) is not int or num_speculative_tokens not in (0, 4):
        raise ValueError("Qwen supports ordinary decode or four-draft MTP only")
    fa_capacity_blocks = _capacity(fa_capacity_blocks, 0xFFFFFFFF, "FA capacity")
    gdn_capacity_blocks = _capacity(gdn_capacity_blocks, 65536, "GDN capacity")
    checkpoint_sha = identity_digest(checkpoint_identity, domain="checkpoint")
    tokenizer_sha = identity_digest(tokenizer_identity, domain="tokenizer")
    if (
        checkpoint_identity.get("config") != MODEL["config_sha256"]
        or "index" not in checkpoint_identity
        or len(checkpoint_identity) < 3
        or any(
            name not in ("config", "index")
            and (not name.endswith(".safetensors") or name == ".safetensors")
            for name in checkpoint_identity
        )
    ):
        raise ValueError(
            "checkpoint identity needs verified config, index, and loaded shards"
        )
    if tokenizer_identity.keys() != TOKENIZER_ASSETS:
        raise ValueError("tokenizer identity must contain the exact loaded assets")
    fa_layers = [
        index
        for index, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    ]
    gdn_layers = [
        index
        for index, kind in enumerate(text["layer_types"])
        if kind == "linear_attention"
    ]
    if num_speculative_tokens:
        fa_layers.append(MODEL["layers"])
    descriptor = {
        "schema_version": 1,
        "architecture": SPEC.architecture,
        "model_type": SPEC.model_type,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "tokenizer_sha256": tokenizer_sha,
        "vocabulary_size": text["vocab_size"],
        "max_context_tokens": MODEL["context_tokens"],
        "capabilities": sorted(SPEC.capabilities),
        "weight_formats": ["bf16", "f32", "qwen_fp8_e4m3_block128x128_fp32_scales"],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                _group(
                    "fa",
                    "full_attention",
                    "qwen38_paired_fa_kv_bf16",
                    fa_layers,
                    fa_capacity_blocks,
                ),
                _group(
                    "gdn",
                    "recurrent",
                    "qwen38_paired_gdn_f32_conv_bf16_hidden_bf16",
                    gdn_layers,
                    gdn_capacity_blocks,
                ),
            ],
        },
        "parallel_plan": {
            "strategy": "tp1",
            "world_size": 1,
            "ranks": [0],
            "request_placement": "replicated",
        },
        "speculation": [
            {
                "method": "mtp",
                "predictor_stages": 1,
                "max_draft_tokens": 4,
                "max_verification_rows": 5,
            }
        ],
    }
    validate_descriptor(descriptor)
    return descriptor
