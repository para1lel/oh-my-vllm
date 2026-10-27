"""DeepSeek-V4.1-Flash descriptor construction without GPU imports."""

import hashlib
import json
import re

from ...models.descriptor import identity_digest, validate_descriptor
from .adapter import (
    ARCHITECTURE,
    CONFIG_SHA256,
    DS_PARK_DRAFT_TOKENS,
    DS_PARK_STAGES,
    DS_PARK_VERIFICATION_ROWS,
    INDEX_SOURCE_LAYERS,
    KV_SOURCE_LAYERS,
    MAX_CONTEXT_TOKENS,
    MODEL_TYPE,
    NUM_LAYERS,
    SPEC,
    WINDOW_SIZE,
    validate_config,
)

MAX_CONFIG_BYTES = 1 << 20
TOKENIZER_ASSETS = frozenset(
    ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "encoding.py")
)
CHECKPOINT_SHARDS = frozenset(
    f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model configuration key")
        result[key] = value
    return result


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same(actual, item) for actual, item in zip(value, expected, strict=True)
        )
    return value == expected


def _verified_config(config, raw_config):
    if (
        type(raw_config) is not bytes
        or not 1 <= len(raw_config) <= MAX_CONFIG_BYTES
        or hashlib.sha256(raw_config).hexdigest() != CONFIG_SHA256
    ):
        raise ValueError(
            "DeepSeek configuration identity differs from the model contract"
        )
    parsed = json.loads(raw_config, object_pairs_hook=_object)
    if not _same(config, parsed):
        raise ValueError("configuration differs from the verified original bytes")
    validate_config(config)
    return config["text_config"]


def _verified_checkpoint_identity(identity):
    if type(identity) is not dict or identity.get("config") != CONFIG_SHA256:
        raise ValueError("checkpoint identity requires the verified DeepSeek config")
    expected = {"config", "index", *CHECKPOINT_SHARDS}
    if identity.keys() != expected:
        raise ValueError("checkpoint identity requires config, index, and all shards")
    for _name, digest in identity.items():
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("checkpoint identity contains an invalid digest")


def _verified_tokenizer_identity(identity):
    if type(identity) is not dict or identity.keys() != TOKENIZER_ASSETS:
        raise ValueError(
            "tokenizer identity must contain the DeepSeek tokenizer assets"
        )
    for digest in identity.values():
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("tokenizer identity contains an invalid digest")


def _group(name, kind, layout, layers, capacity):
    return {
        "name": name,
        "kind": kind,
        "block_tokens": WINDOW_SIZE,
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
    strategy,
    cache_capacities,
):
    """Build a descriptor from verified checkpoint and tokenizer identities."""
    text = _verified_config(config, raw_config)
    _verified_checkpoint_identity(checkpoint_identity)
    _verified_tokenizer_identity(tokenizer_identity)
    if strategy not in ("tp4", "tep4", "dep4"):
        raise ValueError("DeepSeek supports TP4, TEP4, or DEP4")
    if type(cache_capacities) is not dict or set(cache_capacities) != {
        "window",
        "compressed_kv",
        "index_keys",
        "accepted_state",
    }:
        raise ValueError("DeepSeek cache capacities must name all cache groups")
    if any(type(value) is not int or value < 1 for value in cache_capacities.values()):
        raise ValueError("DeepSeek cache capacities must be positive integers")

    descriptor = {
        "schema_version": 1,
        "architecture": ARCHITECTURE,
        "model_type": MODEL_TYPE,
        "checkpoint_sha256": identity_digest(checkpoint_identity, domain="checkpoint"),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "tokenizer_sha256": identity_digest(tokenizer_identity, domain="tokenizer"),
        "vocabulary_size": text["vocab_size"],
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "capabilities": sorted(SPEC.capabilities),
        "weight_formats": [
            "bf16",
            "f32",
            "deepseek_fp4",
            "deepseek_fp8_e4m3_block32_ue8m0",
        ],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                _group(
                    "window",
                    "sliding_window",
                    "deepseek_v41_sliding_window_kv_bf16",
                    list(range(NUM_LAYERS)),
                    cache_capacities["window"],
                ),
                _group(
                    "compressed_kv",
                    "compressed_kv",
                    "deepseek_v41_compressed_kv_source_bf16",
                    list(KV_SOURCE_LAYERS),
                    cache_capacities["compressed_kv"],
                ),
                _group(
                    "index_keys",
                    "index_keys",
                    "deepseek_v41_index_keys_bf16",
                    list(INDEX_SOURCE_LAYERS),
                    cache_capacities["index_keys"],
                ),
                _group(
                    "accepted_state",
                    "model_checkpoint",
                    "deepseek_v41_accepted_state_v1",
                    list(range(NUM_LAYERS)),
                    cache_capacities["accepted_state"],
                ),
            ],
        },
        "parallel_plan": {
            "strategy": strategy,
            "world_size": 4,
            "ranks": [0, 1, 2, 3],
            "request_placement": "rank_partitioned"
            if strategy == "dep4"
            else "replicated",
        },
        "speculation": [
            {
                "method": "dspark5",
                "predictor_stages": DS_PARK_STAGES,
                "max_draft_tokens": DS_PARK_DRAFT_TOKENS,
                "max_verification_rows": DS_PARK_VERIFICATION_ROWS,
            }
        ],
    }
    validate_descriptor(descriptor)
    return descriptor
