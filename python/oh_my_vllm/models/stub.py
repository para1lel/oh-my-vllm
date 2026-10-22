"""Metadata-only descriptor for the explicit service-test placeholder.

The f32 format is a wire sentinel. This backend loads no model weights and
simulates capabilities; it does not decode media or run neural inference.
"""

import hashlib
import json

from .descriptor import CAPABILITIES, validate_descriptor

CHECKPOINT_SHA256 = hashlib.sha256(b"oh-my-vllm/stub/checkpoint/v1").hexdigest()
TOKENIZER_SHA256 = hashlib.sha256(b"oh-my-vllm/stub/tokenizer/v1").hexdigest()


def build_descriptor(fa_capacity_blocks, gdn_capacity_blocks):
    for capacity in (fa_capacity_blocks, gdn_capacity_blocks):
        if type(capacity) is not int or not 2 <= capacity <= 0xFFFFFFFF:
            raise ValueError("stub cache capacity must include usable storage")
    if gdn_capacity_blocks > 65536:
        raise ValueError("stub GDN capacity exceeds its pool limit")
    model = {
        "schema_version": 1,
        "architecture": "OhMyVllmStubForTesting",
        "model_type": "oh_my_vllm_stub",
        "vocabulary_size": 248320,
        "max_context_tokens": 262144,
    }
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":")).encode("ascii")
    descriptor = {
        **model,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": hashlib.sha256(
            b"oh-my-vllm/stub/config/v1\0" + canonical
        ).hexdigest(),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "capabilities": sorted(CAPABILITIES),
        "weight_formats": ["f32"],
        "cache_schema": {
            "schema_version": 1,
            "groups": [
                {
                    "name": name,
                    "kind": "model_checkpoint",
                    "block_tokens": 784,
                    "capacity_blocks": capacity,
                    "storage": {
                        "type": "layout",
                        "id": f"oh_my_vllm_stub_{name}_metadata",
                        "version": 1,
                    },
                    "layers": [0],
                    "source": {"type": "owned"},
                }
                for name, capacity in (
                    ("fa", fa_capacity_blocks),
                    ("gdn", gdn_capacity_blocks),
                )
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
