"""Metadata-only descriptor for the explicit service-test placeholder.

The f32 format is a wire sentinel. This backend loads no model weights and
simulates capabilities; it does not decode media or run neural inference.
"""

import hashlib
import json

from ..worker.offload import completion, identifier, validate_operation
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


class StubOffload:
    def __init__(self, config, requests):
        limit = config.get("host_cache_bytes")
        self.limit = identifier(limit, "host_cache_bytes") if limit is not None else 0
        self.requests, self.config = requests, config
        self.snapshots, self.suspended, self.seen = {}, set(), set()
        self.used_bytes = self.allocated_bytes = 0
        self.store_bytes = self.load_bytes = self.completed_transfers = 0

    def stats(self):
        return {
            "host_cache_capacity_bytes": self.limit,
            "host_cache_allocated_bytes": self.allocated_bytes,
            "host_cache_used_bytes": self.used_bytes,
            "host_cache_snapshots": len(self.snapshots),
            "host_cache_pending_transfers": 0,
            "host_cache_store_bytes": self.store_bytes,
            "host_cache_load_bytes": self.load_bytes,
            "host_cache_completed_transfers": self.completed_transfers,
        }

    def check_execute(self, request_ids):
        if self.suspended.intersection(request_ids):
            raise ValueError("request is not GPU resident during offload")

    def check_register(self, request_id):
        if request_id in self.requests or request_id in self.suspended:
            raise ValueError("request identifier is already owned by the worker")

    def cancel(self, request_id):
        self.suspended.discard(request_id)

    def handle(self, message):
        from .qwen38.cache_budget import cache_layout

        layout, completed = cache_layout(self.config), []
        endpoint = layout.gdn_bytes // layout.slots
        for operation in message["operations"]:
            validate_operation(operation)
            transfer_id, snapshot_id = (
                operation["transfer_id"],
                operation["snapshot_id"],
            )
            if transfer_id in self.seen:
                raise ValueError("cache transfer identifier repeated")
            self.seen.add(transfer_id)
            request_id = operation["request_id"]
            counts = (len(operation["fa_blocks"]), len(operation["mamba_blocks"]))
            for blocks, limit in (
                (operation["fa_blocks"], self.config["num_gpu_blocks"]),
                (operation["mamba_blocks"], layout.mamba_blocks),
            ):
                if any(block >= limit for block in blocks):
                    raise ValueError("invalid cache transfer block")
            if operation["kind"] == "store":
                if snapshot_id in self.snapshots:
                    raise ValueError("cache snapshot identifier repeated")
                self.check_execute([] if request_id is None else [request_id])
                owner = self.requests[request_id] if request_id is not None else None
                size = (
                    counts[0] * layout.fa_block_bytes
                    + (counts[1] + (owner is not None)) * endpoint
                )
                if not self.limit or self.used_bytes + size > self.limit:
                    completed.append(completion(transfer_id, "capacity", size))
                    continue
                self.snapshots[snapshot_id] = (request_id, owner, counts, size)
                self.used_bytes += size
                self.allocated_bytes = max(self.allocated_bytes, self.used_bytes)
                self.store_bytes += size
                self.completed_transfers += 1
                if owner is not None:
                    self.suspended.add(request_id)
            else:
                saved_id, owner, saved_counts, size = self.snapshots[snapshot_id]
                if operation["kind"] == "load":
                    if request_id != saved_id or counts != saved_counts:
                        raise ValueError("cache load does not match snapshot")
                    if owner is not None and self.requests.get(saved_id) is not owner:
                        raise ValueError("cache request is cancelled")
                    self.suspended.discard(saved_id)
                    self.load_bytes += size
                    self.completed_transfers += 1
                else:
                    self.used_bytes -= size
                    del self.snapshots[snapshot_id]
                    if owner is None or self.requests.get(saved_id) is not owner:
                        self.suspended.discard(saved_id)
            completed.append(completion(transfer_id, "complete", size))
        return {
            "type": "cache_result",
            "rpc_id": message["rpc_id"],
            "completed": completed,
            "host_cache_bytes": self.limit,
        }
