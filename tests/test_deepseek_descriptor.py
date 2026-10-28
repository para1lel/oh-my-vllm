import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.adapter import (
    CONFIG_SHA256,
    SPEC,
    validate_config,
)
from oh_my_vllm.models.deepseek_v41.descriptor import (
    build_descriptor,
    checkpoint_metadata_identity,
)
from oh_my_vllm.models.descriptor import validate_descriptor
from oh_my_vllm.models.registry import select

ROOT = Path(__file__).parent
MODEL_CONFIG = json.loads(
    (ROOT / "fixtures/deepseek_v41/config.json").read_text(encoding="ascii")
)
RAW_CONFIG = (ROOT / "fixtures/deepseek_v41/config.json").read_bytes()


def checkpoint_identity():
    return {
        "config": CONFIG_SHA256,
        "index": "b" * 64,
    }


def tokenizer_identity():
    return {
        "tokenizer.json": "c" * 64,
        "tokenizer_config.json": "d" * 64,
        "chat_template.jinja": "e" * 64,
        "encoding.py": "f" * 64,
    }


class DeepSeekDescriptor(unittest.TestCase):
    def test_runtime_identity_reads_metadata_without_reading_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(RAW_CONFIG)
            index = {
                "weight_map": {
                    f"tensor.{index}": (f"model-{index + 1:05d}-of-00048.safetensors")
                    for index in range(48)
                }
            }
            raw_index = json.dumps(index, separators=(",", ":")).encode()
            (root / "model.safetensors.index.json").write_bytes(raw_index)

            identity = checkpoint_metadata_identity(root, raw_config=RAW_CONFIG)

        self.assertEqual(
            identity,
            {
                "config": CONFIG_SHA256,
                "index": hashlib.sha256(raw_index).hexdigest(),
            },
        )

    def test_config_validation_accepts_the_released_checkpoint_contract(self):
        validate_config(MODEL_CONFIG)
        self.assertEqual(hashlib.sha256(RAW_CONFIG).hexdigest(), CONFIG_SHA256)
        self.assertEqual(SPEC.architecture, "DeepseekV41ForCausalLM")
        self.assertEqual(SPEC.model_type, "deepseek_v41")
        self.assertEqual(SPEC.capabilities, frozenset(("text",)))
        self.assertEqual(select(MODEL_CONFIG).key, "deepseek_v41")

    def test_descriptor_binds_identity_cache_groups_parallelism_and_dspark5(self):
        descriptor = build_descriptor(
            MODEL_CONFIG,
            raw_config=RAW_CONFIG,
            checkpoint_identity=checkpoint_identity(),
            tokenizer_identity=tokenizer_identity(),
            strategy="dep4",
            cache_capacities={
                "window": 17,
                "compressed_kv": 11,
                "index_keys": 13,
                "accepted_state": 7,
            },
        )
        validate_descriptor(descriptor)
        self.assertEqual(
            descriptor["parallel_plan"],
            {
                "strategy": "dep4",
                "world_size": 4,
                "ranks": [0, 1, 2, 3],
                "request_placement": "rank_partitioned",
            },
        )
        self.assertEqual(descriptor["vocabulary_size"], 129280)
        self.assertEqual(descriptor["max_context_tokens"], 1048576)
        self.assertEqual(
            descriptor["weight_formats"],
            [
                "bf16",
                "f32",
                "deepseek_fp4",
                "deepseek_fp8_e4m3_block32_ue8m0",
            ],
        )
        self.assertEqual(
            descriptor["speculation"],
            [
                {
                    "method": "dspark5",
                    "predictor_stages": 3,
                    "max_draft_tokens": 5,
                    "max_verification_rows": 6,
                }
            ],
        )
        groups = {
            group["name"]: group for group in descriptor["cache_schema"]["groups"]
        }
        self.assertEqual(
            groups["window"],
            {
                "name": "window",
                "kind": "sliding_window",
                "block_tokens": 128,
                "capacity_blocks": 17,
                "storage": {
                    "type": "layout",
                    "id": "deepseek_v41_sliding_window_kv_bf16",
                    "version": 1,
                },
                "layers": list(range(40)),
                "source": {"type": "owned"},
            },
        )
        self.assertEqual(groups["compressed_kv"]["layers"], [2, 8, 14, 20])
        self.assertEqual(groups["index_keys"]["layers"], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(groups["accepted_state"]["kind"], "model_checkpoint")

    def test_config_and_loaded_identities_are_immutable_contract_inputs(self):
        config = copy.deepcopy(MODEL_CONFIG)
        config["text_config"]["num_nextn_predict_layers"] = 2
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_config(config)

        identity = checkpoint_identity()
        identity["index"] = "0" * 64
        original = build_descriptor(
            MODEL_CONFIG,
            raw_config=RAW_CONFIG,
            checkpoint_identity=checkpoint_identity(),
            tokenizer_identity=tokenizer_identity(),
            strategy="tp4",
            cache_capacities={
                "window": 17,
                "compressed_kv": 11,
                "index_keys": 13,
                "accepted_state": 7,
            },
        )
        changed = build_descriptor(
            MODEL_CONFIG,
            raw_config=RAW_CONFIG,
            checkpoint_identity=identity,
            tokenizer_identity=tokenizer_identity(),
            strategy="tp4",
            cache_capacities={
                "window": 17,
                "compressed_kv": 11,
                "index_keys": 13,
                "accepted_state": 7,
            },
        )
        self.assertNotEqual(changed["checkpoint_sha256"], original["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
