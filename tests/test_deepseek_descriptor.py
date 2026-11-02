import copy
import hashlib
import json
import unittest
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.adapter import (
    CONFIG_SHA256,
    SPEC,
    validate_config,
)
from oh_my_vllm.models.deepseek_v41.descriptor import build_descriptor
from oh_my_vllm.models.deepseek_v41.layout import tensor_layout
from oh_my_vllm.models.deepseek_v41.placement import rank_plan, weight_placements
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
    def test_shared_expert_values_and_scales_have_full_rank_placements(self):
        layout = tensor_layout(MODEL_CONFIG)
        shapes = {
            "w1.weight": (2304, 5120),
            "w1.scale": (72, 160),
            "w2.weight": (5120, 2304),
            "w2.scale": (160, 72),
            "w3.weight": (2304, 5120),
            "w3.scale": (72, 160),
        }
        for strategy in ("TP4", "TEP4", "DEP4"):
            placements = weight_placements(MODEL_CONFIG, layout, strategy)
            plans = [
                rank_plan(MODEL_CONFIG, layout, strategy, rank) for rank in range(4)
            ]
            for namespace, count in (("layers", 40), ("mtp", 3)):
                for layer in range(count):
                    for suffix, shape in shapes.items():
                        name = f"{namespace}.{layer}.ffn.shared_experts.{suffix}"
                        with self.subTest(strategy=strategy, name=name):
                            placement = placements[name]
                            self.assertEqual(placement["placement"], "replicated")
                            self.assertIsNone(placement["shard_axis"])
                            self.assertIsNone(placement["owner_rank"])
                            self.assertIsNone(placement["rank_shard_ranges"])
                            size = shape[0] * shape[1]
                            self.assertEqual(placement["rank_source_bytes"], [size] * 4)
                            self.assertEqual(
                                placement["rank_resident_bytes"], [size] * 4
                            )
                            for plan in plans:
                                selection = plan.selections[name]
                                self.assertEqual(tuple(selection.start), (0, 0))
                                self.assertEqual(tuple(selection.stop), shape)

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
                "window": 7,
                "compressed_kv": 17,
                "index_keys": 17,
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
                "capacity_blocks": 7,
                "storage": {
                    "type": "layout",
                    "id": "deepseek_v41_window_checkpoint_bf16",
                    "version": 1,
                },
                "layers": list(range(40)),
                "source": {"type": "owned"},
            },
        )
        self.assertEqual(groups["compressed_kv"]["layers"], [2, 8, 14, 20])
        self.assertEqual(groups["index_keys"]["layers"], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(groups["compressed_kv"]["capacity_blocks"], 17)
        self.assertEqual(groups["index_keys"]["capacity_blocks"], 17)
        self.assertEqual(groups["accepted_state"]["kind"], "model_checkpoint")
        self.assertEqual(groups["accepted_state"]["capacity_blocks"], 7)
        self.assertEqual(
            groups["accepted_state"]["storage"],
            {"type": "layout", "id": "deepseek_v41_accepted_state_v2", "version": 1},
        )

    def test_window_capacity_must_match_accepted_state_capacity(self):
        for window, accepted in ((17, 7), (7, 17)):
            with (
                self.subTest(window=window, accepted=accepted),
                self.assertRaisesRegex(ValueError, "window.*accepted"),
            ):
                build_descriptor(
                    MODEL_CONFIG,
                    raw_config=RAW_CONFIG,
                    checkpoint_identity=checkpoint_identity(),
                    tokenizer_identity=tokenizer_identity(),
                    strategy="tp4",
                    cache_capacities={
                        "window": window,
                        "compressed_kv": 17,
                        "index_keys": 17,
                        "accepted_state": accepted,
                    },
                )

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
                "window": 7,
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
                "window": 7,
                "compressed_kv": 11,
                "index_keys": 13,
                "accepted_state": 7,
            },
        )
        self.assertNotEqual(changed["checkpoint_sha256"], original["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
