import copy
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import msgpack
from oh_my_vllm.models.descriptor import identity_digest, validate_descriptor
from oh_my_vllm.models.qwen38.descriptor import build_descriptor
from oh_my_vllm.operators.specification import MODEL


def sample_inputs():
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
        "text_config": {
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "num_hidden_layers": 64,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "mtp_num_hidden_layers": 1,
        },
    }
    config["text_config"]["layer_types"] *= 16
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("ascii")
    checkpoint = {
        "config": hashlib.sha256(raw).hexdigest(),
        "index": "b" * 64,
        "layers-0.safetensors": "c" * 64,
    }
    tokenizer = {
        "tokenizer.json": "d" * 64,
        "generation_config.json": "e" * 64,
        "chat_template.jinja": "f" * 64,
    }
    return dict(
        config=config,
        raw_config=raw,
        checkpoint_identity=checkpoint,
        tokenizer_identity=tokenizer,
        fa_capacity_blocks=1608,
        gdn_capacity_blocks=128,
        num_speculative_tokens=4,
    )


def sample_descriptor(**changes):
    """Return a synthetic wire sample; no checkpoint files or GPU are needed."""
    inputs = sample_inputs()
    inputs.update(changes)
    with patch.dict(
        MODEL, config_sha256=sample_inputs()["checkpoint_identity"]["config"]
    ):
        return build_descriptor(**inputs)


class ModelDescriptor(unittest.TestCase):
    def test_identity_digest_binds_exact_named_payload_and_domain(self):
        identity = {"z.safetensors": "a" * 64, "config": "b" * 64}
        canonical = (
            '{"config":"' + "b" * 64 + '","z.safetensors":"' + "a" * 64 + '"}'
        ).encode("ascii")
        expected = hashlib.sha256(
            b"oh-my-vllm/checkpoint-identity/v1\0" + canonical
        ).hexdigest()
        self.assertEqual(identity_digest(identity, domain="checkpoint"), expected)
        self.assertEqual(
            identity_digest(
                dict(reversed(list(identity.items()))), domain="checkpoint"
            ),
            expected,
        )
        self.assertNotEqual(identity_digest(identity, domain="tokenizer"), expected)
        for changed in (
            dict(identity, config="c" * 64),
            {"renamed.safetensors": "a" * 64, "config": "b" * 64},
            dict(identity, **{"extra.safetensors": "c" * 64}),
        ):
            self.assertNotEqual(identity_digest(changed, domain="checkpoint"), expected)
        for invalid in (
            {},
            {"config": True},
            {"config": "A" * 64},
            {"x" * 257: "a" * 64},
        ):
            with self.subTest(identity=invalid), self.assertRaises(ValueError):
                identity_digest(invalid, domain="checkpoint")
        with self.assertRaises(ValueError):
            identity_digest(identity, domain="unknown")

    def test_qwen_sample_binds_capacity_without_changing_loaded_identity(self):
        inputs = sample_inputs()
        descriptor = sample_descriptor(**inputs)
        validate_descriptor(descriptor)
        self.assertEqual(json.loads(json.dumps(descriptor)), descriptor)
        self.assertEqual(
            msgpack.unpackb(msgpack.packb(descriptor, use_bin_type=True), raw=False),
            descriptor,
        )
        self.assertEqual(
            descriptor["config_sha256"], inputs["checkpoint_identity"]["config"]
        )
        self.assertEqual(
            descriptor["checkpoint_sha256"],
            identity_digest(inputs["checkpoint_identity"], domain="checkpoint"),
        )
        self.assertEqual(
            descriptor["tokenizer_sha256"],
            identity_digest(inputs["tokenizer_identity"], domain="tokenizer"),
        )
        self.assertEqual(descriptor["max_context_tokens"], 262144)
        self.assertEqual(descriptor["vocabulary_size"], 248320)
        self.assertEqual(
            descriptor["parallel_plan"],
            dict(
                strategy="tp1", world_size=1, ranks=[0], request_placement="replicated"
            ),
        )
        self.assertEqual(
            descriptor["speculation"],
            [
                dict(
                    method="mtp",
                    predictor_stages=1,
                    max_draft_tokens=4,
                    max_verification_rows=5,
                )
            ],
        )
        fa, gdn = descriptor["cache_schema"]["groups"]
        self.assertEqual(fa["layers"], [*range(3, 64, 4), 64])
        self.assertEqual(gdn["layers"], [n for n in range(64) if n % 4 != 3])
        self.assertEqual(
            fa["storage"], dict(type="layout", id="qwen38_paired_fa_kv_bf16", version=1)
        )
        self.assertEqual(
            gdn["storage"],
            dict(
                type="layout",
                id="qwen38_paired_gdn_f32_conv_bf16_hidden_bf16",
                version=1,
            ),
        )
        self.assertEqual([fa["capacity_blocks"], gdn["capacity_blocks"]], [1608, 128])
        for group in (fa, gdn):
            self.assertEqual(group["block_tokens"], 784)
            self.assertEqual(group["source"], {"type": "owned"})
        resized = sample_descriptor(fa_capacity_blocks=42, gdn_capacity_blocks=17)
        self.assertEqual(
            [g["capacity_blocks"] for g in resized["cache_schema"]["groups"]], [42, 17]
        )
        for key in ("checkpoint_sha256", "tokenizer_sha256", "config_sha256"):
            self.assertEqual(resized[key], descriptor[key])
        ordinary = sample_descriptor(num_speculative_tokens=0)
        self.assertEqual(
            ordinary["cache_schema"]["groups"][0]["layers"], list(range(3, 64, 4))
        )
        changed = dict(
            inputs["checkpoint_identity"], **{"loaded-media.safetensors": "a" * 64}
        )
        self.assertNotEqual(
            sample_descriptor(checkpoint_identity=changed)["checkpoint_sha256"],
            descriptor["checkpoint_sha256"],
        )
        descriptor["cache_schema"]["groups"][0]["layers"].clear()
        self.assertEqual(inputs, sample_inputs())

    def test_qwen_rejects_unbound_configuration_identities_and_capacities(self):
        inputs = sample_inputs()
        invalid = [
            {"raw_config": inputs["raw_config"] + b" "},
            {"raw_config": b"x" * ((1 << 20) + 1)},
            {"raw_config": inputs["raw_config"].decode()},
            {
                "checkpoint_identity": {
                    "config": "a" * 64,
                    "index": "b" * 64,
                    "x.safetensors": "c" * 64,
                }
            },
            {
                "checkpoint_identity": {
                    "config": inputs["checkpoint_identity"]["config"],
                    "index": "b" * 64,
                }
            },
            {
                "checkpoint_identity": dict(
                    inputs["checkpoint_identity"], **{"not-a-shard": "c" * 64}
                )
            },
            {"tokenizer_identity": {"tokenizer.json": "d" * 64}},
            {
                "tokenizer_identity": dict(
                    inputs["tokenizer_identity"], **{"unused.json": "c" * 64}
                )
            },
        ]
        for name in ("fa_capacity_blocks", "gdn_capacity_blocks"):
            invalid.extend({name: value} for value in (None, True, 1, 2.0, -1, 1 << 32))
        invalid.extend(
            {"num_speculative_tokens": value} for value in (True, 1, 3, 5, 4.0)
        )
        for section, field, value in (
            (None, "model_type", "deepseek_v41"),
            ("text_config", "vocab_size", 248319),
            ("text_config", "mtp_num_hidden_layers", True),
            ("text_config", "max_position_embeddings", 1048576),
        ):
            changed = copy.deepcopy(inputs["config"])
            (changed if section is None else changed[section])[field] = value
            invalid.append({"config": changed})
        for changed in invalid:
            with self.subTest(changed=changed.keys()), self.assertRaises(ValueError):
                sample_descriptor(**changed)
        with self.assertRaisesRegex(ValueError, "configuration"):
            build_descriptor(**inputs)

    def test_wire_validator_rejects_bounded_structural_and_type_errors(self):
        descriptor = sample_descriptor()
        cases = [
            (("schema_version",), True),
            (("schema_version",), 2),
            (("architecture",), "x" * 129),
            (("model_type",), "bad/path"),
            (("checkpoint_sha256",), "A" * 64),
            (("vocabulary_size",), 1 << 32),
            (("vocabulary_size",), False),
            (("max_context_tokens",), 1 << 64),
            (("max_context_tokens",), 0),
            (("capabilities",), ["text", "text"]),
            (("capabilities",), ["audio"]),
            (("weight_formats",), []),
            (("weight_formats",), ["fp8"]),
            (("cache_schema", "schema_version"), 2),
            (("cache_schema", "groups"), []),
            (("cache_schema", "groups"), descriptor["cache_schema"]["groups"] * 129),
            (("cache_schema", "groups", 0, "block_tokens"), True),
            (("cache_schema", "groups", 0, "capacity_blocks"), 0),
            (
                ("cache_schema", "groups", 0, "storage"),
                {"type": "primitive", "format": "fp8"},
            ),
            (("cache_schema", "groups", 0, "storage", "version"), 0),
            (("cache_schema", "groups", 0, "layers"), [False]),
            (("cache_schema", "groups", 0, "layers"), [0, 0]),
            (("cache_schema", "groups", 0, "layers"), list(range(4097))),
            (("cache_schema", "groups", 1, "name"), "fa"),
            (
                ("cache_schema", "groups", 0, "source"),
                {"type": "shared", "producer": "missing"},
            ),
            (
                ("cache_schema", "groups", 0, "source"),
                {"type": "shared", "producer": "fa"},
            ),
            (("parallel_plan", "world_size"), True),
            (("parallel_plan", "ranks"), [False]),
            (("parallel_plan", "strategy"), "tp16"),
            (("parallel_plan", "request_placement"), "rank_partitioned"),
            (("speculation", 0, "method"), "unknown"),
            (("speculation", 0, "predictor_stages"), 0),
            (("speculation", 0, "max_draft_tokens"), (1 << 32) - 1),
            (("speculation", 0, "max_verification_rows"), 4),
        ]
        for path, value in cases:
            changed = copy.deepcopy(descriptor)
            parent = changed
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_descriptor(changed)
        for path in (
            (),
            ("cache_schema",),
            ("cache_schema", "groups", 0),
            ("cache_schema", "groups", 0, "source"),
            ("cache_schema", "groups", 0, "storage"),
            ("parallel_plan",),
            ("speculation", 0),
        ):
            changed = copy.deepcopy(descriptor)
            parent = changed
            for key in path:
                parent = parent[key]
            parent["unknown"] = True
            with self.subTest(unknown=path), self.assertRaises(ValueError):
                validate_descriptor(changed)
        groups = descriptor["cache_schema"]["groups"]
        groups[0]["source"] = {"type": "shared", "producer": "gdn"}
        groups[1]["source"] = {"type": "shared", "producer": "fa"}
        with self.assertRaises(ValueError):
            validate_descriptor(descriptor)

    def test_construction_has_no_gpu_or_worker_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from tests.test_model_descriptor import sample_descriptor; "
                "sample_descriptor(); "
                "assert not any(n.startswith(('torch', 'cupy', 'oh_my_vllm.worker', "
                "'oh_my_vllm.models.qwen38.inference', "
                "'oh_my_vllm.models.qwen38.weights', "
                "'oh_my_vllm.operators.runtime.device')) for n in sys.modules)",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
