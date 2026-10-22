import copy
import json
import tempfile
import unittest
from pathlib import Path

from oh_my_vllm.models import registry
from oh_my_vllm.worker.zmq_bridge import create_worker


def qwen_config():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }


class ModelRegistry(unittest.TestCase):
    def test_selection_uses_architecture_formats_and_capabilities(self):
        source = qwen_config()
        original = copy.deepcopy(source)
        selected = registry.select(source, required_capabilities=("text", "video"))
        self.assertEqual(selected.key, "qwen38")
        self.assertEqual(source, original)
        for field, value in (
            ("architectures", ["UnknownModel"]),
            ("architectures", ["Qwen3_5ForConditionalGeneration", "UnknownModel"]),
            ("architectures", "Qwen3_5ForConditionalGeneration"),
            ("model_type", "deepseek_v41"),
            ("quantization_config", {"quant_method": "awq"}),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                registry.select(dict(source, **{field: value}))
        for field, value in (
            ("activation_scheme", "static"),
            ("fmt", "e5m2"),
            ("weight_block_size", [32, 32]),
        ):
            changed = copy.deepcopy(source)
            changed["quantization_config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                registry.select(changed)
        with self.assertRaises(ValueError):
            registry.select(source, required_capabilities=("audio",))
        with self.assertRaises(ValueError):
            registry.select(source, required_capabilities="text")
        with self.assertRaises(ValueError):
            registry.select(
                {
                    "architectures": ["DeepseekV41ForCausalLM"],
                    "model_type": "deepseek_v41",
                }
            )

    def test_bridge_rejects_unverified_configuration_before_backend_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "misleading-DeepSeek-name"
            root.mkdir()
            path = root / "config.json"
            path.write_text(json.dumps(qwen_config()))
            with self.assertRaisesRegex(ValueError, "configuration.*contract"):
                create_worker({"model_path": str(root)})
            path.write_text('{"model_type":"qwen3_5","model_type":"other"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                create_worker({"model_path": str(root)})
            path.write_text(" " * ((1 << 20) + 1))
            with self.assertRaisesRegex(ValueError, "configuration.*limit"):
                create_worker({"model_path": str(root)})
            path.write_text("[]")
            with self.assertRaises(ValueError):
                create_worker({"model_path": str(root)})


if __name__ == "__main__":
    unittest.main()
