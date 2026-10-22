import hashlib

from ...operators.specification import MODEL
from ..api import AdapterSpec


def validate_config(config):
    quantization = config.get("quantization_config")
    expected = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    }
    if not isinstance(quantization, dict) or any(
        quantization.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("unsupported Qwen weight or activation format")


def build(config, raw_config):
    if hashlib.sha256(raw_config).hexdigest() != MODEL["config_sha256"]:
        raise ValueError("checkpoint configuration differs from the model contract")
    from ...worker.inference import Worker

    return Worker(config, raw_config=raw_config)


SPEC = AdapterSpec(
    key="qwen38",
    architecture="Qwen3_5ForConditionalGeneration",
    model_type="qwen3_5",
    capabilities=frozenset(
        (
            "text",
            "images",
            "video",
            "tools",
            "reasoning",
            "structured_outputs",
            "prefix_cache",
            "host_offload",
        )
    ),
    validate_config=validate_config,
    build=build,
)
