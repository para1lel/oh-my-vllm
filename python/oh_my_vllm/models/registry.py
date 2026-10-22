"""Select implemented model adapters before importing a GPU backend."""

import json
from pathlib import Path

from .api import AdapterSpec, ModelAdapter
from .qwen38.adapter import SPEC as QWEN38

ADAPTERS = (QWEN38,)
MAX_CONFIG_BYTES = 1 << 20


def select(config, *, required_capabilities=()) -> AdapterSpec:
    if not isinstance(config, dict):
        raise ValueError("model configuration must be an object")
    if not isinstance(required_capabilities, (tuple, list, set, frozenset)) or any(
        not isinstance(value, str) for value in required_capabilities
    ):
        raise ValueError("required capabilities must be a sequence of names")
    for adapter in ADAPTERS:
        if (
            config.get("architectures") == [adapter.architecture]
            and config.get("model_type") == adapter.model_type
        ):
            adapter.validate_config(config)
            if not set(required_capabilities).issubset(adapter.capabilities):
                raise ValueError(
                    "selected model does not support required capabilities"
                )
            return adapter
    raise ValueError("unsupported model architecture or model type")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model configuration key")
        result[key] = value
    return result


def create(config, *, required_capabilities=()) -> ModelAdapter:
    with (Path(config["model_path"]) / "config.json").open("rb") as source:
        raw = source.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("model configuration exceeds its byte limit")
    parsed = json.loads(raw, object_pairs_hook=_object)
    adapter = select(parsed, required_capabilities=required_capabilities)
    return adapter.build(config, raw)
