"""Separate client tool descriptions from the existing XML decoding schema."""

from __future__ import annotations

import copy

# Keep the existing typed XML structure and supported argument constraints.
# Other descriptive keywords stay in the model prompt and are checked by the
# tool owner. Explicit strict=true uses the original schema without projection.
_DECODING_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "$defs",
    "$ref",
    "description",
    "title",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
}


def decoding_schema(schema):
    """Copy schema nodes without stripping names or data inside enum/const."""
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result = {}
    for key, value in schema.items():
        if key not in _DECODING_KEYS:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            value = {name: decoding_schema(child) for name, child in value.items()}
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            value = decoding_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            value = [decoding_schema(child) for child in value]
        else:
            value = copy.deepcopy(value)
        result[key] = value
    return result


def decoding_tools(tools: list[dict]) -> list[dict]:
    """Preserve caller-owned descriptions and keep strict validation unchanged."""
    result = copy.deepcopy(tools)
    for tool in result:
        function = tool["function"]
        schema = function.get(
            "parameters",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        function["parameters"] = (
            schema if function.get("strict") is True else decoding_schema(schema)
        )
    return result
