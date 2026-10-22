"""CPU-only validation of the model-contract v1 named-map wire format."""

import hashlib
import json
import re

MODEL_DESCRIPTOR_VERSION = 1
CACHE_SCHEMA_VERSION = 1
MAX_IDENTIFIER_BYTES = 128
MAX_CACHE_GROUPS = 256
MAX_LAYER_REFERENCES = 4096
MAX_IDENTITY_FILES = 4096

CAPABILITIES = frozenset(
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
)
WEIGHT_FORMATS = frozenset(
    (
        "bf16",
        "f32",
        "qwen_fp8_e4m3_block128x128_fp32_scales",
        "deepseek_fp8_e4m3_block32x32_ue8m0_scales",
        "routed_fp4_group32_ue8m0_scales",
    )
)
CACHE_KINDS = frozenset(
    (
        "full_attention",
        "recurrent",
        "sliding_window",
        "compressed_kv",
        "index_keys",
        "model_checkpoint",
    )
)
IDENTITY_DOMAINS = {
    "checkpoint": b"oh-my-vllm/checkpoint-identity/v1\0",
    "tokenizer": b"oh-my-vllm/tokenizer-identity/v1\0",
}


def _require(valid, field):
    if not valid:
        raise ValueError(f"invalid {field}")


def _object(value, fields, field):
    _require(type(value) is dict and value.keys() == set(fields.split()), field)


def _uint(value, bits, field, minimum=0):
    _require(type(value) is int and minimum <= value < 1 << bits, field)


def _identifier(value, field):
    _require(
        type(value) is str
        and 1 <= len(value) <= MAX_IDENTIFIER_BYTES
        and re.fullmatch(r"[A-Za-z0-9_.-]+", value) is not None,
        field,
    )


def _sha256(value, field):
    _require(
        type(value) is str
        and len(value) == 64
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        field,
    )


def _list(value, limit, field, minimum=0):
    _require(type(value) is list and minimum <= len(value) <= limit, field)


def _enum(value, choices, field):
    _require(type(value) is str and value in choices, field)


def _enum_list(value, choices, field, minimum=0):
    _list(value, len(choices), field, minimum)
    for item in value:
        _enum(item, choices, field)
    _require(len(set(value)) == len(value), field)


def identity_digest(identity, *, domain):
    """Hash a bounded file-name -> lowercase SHA256 map.

    The preimage is IDENTITY_DOMAINS[domain] followed by ASCII JSON, with sorted
    keys, separators (",", ":"), ensure_ascii=True, and allow_nan=False.
    Names use 1..256 ASCII letters, digits, dots, underscores, or hyphens.
    The entire supplied map is included; no path or file is read.
    """
    _enum(domain, IDENTITY_DOMAINS, "identity domain")
    _require(
        type(identity) is dict and 1 <= len(identity) <= MAX_IDENTITY_FILES,
        "identity files",
    )
    identity = identity.copy()
    for name, digest in identity.items():
        _require(
            type(name) is str
            and 1 <= len(name) <= 256
            and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None,
            "identity file name",
        )
        _sha256(digest, "identity file SHA256")
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(IDENTITY_DOMAINS[domain] + canonical).hexdigest()


def _storage(storage):
    _require(type(storage) is dict, "cache_group.storage")
    kind = storage.get("type")
    _enum(kind, ("primitive", "layout"), "cache_group.storage.type")
    if kind == "primitive":
        _object(storage, "type format", "cache_group.storage")
        _enum(storage["format"], ("bf16", "f32"), "cache_group.storage.format")
    else:
        _object(storage, "type id version", "cache_group.storage")
        _identifier(storage["id"], "cache_group.storage.id")
        _uint(storage["version"], 32, "cache_group.storage.version", 1)


def _source(source):
    _require(type(source) is dict, "cache_group.source")
    kind = source.get("type")
    _enum(kind, ("owned", "shared"), "cache_group.source.type")
    if kind == "owned":
        _object(source, "type", "cache_group.source")
    else:
        _object(source, "type producer", "cache_group.source")
        _identifier(source["producer"], "cache_group.source.producer")


def _cache_schema(schema):
    _object(schema, "schema_version groups", "cache_schema")
    _uint(schema["schema_version"], 32, "cache_schema.schema_version")
    _require(
        schema["schema_version"] == CACHE_SCHEMA_VERSION, "cache_schema.schema_version"
    )
    groups = schema["groups"]
    _list(groups, MAX_CACHE_GROUPS, "cache_schema.groups", 1)
    names = {}
    layer_count = 0
    for group in groups:
        _object(
            group,
            "name kind block_tokens capacity_blocks storage layers source",
            "cache_group",
        )
        name = group["name"]
        _identifier(name, "cache_group.name")
        _require(name not in names, "cache_schema.groups duplicate name")
        names[name] = group
        _enum(group["kind"], CACHE_KINDS, "cache_group.kind")
        _uint(group["block_tokens"], 32, "cache_group.block_tokens", 1)
        _uint(group["capacity_blocks"], 64, "cache_group.capacity_blocks", 1)
        layers = group["layers"]
        _list(layers, MAX_LAYER_REFERENCES, "cache_group.layers", 1)
        layer_count += len(layers)
        _require(layer_count <= MAX_LAYER_REFERENCES, "cache_schema.groups.layers")
        for layer in layers:
            _uint(layer, 32, "cache_group.layers")
        _require(len(set(layers)) == len(layers), "cache_group.layers duplicate")
        _storage(group["storage"])
        _source(group["source"])
    for group in groups:
        current = group
        for _ in range(len(groups)):
            source = current["source"]
            if source["type"] == "owned":
                break
            _require(source["producer"] in names, "cache_group.source unknown producer")
            current = names[source["producer"]]
        else:
            raise ValueError("invalid cache_group.source cyclic producer reference")


def _parallel_plan(plan):
    _object(plan, "strategy world_size ranks request_placement", "parallel_plan")
    strategy = plan["strategy"]
    _enum(strategy, ("tp1", "tp4", "tep4", "dep4"), "parallel_plan.strategy")
    expected_size = 1 if strategy == "tp1" else 4
    _uint(plan["world_size"], 32, "parallel_plan.world_size")
    _require(plan["world_size"] == expected_size, "parallel_plan.world_size")
    ranks = plan["ranks"]
    _list(ranks, 4, "parallel_plan.ranks")
    for rank in ranks:
        _uint(rank, 32, "parallel_plan.ranks")
    _require(
        len(ranks) == expected_size and set(ranks) == set(range(expected_size)),
        "parallel_plan.ranks",
    )
    placement = "rank_partitioned" if strategy == "dep4" else "replicated"
    _require(plan["request_placement"] == placement, "parallel_plan.request_placement")


def _speculation(specs):
    _list(specs, 2, "speculation")
    methods = set()
    for spec in specs:
        _object(
            spec,
            "method predictor_stages max_draft_tokens max_verification_rows",
            "speculation",
        )
        method = spec["method"]
        _enum(method, ("mtp", "dspark"), "speculation.method")
        _require(method not in methods, "speculation duplicate method")
        methods.add(method)
        for field in ("predictor_stages", "max_draft_tokens", "max_verification_rows"):
            _uint(spec[field], 32, "speculation." + field, 1)
        _require(
            spec["max_verification_rows"] == spec["max_draft_tokens"] + 1,
            "speculation.max_verification_rows",
        )


def validate_descriptor(descriptor):
    """Validate wire structure, not family support or file contents.

    A transport must bound its input bytes before JSON/msgpack decoding.
    Layout IDs and family-specific dimensions require the selected adapter.
    """
    _object(
        descriptor,
        "schema_version architecture model_type checkpoint_sha256 config_sha256 "
        "tokenizer_sha256 vocabulary_size max_context_tokens capabilities "
        "weight_formats cache_schema parallel_plan speculation",
        "descriptor",
    )
    _uint(descriptor["schema_version"], 32, "schema_version")
    _require(descriptor["schema_version"] == MODEL_DESCRIPTOR_VERSION, "schema_version")
    for field in ("architecture", "model_type"):
        _identifier(descriptor[field], field)
    for field in ("checkpoint_sha256", "config_sha256", "tokenizer_sha256"):
        _sha256(descriptor[field], field)
    _uint(descriptor["vocabulary_size"], 32, "vocabulary_size", 1)
    _uint(descriptor["max_context_tokens"], 64, "max_context_tokens", 1)
    _enum_list(descriptor["capabilities"], CAPABILITIES, "capabilities")
    _enum_list(descriptor["weight_formats"], WEIGHT_FORMATS, "weight_formats", 1)
    _cache_schema(descriptor["cache_schema"])
    _parallel_plan(descriptor["parallel_plan"])
    _speculation(descriptor["speculation"])
