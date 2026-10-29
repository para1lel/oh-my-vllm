"""Checkpoint-bound CPU weight placement; payload authentication stays in the reader."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .checkpoint import DTYPE_BYTES, TensorSlice

CONFIG_SHA256 = "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
LAYOUT_SHA256 = "f0e72b086805a24c237ea3aa23420023ff634fbf4faa5ba9554491683eedc50f"
UINT64_MAX = 2**64 - 1


def _uint(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer byte size or count")
    if value > UINT64_MAX:
        raise OverflowError("Unsigned 64-bit size overflow")
    return value


def _size(*factors):
    value = 1
    for factor in factors:
        value = _uint(value * _uint(factor))
    return value


def _rule(rules, name, placement="replicated", axis=None, owner=None, dtype=None):
    if name in rules:
        raise ValueError(f"Duplicate placement rule: {name}")
    rules[name] = {
        "placement": placement,
        "shard_axis": axis,
        "owner_rank": owner,
        "promoted_dtype": dtype,
    }


def _linear_rules(
    rules, name, axis=None, owner=None, quant=None, dtype=None, bias=False
):
    placement = (
        "owned"
        if owner is not None
        else "sharded"
        if axis is not None
        else "replicated"
    )
    _rule(rules, f"{name}.weight", placement, axis, owner, dtype)
    if quant:
        scale_placement = "consumed_on_load" if dtype else placement
        _rule(rules, f"{name}.scale", scale_placement, axis, owner)
    if bias:
        _rule(rules, f"{name}.bias", placement, axis, owner)


def _attention_rules(rules, name, text, layer, dense_tp):
    output_axis = 0 if dense_tp else None
    input_axis = 1 if dense_tp else None
    _rule(
        rules, f"{name}.attn_sink", "sharded" if dense_tp else "replicated", output_axis
    )
    for norm in ("q_norm", "kv_norm"):
        _rule(rules, f"{name}.{norm}.weight")
    _linear_rules(rules, f"{name}.wq_a", quant="fp8")
    _linear_rules(rules, f"{name}.wq_b", output_axis, quant="fp8")
    _linear_rules(rules, f"{name}.wkv", quant="fp8")
    _linear_rules(rules, f"{name}.wo_a", output_axis, quant="fp8", dtype="BF16")
    _linear_rules(rules, f"{name}.wo_b", input_axis, quant="fp8")
    if layer in text["kv_source_layer_ids"]:
        compressor = f"{name}.compressor"
        _rule(rules, f"{compressor}.norm.weight")
        dtype = "F32" if text["compress_ratios"][layer] > 1 else None
        _linear_rules(rules, f"{compressor}.wkv", dtype=dtype)
        if dtype:
            _linear_rules(rules, f"{compressor}.wgate", dtype=dtype)
    if layer in text["index_source_layer_ids"]:
        indexer = f"{name}.indexer"
        _linear_rules(rules, f"{indexer}.wq_b", output_axis, quant="fp8")
        _linear_rules(rules, f"{indexer}.weights_proj", output_axis)
        if layer in text["kv_source_layer_ids"]:
            _linear_rules(rules, f"{indexer}.wk")
            _rule(rules, f"{indexer}.k_norm.weight")


def _block_rules(rules, name, text, layer, experts, strategy):
    dense_tp = strategy != "DEP4"
    for branch in ("attn", "ffn"):
        for suffix in (
            f"{branch}_norm.weight",
            f"hc_{branch}_fn",
            f"hc_{branch}_base",
            f"hc_{branch}_scale",
        ):
            _rule(rules, f"{name}.{suffix}")
    _attention_rules(rules, f"{name}.attn", text, layer, dense_tp)
    _linear_rules(rules, f"{name}.ffn.gate")
    for bias in ("bias", "bias_vl"):
        _rule(rules, f"{name}.ffn.gate.{bias}")
    for expert in range(experts):
        owner = None if strategy == "TP4" else expert // (experts // 4)
        for projection in ("w1", "w2", "w3"):
            axis = (1 if projection == "w2" else 0) if strategy == "TP4" else None
            _linear_rules(
                rules,
                f"{name}.ffn.experts.{expert}.{projection}",
                axis,
                owner,
                quant="fp4",
            )
    # Released MP4 keeps full shared experts on each rank, outside routed reduction.
    for projection in ("w1", "w2", "w3"):
        _linear_rules(rules, f"{name}.ffn.shared_experts.{projection}", quant="fp8")


def _placement_rules(model, strategy):
    text = model["text_config"]
    dense_axis = 0 if strategy != "DEP4" else None
    rules = {}
    _linear_rules(rules, "embed", dense_axis)
    _linear_rules(rules, "head", dense_axis, dtype="F32")
    _rule(rules, "norm.weight")
    for layer in range(text["num_hidden_layers"]):
        _block_rules(
            rules, f"layers.{layer}", text, layer, text["n_routed_experts"], strategy
        )
    for layer in text["engram_layer_ids"]:
        name = f"layers.{layer}.engram"
        for suffix in ("weight", "scale"):
            _rule(rules, f"{name}.embed.{suffix}", "shared_pinned_host")
        for weight in ("q_weight", "k_weight"):
            _rule(rules, f"{name}.{weight}")
        _linear_rules(rules, f"{name}.wkv", quant="fp8")
    _linear_rules(rules, "vision.patch_embed.proj", owner=0, bias=True)
    for layer in range(model["vision_config"]["num_hidden_layers"]):
        name = f"vision.blocks.{layer}"
        for norm in ("norm1", "norm2"):
            _rule(rules, f"{name}.{norm}.weight", "owned", owner=0)
        for projection in ("wqkv", "wo"):
            _linear_rules(rules, f"{name}.attn.{projection}", owner=0, bias=True)
        for projection in ("w1", "w2"):
            _linear_rules(rules, f"{name}.mlp.{projection}", owner=0)
    _rule(rules, "vision.norm.weight", "owned", owner=0)
    for projection in ("w1", "w2"):
        _linear_rules(rules, f"aligner.{projection}", owner=0, bias=True)
    for name in ("image_start", "image_end", "image_newline"):
        _rule(rules, name, "owned", owner=0)
    stages = text["num_nextn_predict_layers"]
    for stage in range(stages):
        _block_rules(
            rules,
            f"mtp.{stage}",
            text,
            text["num_hidden_layers"] + stage,
            text["dspark_n_routed_experts"],
            strategy,
        )
    _linear_rules(rules, "mtp.0.main_proj", quant="fp8")
    _rule(rules, "mtp.0.main_norm.weight")
    last = f"mtp.{stages - 1}"
    _rule(rules, f"{last}.norm.weight")
    _linear_rules(rules, f"{last}.markov_head.embed", dense_axis)
    _linear_rules(rules, f"{last}.markov_head.head", dense_axis, dtype="F32")
    _linear_rules(rules, f"{last}.confidence_head.proj", dtype="F32")
    return rules


def _validate_scales(tensors, rules):
    for name, tensor in tensors.items():
        if not name.endswith(".scale") or tensor["dtype"] != "F8_E8M0":
            continue
        weight = tensors[name.removesuffix(".scale") + ".weight"]
        rows, columns = weight["shape"]
        fp4 = weight["dtype"] == "I8"
        columns *= 2 if fp4 else 1
        host = rules[name]["placement"] == "shared_pinned_host"
        if fp4 or host:
            if columns % 32:
                raise ValueError(f"Unaligned per-row scale blocks: {name}")
            expected = [rows, columns // 32]
        else:
            expected = [(rows + 31) // 32, (columns + 31) // 32]
        if tensor["shape"] != expected:
            raise ValueError(f"Scale shape mismatch: {name}")
        axis = rules[name]["shard_axis"]
        logical = (rows, columns)
        if axis is not None and logical[axis] % (4 * (1 if fp4 and axis == 0 else 32)):
            raise ValueError(f"Shard splits a quantization block: {name}")


def _tensor_placement(tensor, rule):
    dtype = tensor["dtype"]
    shape = tensor["shape"]
    if (
        dtype not in DTYPE_BYTES
        or not shape
        or any(type(d) is not int or d <= 0 for d in shape)
    ):
        raise ValueError("Unsupported tensor dtype or shape")
    stored_bytes = _size(*shape, DTYPE_BYTES[dtype])
    placement = rule["placement"]
    axis = rule["shard_axis"]
    owner = rule["owner_rank"]
    local_shape = list(shape)
    if axis is not None:
        if axis >= len(shape) or shape[axis] % 4:
            raise ValueError("Tensor cannot be divided into four exact shards")
        local_shape[axis] //= 4
    if owner is not None and owner not in range(4):
        raise ValueError("Invalid tensor owner")
    local_source = _size(*local_shape, DTYPE_BYTES[dtype])
    resident_dtype = rule["promoted_dtype"] or dtype
    local_resident = _size(*local_shape, DTYPE_BYTES[resident_dtype])
    rank_source = [
        local_source if owner is None or owner == rank else 0 for rank in range(4)
    ]
    rank_resident = [
        local_resident if owner is None or owner == rank else 0 for rank in range(4)
    ]
    if placement == "shared_pinned_host":
        rank_source = [0] * 4
        rank_resident = [0] * 4
    elif placement == "consumed_on_load":
        rank_resident = [0] * 4
    return {
        "stored_dtype": dtype,
        "stored_shape": list(shape),
        "stored_bytes": stored_bytes,
        "placement": placement,
        "shard_axis": axis,
        "owner_rank": owner,
        "rank_shard_ranges": (
            [
                [rank * shape[axis] // 4, (rank + 1) * shape[axis] // 4]
                for rank in range(4)
            ]
            if axis is not None
            else None
        ),
        "resident_dtype": resident_dtype,
        "rank_source_bytes": rank_source,
        "rank_resident_bytes": rank_resident,
    }


def _digest(value):
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid checkpoint metadata") from error
    return hashlib.sha256(encoded).hexdigest()


def weight_placements(model, tensors, strategy):
    """Return four-rank placements for the fixed config and stored tensor schema.

    Tensor entries contain only dtype and shape. These identities validate metadata,
    not weight bytes; EncodedReader still requires explicit verified file identities.
    """
    if type(strategy) is not str or strategy not in ("TP4", "TEP4", "DEP4"):
        raise ValueError("Strategy must be TP4, TEP4, or DEP4")
    if type(model) is not dict or _digest(model) != CONFIG_SHA256:
        raise ValueError("Unsupported checkpoint configuration identity")
    if type(tensors) is not dict or _digest(tensors) != LAYOUT_SHA256:
        raise ValueError("Unsupported checkpoint layout identity")
    rules = _placement_rules(model, strategy)
    if tensors.keys() != rules.keys():
        raise ValueError("Checkpoint tensors differ from placement rules")
    _validate_scales(tensors, rules)
    return {
        name: _tensor_placement(tensor, rules[name]) for name, tensor in tensors.items()
    }


@dataclass(frozen=True)
class RankPlan:
    selections: Mapping[str, TensorSlice]
    promotions: Mapping[str, str]
    resident_bytes: int
    source_bytes: int
    strategy: str
    rank: int


def rank_plan(model, tensors, strategy, rank):
    """Select one rank's raw stored coordinates and required resident promotions.

    Engram shared tables and other ranks' owned weights are excluded. Consumed wo_a
    scales remain selected. Source bytes include those scales; resident bytes do not.
    Native packing/padding and load scratch are outside these persistent byte totals.
    """
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Rank must be an integer from zero through three")
    placements = weight_placements(model, tensors, strategy)
    selections, promotions = {}, {}
    for name, tensor in placements.items():
        if not tensor["rank_source_bytes"][rank]:
            continue
        shape = tensor["stored_shape"]
        start, stop = [0] * len(shape), list(shape)
        axis = tensor["shard_axis"]
        if axis is not None:
            start[axis], stop[axis] = tensor["rank_shard_ranges"][rank]
        selections[name] = TensorSlice(tensor["stored_dtype"], shape, start, stop)
        if (
            tensor["rank_resident_bytes"][rank]
            and tensor["resident_dtype"] != tensor["stored_dtype"]
        ):
            promotions[name] = tensor["resident_dtype"]
    return RankPlan(
        MappingProxyType(selections),
        MappingProxyType(promotions),
        _uint(sum(t["rank_resident_bytes"][rank] for t in placements.values())),
        _uint(sum(t["rank_source_bytes"][rank] for t in placements.values())),
        strategy,
        rank,
    )
