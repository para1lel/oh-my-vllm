import hashlib
import json
from pathlib import Path

import torch

from .placement import _placement_rules

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}

RAW_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}

PARAM_DTYPES = {
    **RAW_DTYPES,
    "I8": torch.float4_e2m1fn_x2,
}

_MAPPING = {
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def converted_name(source_name):
    name = source_name.removeprefix("model.")
    if name.startswith("mtp.") and name.split(".", 2)[-1] in (
        "embed.weight",
        "head.weight",
    ):
        return None
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    key = (
        name.split(".")[-1]
        if any(
            value in name
            for value in ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")
        )
        else name.split(".")[-2]
    )
    mapped = _MAPPING.get(key)
    if mapped is not None:
        name = name.replace(key, mapped[0])
    return name


def source_index(root):
    raw_index = (Path(root) / "model.safetensors.index.json").read_bytes()
    index = json.loads(raw_index)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight map")
    return weight_map, hashlib.sha256(raw_index).hexdigest()


def _parameter_specs(config, layout, rank, strategy):
    rules = _placement_rules(config, strategy)
    result = {}
    for name, stored in layout.items():
        rule = rules[name]
        if rule["placement"] == "consumed_on_load":
            continue
        owner = rule["owner_rank"]
        if (
            owner is not None
            and owner != rank
            and not name.startswith(("vision.", "aligner.", "image_"))
        ):
            continue
        shape = list(stored["shape"])
        axis, start = rule["shard_axis"], 0
        dtype = rule["promoted_dtype"] or stored["dtype"]
        if ".engram.embed." in name and strategy != "DEP4":
            axis, shape[0] = 0, (shape[0] + 3) // 4
            start = rank * shape[0]
        else:
            if name.startswith(("vision.", "aligner.", "image_")):
                axis = None
            if axis is not None:
                if shape[axis] % 4:
                    raise ValueError(f"unsupported source shard shape: {name}")
                shape[axis] //= 4
                start = rank * shape[axis]
        if name.startswith("vision.") and name.endswith(
            (".norm1.weight", ".norm2.weight", ".norm.weight")
        ):
            dtype = "F32"
        result[name] = {
            "shape": tuple(shape),
            "dtype": dtype,
            "stored_dtype": stored["dtype"],
            "axis": axis,
            "start": start,
        }
    return result


def _fp8_to_bf16(raw, scales):
    rows, columns = raw.shape
    blocks = (rows + 31) // 32
    padded = torch.zeros((blocks * 32, columns), dtype=raw.dtype, device=raw.device)
    padded[:rows] = raw
    values = padded.float().reshape(blocks, 32, columns // 32, 32)
    scale_values = scales[:blocks].float().reshape(blocks, 1, columns // 32, 1)
    return (
        (values * scale_values).reshape(blocks * 32, columns)[:rows].to(torch.bfloat16)
    )


DEFAULT_TILE_ROWS = 65536

LOAD_CHUNK_BYTES = 256 << 20

DIRECT_COPY_BYTES = 4 << 20
