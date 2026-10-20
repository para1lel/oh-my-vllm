ATTENTION = (
    "input_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    *tuple(
        f"self_attn.{projection}_proj.{suffix}"
        for projection in ("q", "k", "v", "o")
        for suffix in ("weight", "weight_scale_inv")
    ),
)
FFN = (
    "post_attention_layernorm.weight",
    *tuple(
        f"mlp.{projection}_proj.{suffix}"
        for projection in ("gate", "up", "down")
        for suffix in ("weight", "weight_scale_inv")
    ),
)
GDN = (
    "input_layernorm.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    *tuple(
        f"linear_attn.{projection}.{suffix}"
        for projection in ("in_proj_qkv", "in_proj_z", "out_proj")
        for suffix in ("weight", "weight_scale_inv")
    ),
)
VISION_BLOCK = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "norm2.weight",
    "norm2.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias",
)
VISION_MERGER = (
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
)
EMBEDDING = "model.language_model.embed_tokens.weight"


def weight_names(case):
    family = case.boundary.split(".")[0]
    if family in {"fa", "gdn", "ffn"}:
        names = [
            f"model.language_model.layers.{layer}.{name}"
            for layer in case.weights
            for name in {"fa": ATTENTION, "gdn": GDN, "ffn": FFN}[family]
        ]
        if family == "gdn" and 0 in case.weights:
            names.append(EMBEDDING)
    elif family == "output":
        names = ["model.language_model.norm.weight"]
        if case.boundary != "output.hidden":
            names.append("lm_head.weight")
    elif family == "mtp":
        if case.boundary == "mtp.accept":
            return []
        names = [
            EMBEDDING,
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.fc.weight",
        ]
        names += ["mtp.layers.0." + name for name in ATTENTION]
        if case.boundary != "mtp.context":
            names += ["mtp.layers.0." + name for name in FFN]
            names += ["mtp.norm.weight", "lm_head.weight"]
    else:
        names = [
            f"model.visual.blocks.{layer}.{name}"
            for layer in case.weights
            for name in VISION_BLOCK
        ]
        if case.boundary == "vision.input":
            names += [
                "model.visual.patch_embed.proj.weight",
                "model.visual.patch_embed.proj.bias",
                "model.visual.pos_embed.weight",
            ]
        elif case.boundary == "vision.merge":
            names += ["model.visual.merger." + name for name in VISION_MERGER]
    return sorted(names)
