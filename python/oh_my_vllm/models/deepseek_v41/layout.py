"""Checkpoint storage layout, not the reference model's runtime parameter layout.

The mathematical sources are the released inference/model.py, vision.py, engram.py,
and convert.py. No checkpoint index or tensor header supplies the expected layout.
Only the released Flash configuration is supported. Unknown fields fail validation.
"""

_SUPPORTED_TEXT = {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": True,
    "tie_word_embeddings": False,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
}

_SUPPORTED_VISION = {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": None,
}

_SUPPORTED_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "transformers_version": "5.6.0",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "text_config": _SUPPORTED_TEXT,
    "vision_config": _SUPPORTED_VISION,
}


def _validate_config(actual, expected, path="config"):
    if type(actual) is not type(expected):
        raise ValueError(f"{path}: expected {type(expected).__name__}")
    if isinstance(expected, dict):
        missing = expected.keys() - actual.keys()
        unknown = actual.keys() - expected.keys()
        if missing or unknown:
            raise ValueError(
                f"{path}: missing fields {missing!r}; unknown fields {unknown!r}"
            )
        for key, value in expected.items():
            _validate_config(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: unsupported list length")
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            _validate_config(value, reference, f"{path}[{index}]")
    elif actual != expected:
        raise ValueError(f"{path}: unsupported value {actual!r}; expected {expected!r}")


def _tensor(tensors, name, dtype, *shape):
    if name in tensors:
        raise ValueError(f"Duplicate expected tensor: {name}")
    tensors[name] = {"dtype": dtype, "shape": list(shape)}


def _linear(tensors, name, rows, columns, dtype="F8_E4M3", bias=False):
    stored_columns = columns // 2 if dtype == "I8" else columns
    _tensor(tensors, f"{name}.weight", dtype, rows, stored_columns)
    if dtype == "I8":
        _tensor(tensors, f"{name}.scale", "F8_E8M0", rows, columns // 32)
    elif dtype == "F8_E4M3":
        _tensor(
            tensors, f"{name}.scale", "F8_E8M0", (rows + 31) // 32, (columns + 31) // 32
        )
    if bias:
        _tensor(tensors, f"{name}.bias", "BF16", rows)


def _attention(tensors, name, text, layer_id):
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    head_dim = text["head_dim"]
    q_rank = text["q_lora_rank"]
    groups = text["o_groups"]
    o_rank = text["o_lora_rank"]
    _tensor(tensors, f"{name}.attn_sink", "F32", heads)
    _tensor(tensors, f"{name}.q_norm.weight", "BF16", q_rank)
    _tensor(tensors, f"{name}.kv_norm.weight", "BF16", head_dim)
    _linear(tensors, f"{name}.wq_a", q_rank, hidden)
    _linear(tensors, f"{name}.wq_b", heads * head_dim, q_rank)
    _linear(tensors, f"{name}.wkv", head_dim, hidden)
    # convert.py dequantizes wo_a before the reference BF16 grouped projection.
    _linear(tensors, f"{name}.wo_a", groups * o_rank, heads * head_dim // groups)
    _linear(tensors, f"{name}.wo_b", hidden, groups * o_rank)
    if layer_id in text["kv_source_layer_ids"]:
        _tensor(tensors, f"{name}.compressor.norm.weight", "BF16", head_dim)
        _linear(tensors, f"{name}.compressor.wkv", head_dim, hidden, "BF16")
        if text["compress_ratios"][layer_id] > 1:
            _linear(tensors, f"{name}.compressor.wgate", head_dim, hidden, "BF16")
    if layer_id in text["index_source_layer_ids"]:
        _indexer(tensors, f"{name}.indexer", text, layer_id)


def _indexer(tensors, name, text, layer_id):
    heads = text["index_n_heads"]
    dim = text["index_head_dim"]
    _linear(tensors, f"{name}.wq_b", heads * dim, text["q_lora_rank"])
    _linear(tensors, f"{name}.weights_proj", heads, text["hidden_size"], "BF16")
    if layer_id in text["kv_source_layer_ids"]:
        _linear(tensors, f"{name}.wk", dim, text["head_dim"], "BF16")
        _tensor(tensors, f"{name}.k_norm.weight", "BF16", dim)


def _expert(tensors, name, hidden, intermediate, dtype):
    _linear(tensors, f"{name}.w1", intermediate, hidden, dtype)
    _linear(tensors, f"{name}.w2", hidden, intermediate, dtype)
    _linear(tensors, f"{name}.w3", intermediate, hidden, dtype)


def _block(tensors, name, text, layer_id, experts):
    hidden = text["hidden_size"]
    hc = text["hc_mult"]
    mix = (2 + hc) * hc
    for branch in ("attn", "ffn"):
        _tensor(tensors, f"{name}.{branch}_norm.weight", "BF16", hidden)
        _tensor(tensors, f"{name}.hc_{branch}_fn", "F32", mix, hc * hidden)
        _tensor(tensors, f"{name}.hc_{branch}_base", "F32", mix)
        _tensor(tensors, f"{name}.hc_{branch}_scale", "F32", 3)
    _attention(tensors, f"{name}.attn", text, layer_id)
    _linear(tensors, f"{name}.ffn.gate", experts, hidden, "BF16")
    for bias in ("bias", "bias_vl"):
        _tensor(tensors, f"{name}.ffn.gate.{bias}", "F32", experts)
    intermediate = text["moe_intermediate_size"]
    for expert in range(experts):
        _expert(tensors, f"{name}.ffn.experts.{expert}", hidden, intermediate, "I8")
    _expert(
        tensors,
        f"{name}.ffn.shared_experts",
        hidden,
        intermediate * text["n_shared_experts"],
        "F8_E4M3",
    )


def _engram(tensors, text):
    hidden = text["hidden_size"]
    dim = text["engram_head_dim"]
    hc = text["hc_mult"]
    columns = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    for layer, rows in zip(
        text["engram_layer_ids"], text["engram_num_embeddings"], strict=True
    ):
        name = f"layers.{layer}.engram"
        _tensor(tensors, f"{name}.embed.weight", "F8_E4M3", rows, dim)
        # Engram scales cover 32 columns in each row, not a 32-by-32 block.
        _tensor(tensors, f"{name}.embed.scale", "F8_E8M0", rows, dim // 32)
        for weight in ("q_weight", "k_weight"):
            _tensor(tensors, f"{name}.{weight}", "BF16", hc, hidden)
        _linear(tensors, f"{name}.wkv", hidden * (hc + 1), columns * dim)


def _vision(tensors, vision, hidden):
    dim = vision["hidden_size"]
    intermediate = vision["intermediate_size"]
    _linear(
        tensors,
        "vision.patch_embed.proj",
        dim,
        3 * vision["patch_size"] ** 2,
        "BF16",
        True,
    )
    for layer in range(vision["num_hidden_layers"]):
        name = f"vision.blocks.{layer}"
        for norm in ("norm1", "norm2"):
            _tensor(tensors, f"{name}.{norm}.weight", "BF16", dim)
        _linear(tensors, f"{name}.attn.wqkv", 3 * dim, dim, "BF16", True)
        _linear(tensors, f"{name}.attn.wo", dim, dim, "BF16", True)
        _linear(tensors, f"{name}.mlp.w1", 2 * intermediate, dim, "BF16")
        _linear(tensors, f"{name}.mlp.w2", dim, intermediate, "BF16")
    _tensor(tensors, "vision.norm.weight", "BF16", dim)
    _linear(
        tensors,
        "aligner.w1",
        hidden,
        dim * vision["downsample_ratio"] ** 2,
        "BF16",
        True,
    )
    _linear(tensors, "aligner.w2", hidden, hidden, "BF16", True)
    for name in ("image_start", "image_end", "image_newline"):
        _tensor(tensors, name, "BF16", hidden)


def _dspark(tensors, text):
    hidden = text["hidden_size"]
    stages = text["num_nextn_predict_layers"]
    for stage in range(stages):
        _block(
            tensors,
            f"mtp.{stage}",
            text,
            text["num_hidden_layers"] + stage,
            text["dspark_n_routed_experts"],
        )
    _linear(
        tensors,
        "mtp.0.main_proj",
        hidden,
        hidden * len(text["dspark_target_layer_ids"]),
    )
    _tensor(tensors, "mtp.0.main_norm.weight", "BF16", hidden)
    last = f"mtp.{stages - 1}"
    _tensor(tensors, f"{last}.norm.weight", "BF16", hidden)
    rank = text["dspark_markov_rank"]
    for name in ("embed", "head"):
        _tensor(
            tensors,
            f"{last}.markov_head.{name}.weight",
            "BF16",
            text["vocab_size"],
            rank,
        )
    _linear(tensors, f"{last}.confidence_head.proj", 1, hidden + rank, "BF16")
    # Each stage uses the target embedding and head; the export has no duplicate keys.


def tensor_layout(config: dict) -> dict[str, dict]:
    """Derive the storage schema; reject unsupported configuration."""
    _validate_config(config, _SUPPORTED_CONFIG)
    text = config["text_config"]
    hidden = text["hidden_size"]
    tensors = {}
    for name in ("embed", "head"):
        _tensor(tensors, f"{name}.weight", "BF16", text["vocab_size"], hidden)
    _tensor(tensors, "norm.weight", "BF16", hidden)
    for layer in range(text["num_hidden_layers"]):
        _block(tensors, f"layers.{layer}", text, layer, text["n_routed_experts"])
    _engram(tensors, text)
    _vision(tensors, config["vision_config"], hidden)
    _dspark(tensors, text)
    return tensors
