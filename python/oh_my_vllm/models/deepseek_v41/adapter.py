"""DeepSeek-V4.1-Flash identity and configuration checks."""

from ..api import AdapterSpec

CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
INFERENCE_CONFIG_SHA256 = (
    "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"
)
ARCHITECTURE = "DeepseekV41ForCausalLM"
MODEL_TYPE = "deepseek_v41"
VOCABULARY_SIZE = 129280
MAX_CONTEXT_TOKENS = 1048576
NUM_LAYERS = 40
WINDOW_SIZE = 128
DS_PARK_STAGES = 3
DS_PARK_DRAFT_TOKENS = 5
DS_PARK_VERIFICATION_ROWS = 6
COMPRESS_RATIOS = (
    0,
    0,
    *([2] * 18),
    *([1] * 20),
    0,
    0,
    0,
)
KV_SOURCE_LAYERS = (2, 8, 14, 20)
INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(value) == len(expected) and all(
            _same(actual, item) for actual, item in zip(value, expected, strict=True)
        )
    return value == expected


def validate_config(config):
    if type(config) is not dict:
        raise ValueError("DeepSeek configuration must be an object")
    if (
        config.get("architectures") != [ARCHITECTURE]
        or config.get("model_type") != MODEL_TYPE
    ):
        raise ValueError("unsupported DeepSeek architecture or model type")

    quantization = config.get("quantization_config")
    expected_quantization = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }
    if not _same(quantization, expected_quantization):
        raise ValueError("unsupported DeepSeek weight or activation format")

    text = config.get("text_config")
    expected_text = {
        "model_type": "deepseek_v41_text",
        "vocab_size": VOCABULARY_SIZE,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": NUM_LAYERS,
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
        "max_position_embeddings": MAX_CONTEXT_TOKENS,
        "rope_theta": 10000,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": WINDOW_SIZE,
        "compress_ratios": list(COMPRESS_RATIOS),
        "compress_rope_theta": 160000,
        "kv_source_layer_ids": list(KV_SOURCE_LAYERS),
        "index_source_layer_ids": list(INDEX_SOURCE_LAYERS),
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "engram_layer_ids": list(ENGRAM_LAYERS),
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": DS_PARK_STAGES,
        "dspark_block_size": DS_PARK_DRAFT_TOKENS,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }
    if not isinstance(text, dict) or any(
        not _same(text.get(key), expected) for key, expected in expected_text.items()
    ):
        raise ValueError("unsupported DeepSeek text configuration")


def build(config, raw_config):
    from .launcher import GroupWorker

    return GroupWorker(config, raw_config=raw_config)


SPEC = AdapterSpec(
    key="deepseek_v41",
    architecture=ARCHITECTURE,
    model_type=MODEL_TYPE,
    capabilities=frozenset(("text",)),
    validate_config=validate_config,
    build=build,
)
