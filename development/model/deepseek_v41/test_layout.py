import copy
import unittest
from collections import Counter

from development.model.deepseek_v41 import layout


def checkpoint_config():
    return {
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
        "text_config": {
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
        },
        "vision_config": {
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
        },
    }


class TensorLayoutTests(unittest.TestCase):
    def test_checkpoint_storage_schema(self):
        tensors = layout.tensor_layout(checkpoint_config())
        self.assertEqual(len(tensors), 96085)
        self.assertEqual(
            Counter(t["dtype"] for t in tensors.values()),
            {"BF16": 520, "F32": 387, "F8_E4M3": 357, "F8_E8M0": 47589, "I8": 47232},
        )
        anchors = {
            "embed.weight": ("BF16", [129280, 5120]),
            "head.weight": ("BF16", [129280, 5120]),
            "norm.weight": ("BF16", [5120]),
            "layers.0.attn.wo_a.weight": ("F8_E4M3", [8192, 4096]),
            "layers.0.attn.wo_a.scale": ("F8_E8M0", [256, 128]),
            "layers.39.ffn.experts.383.w2.weight": ("I8", [5120, 1152]),
            "layers.39.ffn.experts.383.w2.scale": ("F8_E8M0", [5120, 72]),
            "layers.0.ffn.shared_experts.w1.scale": ("F8_E8M0", [72, 160]),
            "layers.0.ffn.gate.bias_vl": ("F32", [384]),
            "layers.2.attn.compressor.wgate.weight": ("BF16", [512, 5120]),
            "layers.20.attn.compressor.wkv.weight": ("BF16", [512, 5120]),
            "layers.2.attn.indexer.wk.weight": ("BF16", [128, 512]),
            "layers.36.attn.indexer.wq_b.scale": ("F8_E8M0", [128, 40]),
            "layers.0.hc_attn_fn": ("F32", [24, 20480]),
            "layers.1.engram.embed.weight": ("F8_E4M3", [384006168, 256]),
            "layers.14.engram.embed.scale": ("F8_E8M0", [384016682, 8]),
            "layers.14.engram.wkv.weight": ("F8_E4M3", [25600, 6144]),
            "layers.14.engram.q_weight": ("BF16", [4, 5120]),
            "vision.patch_embed.proj.weight": ("BF16", [1024, 588]),
            "vision.blocks.31.attn.wqkv.bias": ("BF16", [3072]),
            "vision.blocks.31.mlp.w1.weight": ("BF16", [5632, 1024]),
            "vision.norm.weight": ("BF16", [1024]),
            "aligner.w1.weight": ("BF16", [5120, 9216]),
            "image_newline": ("BF16", [5120]),
            "mtp.0.main_proj.weight": ("F8_E4M3", [5120, 15360]),
            "mtp.1.ffn.experts.127.w1.weight": ("I8", [2304, 2560]),
            "mtp.2.markov_head.embed.weight": ("BF16", [129280, 256]),
            "mtp.2.markov_head.head.weight": ("BF16", [129280, 256]),
            "mtp.2.confidence_head.proj.weight": ("BF16", [1, 5376]),
            "mtp.2.norm.weight": ("BF16", [5120]),
        }
        for name, (dtype, shape) in anchors.items():
            with self.subTest(name=name):
                self.assertEqual(tensors[name], {"dtype": dtype, "shape": shape})
        for name in (
            "layers.3.attn.compressor.wkv.weight",
            "layers.24.attn.indexer.wk.weight",
            "layers.20.attn.compressor.wgate.weight",
            "mtp.1.main_proj.weight",
            "mtp.0.confidence_head.proj.weight",
            "mtp.2.embed.weight",
            "mtp.2.head.weight",
            "mtp.2.attn.indexer.wq_b.weight",
            "layers.0.ffn.experts.384.w1.weight",
            "mtp.0.ffn.experts.128.w1.weight",
        ):
            self.assertNotIn(name, tensors)
        for tensor in tensors.values():
            self.assertEqual(set(tensor), {"dtype", "shape"})
            self.assertIs(type(tensor["shape"]), list)
            self.assertTrue(all(type(d) is int and d > 0 for d in tensor["shape"]))

    def test_rejects_unsupported_configuration(self):
        mutations = (
            ((), "architectures", ["UnknownForCausalLM"]),
            ((), "dtype", "float16"),
            ((), "unexpected", 1),
            (("quantization_config",), "weight_block_size", [128, 128]),
            (("quantization_config",), "expert_dtype", "fp8"),
            (("quantization_config",), "scale_fmt", "float32"),
            (("quantization_config",), "quant_method", "awq"),
            (("quantization_config",), "unexpected", True),
            (("text_config",), "hidden_size", 5121),
            (("text_config",), "num_hidden_layers", 0),
            (("text_config",), "num_key_value_heads", True),
            (("text_config",), "num_attention_heads", 64.0),
            (("text_config",), "tie_word_embeddings", True),
            (("text_config",), "kv_source_layer_ids", [2, 8, 14]),
            (
                ("text_config",),
                "index_source_layer_ids",
                [2, 8, 14, 20, 24, 28, 32, 40],
            ),
            (("text_config",), "compress_ratios", [0] * 43),
            (("text_config",), "engram_num_embeddings", [384006169, 384016682]),
            (("text_config",), "engram_layer_ids", [1, 1]),
            (("text_config",), "dspark_target_layer_ids", [36, 38, 39]),
            (("text_config",), "num_nextn_predict_layers", 5),
            (("text_config",), "unknown_topology", []),
            (("text_config", "rope_scaling"), "rope_type", "unknown"),
            (("vision_config",), "patch_size", 16),
            (("vision_config",), "num_attention_heads", 15),
        )
        for path, key, value in mutations:
            config = checkpoint_config()
            section = config
            for part in path:
                section = section[part]
            section[key] = value
            with self.subTest(path=path, key=key), self.assertRaises(ValueError):
                layout.tensor_layout(config)
        for path, key, _ in mutations:
            config = checkpoint_config()
            section = config
            for part in path:
                section = section[part]
            section.pop(key, None)
            if key.startswith(("unexpected", "unknown")):
                continue
            with self.subTest(missing=(*path, key)), self.assertRaises(ValueError):
                layout.tensor_layout(config)
        for invalid in (None, [], {"text_config": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                layout.tensor_layout(invalid)

    def test_does_not_mutate_or_share_configuration(self):
        config = checkpoint_config()
        original = copy.deepcopy(config)
        tensors = layout.tensor_layout(config)
        self.assertEqual(config, original)
        tensors["head.weight"]["shape"][0] = 1
        self.assertEqual(tensors["embed.weight"]["shape"], [129280, 5120])
        self.assertEqual(
            layout.tensor_layout(config)["head.weight"]["shape"], [129280, 5120]
        )


if __name__ == "__main__":
    unittest.main()
