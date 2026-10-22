import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from .reference_model import Model, TorchWeights, engram, expert, moe


class ArrayWeights:
    def __init__(self, arrays):
        self.arrays = arrays
        self.layout = {
            name: {"shape": list(array.shape)} for name, array in arrays.items()
        }
        self.reads = []

    def tensor(self, name, *, rows=None):
        self.reads.append((name, None if rows is None else tuple(rows)))
        value = self.arrays[name]
        return value.copy() if rows is None else value[rows].copy()


class StreamedReference(unittest.TestCase):
    def test_backbone_composition_keeps_endpoint_and_continuation_state(self):
        text = dict(
            hidden_size=2,
            vocab_size=3,
            num_hidden_layers=1,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            hc_eps=1e-6,
            rms_norm_eps=1e-20,
            max_position_embeddings=8,
            engram_layer_ids=[],
            dspark_target_layer_ids=[0],
            head_dim=2,
            num_attention_heads=1,
            q_lora_rank=2,
            o_groups=1,
            o_lora_rank=2,
            sliding_window=3,
            compress_ratios=[0],
            qk_rope_head_dim=0,
            num_experts_per_tok=1,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
        )
        arrays = {
            "embed.weight": np.array([[3.0, 4.0], [1.0, 0.0], [-2.0, 1.0]]),
            "head.weight": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            "norm.weight": np.ones(2),
            "layers.0.attn.attn_sink": np.zeros(1),
            "layers.0.ffn.gate.weight": np.zeros((2, 2)),
            "layers.0.ffn.gate.bias": np.array([1.0, 0.0]),
        }
        for branch in ("attn", "ffn"):
            arrays[f"layers.0.{branch}_norm.weight"] = np.ones(2)
            arrays[f"layers.0.hc_{branch}_fn"] = np.zeros((8, 4))
            arrays[f"layers.0.hc_{branch}_base"] = np.zeros(8)
            arrays[f"layers.0.hc_{branch}_scale"] = np.ones(3)
        for norm in ("q_norm", "kv_norm"):
            arrays[f"layers.0.attn.{norm}.weight"] = np.ones(2)
        for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            arrays[f"layers.0.attn.{projection}.weight"] = (
                np.eye(2) if projection == "wkv" else np.zeros((2, 2))
            )
        for expert_name in ("experts.0", "shared_experts"):
            for projection in ("w1", "w2", "w3"):
                arrays[f"layers.0.ffn.{expert_name}.{projection}.weight"] = np.zeros(
                    (2, 2)
                )
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
            {"text_config": text},
            np.arange(3),
        )
        first = model.forward([0])
        expected = torch.tensor([[3.0, 4.0, -7.0]], dtype=torch.float64) / math.sqrt(
            12.5
        )
        torch.testing.assert_close(first.logits, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            first.features, torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        )
        saved_window = first.state.layers[0].window_kv.clone()
        second = model.forward([1], previous=first.state)
        together = model.forward([0, 1], endpoint_rows=[0, 1])
        torch.testing.assert_close(
            torch.cat((first.logits, second.logits)),
            together.logits,
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertEqual(second.state.length, 2)
        torch.testing.assert_close(
            first.state.layers[0].window_kv, saved_window, rtol=0, atol=0
        )
        with self.assertRaises(ValueError):
            model.forward([True])
        with self.assertRaises(ValueError):
            model.forward([0] * 9)
        text.update(
            engram_layer_ids=[0],
            engram_max_ngram_size=4,
            engram_n_heads=1,
            engram_vocab_size=5,
            engram_compressed_vocab_size=3,
            engram_num_embeddings=[23],
            engram_pad_token_id=2,
        )
        arrays.update(
            {
                "layers.0.engram.embed.weight": np.arange(46, dtype=np.float64).reshape(
                    23, 2
                )
                / 20,
                "layers.0.engram.wkv.weight": np.full((6, 6), 0.125),
                "layers.0.engram.q_weight": np.ones((2, 2)),
                "layers.0.engram.k_weight": np.ones((2, 2)),
            }
        )
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
            {"text_config": text},
            np.arange(3),
        )
        prefix = model.forward([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "history"):
            model.forward([1], previous=replace(prefix.state, engram_history=()))
        continued = model.forward([1], previous=prefix.state)
        whole = model.forward([0, 1, 2, 1])
        torch.testing.assert_close(
            continued.logits, whole.logits, rtol=1e-14, atol=1e-14
        )
        self.assertEqual(continued.state.engram_history, (1, 2, 1))

    def test_projection_streams_output_rows_without_losing_columns(self):
        source = ArrayWeights(
            {"proj.weight": np.arange(15, dtype=np.float64).reshape(5, 3)}
        )
        weights = TorchWeights(source, device="cpu:0", row_limit=2)
        value = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        output = weights.linear(value, "proj")
        torch.testing.assert_close(
            output, torch.tensor([[8.0, 26.0, 44.0, 62.0, 80.0]], dtype=torch.float64)
        )
        self.assertEqual(
            source.reads,
            [("proj.weight", (0, 1)), ("proj.weight", (2, 3)), ("proj.weight", (4,))],
        )

    def test_expert_caps_gate_only_above_and_scales_before_down_projection(self):
        source = ArrayWeights(
            {f"expert.{name}.weight": np.eye(2) for name in ("w1", "w2", "w3")}
        )
        weights = TorchWeights(source, device="cpu", row_limit=2)
        result = expert(
            weights,
            "expert",
            torch.tensor([[2.0, -3.0]], dtype=torch.float64),
            limit=1.0,
            routing_weight=torch.tensor([0.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result,
            torch.tensor(
                [[0.25 / (1 + math.exp(-1)), 0.75 / (1 + math.exp(3))]],
                dtype=torch.float64,
            ),
        )
        arrays = {
            "m.gate.weight": np.zeros((3, 2)),
            "m.gate.bias": np.array([2.0, 1.0, 0.0]),
        }
        for index in (0, 1):
            arrays[f"m.experts.{index}.w1.weight"] = np.eye(2)
            arrays[f"m.experts.{index}.w3.weight"] = np.eye(2)
            arrays[f"m.experts.{index}.w2.weight"] = np.eye(2) * (index + 1)
        for name in ("w1", "w2", "w3"):
            arrays[f"m.shared_experts.{name}.weight"] = np.eye(2)
        weights = TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2)
        config = dict(
            num_hidden_layers=1,
            num_experts_per_tok=2,
            routed_scaling_factor=1.5,
            swiglu_limit=0,
        )
        value = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], dtype=torch.float64)
        output = moe(weights, "m", value, config=config, layer=0)
        expected = [
            [3.25 / (1 + math.exp(-1)), 13 / (1 + math.exp(-2))],
            [3.25 / (1 + math.exp(1)), 13 / (1 + math.exp(2))],
        ]
        torch.testing.assert_close(output, torch.tensor(expected, dtype=torch.float64))

    def test_engram_zero_similarity_uses_signed_floor_and_masks_media(self):
        source = ArrayWeights(
            {
                "e.embed.weight": np.ones((1, 2)),
                "e.wkv.weight": np.array([[0.5, 0.5]] * 4 + [[1.0, 1.0], [1.5, 1.5]]),
                "e.q_weight": np.ones((2, 2)),
                "e.k_weight": np.ones((2, 2)),
            }
        )
        weights = TorchWeights(source, device="cpu", row_limit=2)
        value = torch.zeros((2, 2, 2), dtype=torch.float64)
        result = engram(
            weights,
            "e",
            value,
            np.array([[0], [0]]),
            norm_eps=1e-20,
            token_mask=torch.tensor([True, False]),
        )
        gate = 1 / (1 + math.exp(-0.001))
        expected = torch.tensor(
            [[[2 * gate, 3 * gate]] * 2, [[0.0, 0.0]] * 2], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        self.assertEqual(source.reads[0], ("e.embed.weight", (0, 0)))


if __name__ == "__main__":
    unittest.main()
