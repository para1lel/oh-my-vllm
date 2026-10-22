"""Small CPU checks for independent target-attention mathematics and state."""

import copy
import math
import unittest

import torch

from development.model.deepseek_v41 import reference_attention as attention


def small_config():
    return {
        "hidden_size": 4,
        "num_hidden_layers": 40,
        "num_attention_heads": 2,
        "head_dim": 2,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 2,
        "o_lora_rank": 1,
        "o_groups": 2,
        "rms_norm_eps": 1e-20,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "rope_scaling": {
            "factor": 16,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "index_n_heads": 2,
        "index_head_dim": 2,
        "index_topk": 2,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 1,
        "candidate_block_size": 2,
    }


class SmallWeights:
    def __init__(self, config):
        self.values = {}
        generator = torch.Generator().manual_seed(67)
        hidden, width = config["hidden_size"], config["head_dim"]
        heads, rank = config["num_attention_heads"], config["q_lora_rank"]
        groups, output_rank = config["o_groups"], config["o_lora_rank"]
        index_heads, index_width = config["index_n_heads"], config["index_head_dim"]
        for layer in range(config["num_hidden_layers"]):
            prefix = f"layers.{layer}.attn"
            shapes = {
                "wq_a.weight": (rank, hidden),
                "q_norm.weight": (rank,),
                "wq_b.weight": (heads * width, rank),
                "wkv.weight": (width, hidden),
                "kv_norm.weight": (width,),
                "attn_sink": (heads,),
                "wo_a.weight": (groups * output_rank, heads * width // groups),
                "wo_b.weight": (hidden, groups * output_rank),
            }
            if layer in config["kv_source_layer_ids"]:
                shapes.update(
                    {
                        "compressor.wkv.weight": (width, hidden),
                        "compressor.norm.weight": (width,),
                        "indexer.wk.weight": (index_width, width),
                        "indexer.k_norm.weight": (index_width,),
                    }
                )
                if config["compress_ratios"][layer] > 1:
                    shapes["compressor.wgate.weight"] = (width, hidden)
            if layer in config["index_source_layer_ids"]:
                shapes.update(
                    {
                        "indexer.wq_b.weight": (index_heads * index_width, rank),
                        "indexer.weights_proj.weight": (index_heads, hidden),
                    }
                )
            for name, shape in shapes.items():
                value = torch.randn(shape, generator=generator, dtype=torch.float64) / 3
                if "norm.weight" in name:
                    value = torch.ones(shape, dtype=torch.float64)
                self.values[f"{prefix}.{name}"] = value

    def tensor(self, name, rows=None):
        value = self.values[name]
        return value if rows is None else value[rows]

    def linear(self, value, prefix):
        if prefix.endswith("wo_a"):
            raise AssertionError("wo_a must use grouped einsum")
        return value @ self.tensor(prefix + ".weight").T


class ReferenceAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_known_sink_causality_and_pre_rotary_compression(self):
        config = small_config()
        weights = SmallWeights(config)
        prefix = "layers.0.attn"
        weights.values[prefix + ".wq_b.weight"].zero_()
        weights.values[prefix + ".wkv.weight"] = torch.eye(2, 4, dtype=torch.float64)
        weights.values[prefix + ".wo_a.weight"] = torch.tensor(
            [[2, 3], [5, 7]], dtype=torch.float64
        )
        weights.values[prefix + ".wo_b.weight"] = torch.eye(4, 2, dtype=torch.float64)
        weights.values[prefix + ".attn_sink"] = torch.tensor(
            [0, math.log(2)], dtype=torch.float64
        )
        value = torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0]], dtype=torch.float64)
        output, state = attention.forward(
            weights, prefix, value, layer=0, config=config, shared={}
        )
        root = math.sqrt(2)
        expected = torch.tensor(
            [
                [root, 5 * root / 3, 0, 0],
                [
                    root * (2 * math.cos(1) + 3 * (1 - math.sin(1))) / 3,
                    root * (5 * math.cos(1) + 7 * (1 - math.sin(1))) / 4,
                    0,
                    0,
                ],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(output, expected, rtol=1e-14, atol=1e-14)
        changed, _ = attention.forward(
            weights, prefix, value[:1], layer=0, config=config, shared={}
        )
        torch.testing.assert_close(changed, output[:1], rtol=0, atol=0)
        self.assertEqual(state.length, 2)

        prefix = "layers.2.attn"
        weights.values[prefix + ".compressor.wkv.weight"] = torch.eye(
            2, 4, dtype=torch.float64
        )
        weights.values[prefix + ".compressor.wgate.weight"] = torch.tensor(
            [[0, math.log(3) / 2, 0, 0]] * 2, dtype=torch.float64
        )
        weights.values[prefix + ".indexer.wk.weight"] = torch.diag(
            torch.tensor([2, 1], dtype=torch.float64)
        )
        value = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0]] * 2, dtype=torch.float64)
        _, state = attention.forward(
            weights, prefix, value, layer=2, config=config, shared={}
        )
        latent = torch.tensor([0.25, 1.5], dtype=torch.float64)
        latent /= torch.sqrt(latent.square().mean() + 1e-20)
        key = latent * torch.tensor([2, 1], dtype=torch.float64)
        key /= torch.sqrt(key.square().mean() + 1e-20)
        rotation = torch.tensor(
            [[math.cos(2), -math.sin(2)], [math.sin(2), math.cos(2)]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            state.compress_kv,
            torch.stack((latent, rotation @ latent)),
            rtol=1e-14,
            atol=1e-14,
        )
        torch.testing.assert_close(
            state.index_k, torch.stack((key, rotation @ key)), rtol=1e-14, atol=1e-14
        )

        config.update(head_dim=4, qk_rope_head_dim=4, index_head_dim=4)
        weights = SmallWeights(config)
        value = torch.tensor([[1, 2, 3, 4]] * 2, dtype=torch.float64)
        normalized = value[0] / torch.sqrt(value[0].square().mean() + 1e-20)
        for layer, angle in ((0, 10000**-0.5), (2, 160000**-0.5 / 16)):
            prefix = f"layers.{layer}.attn"
            weights.values[prefix + ".wkv.weight"] = torch.eye(4, dtype=torch.float64)
            _, state = attention.forward(
                weights, prefix, value, layer=layer, config=config, shared={}
            )
            expected = normalized.clone()
            for pair, frequency in ((0, 1), (2, angle)):
                real, imaginary = normalized[pair : pair + 2]
                expected[pair] = real * math.cos(frequency) - imaginary * math.sin(
                    frequency
                )
                expected[pair + 1] = real * math.sin(frequency) + imaginary * math.cos(
                    frequency
                )
            torch.testing.assert_close(
                state.window_kv[1], expected, rtol=1e-14, atol=1e-14
            )

    def test_chunked_layers_keep_owner_state_without_mutation_or_request_leaks(self):
        config = small_config()
        weights = SmallWeights(config)
        generator = torch.Generator().manual_seed(17)
        inputs = torch.randn((40, 133, 4), generator=generator, dtype=torch.float64)

        def run(cuts):
            states, outputs = {}, [[] for _ in range(40)]
            start = 0
            for count in cuts:
                shared = {}
                for layer in range(40):
                    previous = states.get(layer)
                    snapshot = copy.deepcopy(previous)
                    output, state = attention.forward(
                        weights,
                        f"layers.{layer}.attn",
                        inputs[layer, start : start + count],
                        layer=layer,
                        config=config,
                        previous=previous,
                        start=start,
                        shared=shared,
                    )
                    if previous is not None:
                        self.assert_state_equal(previous, snapshot, exact=True)
                    states[layer] = state
                    outputs[layer].append(output)
                    if layer in (3, 9, 15, 21, 25, 29, 33, 37):
                        self.assertEqual(
                            shared["kv_source"],
                            max(source for source in (2, 8, 14, 20) if source <= layer),
                        )
                        self.assertEqual(shared["index_source"], layer - 1)
                self.assertEqual(shared["candidate_source"], 20)
                self.assertTrue(
                    all(len(blocks) <= 1 for blocks in shared["candidates"])
                )
                start += count
            return [torch.cat(rows) for rows in outputs], states

        full, full_states = run([133])
        chunks, chunk_states = run([1, 126, 1, 3, 1, 1])
        repeat, _ = run([133])
        for layer in range(40):
            torch.testing.assert_close(
                full[layer], chunks[layer], rtol=2e-12, atol=2e-12
            )
            torch.testing.assert_close(full[layer], repeat[layer], rtol=0, atol=0)
            self.assert_state_equal(full_states[layer], chunk_states[layer])
            state = chunk_states[layer]
            self.assertEqual(state.window_kv.shape, (128, 2))
            if layer in (2, 8, 14):
                self.assertEqual(state.compress_kv.shape, (66, 2))
                self.assertEqual(state.pending_values.shape, (1, 2))
            if layer == 20:
                self.assertEqual(state.compress_kv.shape, (133, 2))
        with self.assertRaises(ValueError):
            attention.forward(
                weights,
                "layers.2.attn",
                inputs[2, :1],
                layer=2,
                config=config,
                previous=chunk_states[2],
                start=132,
                shared={},
            )
        with self.assertRaises(ValueError):
            attention.forward(
                weights,
                "layers.3.attn",
                inputs[3, :1],
                layer=3,
                config=config,
                shared={},
            )

    def assert_state_equal(self, actual, expected, exact=False):
        self.assertEqual(vars(actual).keys(), vars(expected).keys())
        for name, value in vars(actual).items():
            other = getattr(expected, name)
            if isinstance(value, torch.Tensor):
                options = (
                    {"rtol": 0, "atol": 0} if exact else {"rtol": 2e-12, "atol": 2e-12}
                )
                torch.testing.assert_close(value, other, **options)
            else:
                self.assertEqual(value, other)

    def test_tiled_signed_scores_stable_ties_and_reachable_candidate_blocks(self):
        query = torch.tensor([[1, 0], [-1, 0]], dtype=torch.float64)
        keys = torch.zeros((4101, 2), dtype=torch.float64)
        keys[0, 0], keys[4096, 0], keys[4100, 0] = 2, -3, 4
        head_weights = torch.tensor([2, -1], dtype=torch.float64)
        scores = attention._index_scores(query, keys, head_weights, visible=4097)
        self.assertEqual(scores.shape, (4097,))
        self.assertEqual(scores[0], 4)
        self.assertEqual(scores[4096], -3)
        torch.testing.assert_close(
            attention._select_global(scores, 3), torch.tensor([0, 1, 2])
        )
        blocks = attention._candidate_blocks(scores, topk=1, block_size=8)
        torch.testing.assert_close(blocks, torch.tensor([512]))
        masked = attention._index_scores(
            query,
            keys,
            head_weights,
            visible=4097,
            candidates=torch.cat((blocks, torch.tensor([-1]))),
            block_size=8,
        )
        torch.testing.assert_close(
            attention._select_global(masked, 8), torch.tensor([4096])
        )
        invalid = torch.tensor([1, -torch.inf, 1, torch.nan], dtype=torch.float64)
        torch.testing.assert_close(
            attention._select_global(invalid, 4), torch.tensor([0, 2])
        )
        empty = attention._candidate_blocks(scores[:0], topk=2, block_size=8)
        self.assertEqual(empty.numel(), 0)


if __name__ == "__main__":
    unittest.main()
