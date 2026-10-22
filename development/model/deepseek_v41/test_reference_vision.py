import math
import unittest

import numpy as np
import torch

from . import reference_vision as vision
from .reference_model import TorchWeights


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


def weights(arrays):
    return TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2)


def gelu(value):
    return value * (1 + math.erf(value / math.sqrt(2))) / 2


class VisionReference(unittest.TestCase):
    def test_rope_uses_height_then_width_frequencies_and_split_halves(self):
        value = torch.arange(1, 9, dtype=torch.float64).repeat(4, 2, 1)
        result = vision.rotary_2d(value, 2, 2, theta=10000)
        c, s = math.cos(1), math.sin(1)
        d, t = math.cos(0.01), math.sin(0.01)
        expected = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [
                    1,
                    2,
                    3 * c - 7 * s,
                    4 * d - 8 * t,
                    5,
                    6,
                    7 * c + 3 * s,
                    8 * d + 4 * t,
                ],
                [c - 5 * s, 2 * d - 6 * t, 3, 4, 5 * c + s, 6 * d + 2 * t, 7, 8],
                [
                    c - 5 * s,
                    2 * d - 6 * t,
                    3 * c - 7 * s,
                    4 * d - 8 * t,
                    5 * c + s,
                    6 * d + 2 * t,
                    7 * c + 3 * s,
                    8 * d + 4 * t,
                ],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            result, expected[:, None].expand(-1, 2, -1), rtol=1e-14, atol=1e-14
        )
        with self.assertRaises(ValueError):
            vision.rotary_2d(value.float(), 2, 2)
        with self.assertRaises(ValueError):
            vision.rotary_2d(value[..., :6], 2, 2)

    def test_tiled_attention_keeps_all_bidirectional_keys_and_biases(self):
        qkv = np.zeros((12, 4))
        qkv[4, 0] = qkv[8, 0] = 1
        bias = np.zeros(12)
        bias[0], bias[8] = 1, 0.25
        reader = weights(
            {
                "a.wqkv.weight": qkv,
                "a.wqkv.bias": bias,
                "a.wo.weight": 2 * np.eye(4),
                "a.wo.bias": np.array([0.125, 0, 0, -0.5]),
            }
        )
        value = torch.tensor(
            [[1, 0, 0, 0], [3, 0, 0, 0], [5, 0, 0, 0]], dtype=torch.float64
        )
        result = vision.attention(reader, "a", value, 3, 1, heads=1, query_tile=1)
        expected = []
        for row in range(3):
            scores = [
                math.exp(k * math.cos(row - col) / 2) for col, k in enumerate((1, 3, 5))
            ]
            average = sum(
                p * (k + 0.25) for p, k in zip(scores, (1, 3, 5), strict=True)
            ) / sum(scores)
            expected.append([2 * average + 0.125, 0, 0, -0.5])
        torch.testing.assert_close(
            result, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=1e-14
        )
        torch.testing.assert_close(
            result,
            vision.attention(reader, "a", value, 3, 1, heads=1, query_tile=3),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertTrue(
            all(
                rows is not None and len(rows) <= 2
                for name, rows in reader.source.reads
                if name.endswith(".weight")
            )
        )

    def test_aligner_pads_3_by_3_cells_uses_erf_gelu_and_orders_learned_tokens(self):
        first = np.zeros((2, 18))
        first[0, 1] = first[0, 4] = first[1, 9] = 1
        reader = weights(
            {
                "aligner.w1.weight": first,
                "aligner.w1.bias": np.array([-0.5, -1]),
                "aligner.w2.weight": np.array([[2, -1], [0.5, 3]], dtype=np.float64),
                "aligner.w2.bias": np.array([0.125, -0.25]),
                "image_start": np.array([-1, -2], dtype=np.float64),
                "image_newline": np.array([-3, -4], dtype=np.float64),
                "image_end": np.array([-5, -6], dtype=np.float64),
            }
        )
        indices = torch.arange(1, 17, dtype=torch.float64)
        value = torch.stack((indices, 100 + indices), dim=-1) / 100
        features = vision.align(reader, value, 4, 4)
        expected = [
            [2 * gelu(a) - gelu(b) + 0.125, 0.5 * gelu(a) + 3 * gelu(b) - 0.25]
            for a, b in ((-0.42, 0.01), (-0.5, 0.04), (-0.36, 0.13), (-0.5, 0.16))
        ]
        torch.testing.assert_close(
            features,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-14,
            atol=1e-14,
        )
        span = vision.image_span(reader, features, 2, 2)
        expected_span = [
            [-1, -2],
            *expected[:2],
            [-3, -4],
            *expected[2:],
            [-3, -4],
            [-5, -6],
        ]
        torch.testing.assert_close(
            span,
            torch.tensor(expected_span, dtype=torch.float64),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_tiny_full_composition_normalizes_swiglu_and_packs_pixels_in_source_order(
        self,
    ):
        config = {
            "vision_config": dict(
                hidden_size=4,
                num_attention_heads=1,
                intermediate_size=1,
                num_hidden_layers=2,
                patch_size=2,
                rope_theta=10000,
                downsample_ratio=3,
                max_image_tokens=8,
            ),
            "text_config": {"hidden_size": 2},
        }
        arrays = {
            "vision.patch_embed.proj.weight": np.zeros((4, 12)),
            "vision.patch_embed.proj.bias": np.full(4, 0.25),
            "vision.norm.weight": np.ones(4),
            "aligner.w1.weight": np.zeros((2, 36)),
            "aligner.w1.bias": np.array([-0.5, 0.5]),
            "aligner.w2.weight": np.array([[2, -1], [0.5, 3]], dtype=np.float64),
            "aligner.w2.bias": np.array([0.125, -0.25]),
            "image_start": np.array([-1, -2], dtype=np.float64),
            "image_newline": np.array([-3, -4], dtype=np.float64),
            "image_end": np.array([-5, -6], dtype=np.float64),
        }
        arrays["vision.patch_embed.proj.weight"][:, [0, 5, 10]] = [1, 2, 3]
        arrays["aligner.w1.weight"][0, 0] = 1
        arrays["aligner.w1.weight"][1, 9] = -1
        for layer in range(2):
            prefix = f"vision.blocks.{layer}"
            arrays.update(
                {
                    prefix + ".norm1.weight": np.full(4, 2.0),
                    prefix + ".norm2.weight": np.full(4, 0.5),
                    prefix + ".attn.wqkv.weight": np.zeros((12, 4)),
                    prefix + ".attn.wqkv.bias": np.zeros(12),
                    prefix + ".attn.wo.weight": 0.25 * np.eye(4),
                    prefix + ".attn.wo.bias": np.full(4, 0.125),
                    prefix + ".mlp.w1.weight": np.array(
                        [[-1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float64
                    ),
                    prefix + ".mlp.w2.weight": np.full((4, 1), 0.25),
                }
            )
            arrays[prefix + ".attn.wqkv.weight"][8:, :] = np.eye(4)
        reader = weights(arrays)
        model = vision.Vision(reader, config, query_tile=1)
        pixels = torch.arange(1, 13, dtype=torch.float64).reshape(3, 2, 2) / 10
        patches = vision.pack_patches(pixels, 1, 1, patch_size=2)
        self.assertTrue(torch.equal(patches, pixels[None]))
        packed = vision.pack_patches(
            torch.arange(24, dtype=torch.float64).reshape(3, 2, 4), 1, 2, patch_size=2
        )
        self.assertEqual(
            packed.flatten(1).tolist(),
            [
                [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
                [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23],
            ],
        )
        scalar = 0.1 + 2 * 0.6 + 3 * 1.1 + 0.25
        for _ in range(2):
            scalar += 0.5 * scalar / math.sqrt(scalar**2 + 1e-6) + 0.125
            normalized = 0.5 * scalar / math.sqrt(scalar**2 + 1e-6)
            scalar -= 0.5 * normalized**2 / (1 + math.exp(normalized))
        normalized = scalar / math.sqrt(scalar**2 + 1e-6)
        a, b = gelu(normalized - 0.5), gelu(0.5 - normalized)
        expected = torch.tensor(
            [[-1, -2], [2 * a - b + 0.125, 0.5 * a + 3 * b - 0.25], [-3, -4], [-5, -6]],
            dtype=torch.float64,
        )
        result = model.forward(patches, 1, 1)
        torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            model.forward_pixels(pixels, 1, 1), expected, rtol=1e-14, atol=1e-14
        )
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(result.requires_grad)
        for invalid in (
            patches.float(),
            patches.flatten(1),
            patches.expand(2, -1, -1, -1),
        ):
            with self.assertRaises(ValueError):
                model.forward(invalid, 1, 1)
        for h, w in ((0, 1), (True, 1), (1, 22)):
            with self.assertRaises(ValueError):
                model.forward(patches, h, w)
        with self.assertRaises(ValueError):
            vision.Vision(reader, config, query_tile=0)
        reader.source.layout["vision.patch_embed.proj.weight"]["shape"] = [4, 3, 2, 2]
        with self.assertRaises(ValueError):
            vision.Vision(reader, config)


if __name__ == "__main__":
    unittest.main()
