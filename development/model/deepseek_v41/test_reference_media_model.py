import unittest

import numpy as np
import torch

from .reference_model import Model, TorchWeights
from .test_reference_model import ArrayWeights, add_engram, backbone_fixture


class MediaModelReference(unittest.TestCase):
    def test_image_embeddings_masks_and_text_continuation_share_one_state(self):
        text, arrays = backbone_fixture()
        add_engram(text, arrays)
        arrays["layers.0.ffn.gate.bias_vl"] = np.array([0.0, 1.0])
        for name in ("w1", "w3"):
            arrays[f"layers.0.ffn.experts.1.{name}.weight"] = np.eye(2)
        arrays["layers.0.ffn.experts.1.w2.weight"] = np.diag([2.0, 0.0])
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
            {"text_config": text, "image_token_id": 2},
            np.arange(3),
        )
        types = np.array([-1, 0, 1, 2, 3, -1])
        embeddings = torch.tensor(
            [[1, 3], [-2, 4], [5, -1], [2, -3]], dtype=torch.float64
        )
        saved = embeddings.clone()
        tokens = [0, 2, 2, 2, 2, 1]
        output = model.forward(
            tokens,
            endpoint_rows=list(range(6)),
            token_types=types,
            image_embeddings=embeddings,
        )
        torch.testing.assert_close(output.features[1:5], embeddings, rtol=0, atol=0)
        self.assertEqual(output.state.engram_history, (-1, -1, 1))
        for dtype in (np.uint32, np.uint64):
            unsigned = Model(
                TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
                {"text_config": text, "image_token_id": 2},
                np.arange(3, dtype=dtype),
            )
            with self.subTest(dtype=dtype):
                alternative = unsigned.forward(
                    tokens,
                    endpoint_rows=list(range(6)),
                    token_types=types,
                    image_embeddings=embeddings,
                )
                torch.testing.assert_close(alternative.logits, output.logits)
                self.assertEqual(alternative.state.engram_history, (-1, -1, 1))
        old_window = output.state.layers[0].window_kv.clone()
        continued = model.forward([0], previous=output.state)
        together = model.forward(
            [*tokens, 0],
            token_types=np.append(types, -1),
            image_embeddings=embeddings,
        )
        torch.testing.assert_close(
            continued.logits, together.logits, rtol=1e-13, atol=1e-13
        )
        torch.testing.assert_close(output.state.layers[0].window_kv, old_window)
        torch.testing.assert_close(embeddings, saved, rtol=0, atol=0)
        np.testing.assert_array_equal(types, [-1, 0, 1, 2, 3, -1])
        arrays["layers.0.ffn.gate.bias_vl"] = arrays["layers.0.ffn.gate.bias"].copy()
        ordinary_bias = model.forward(
            tokens,
            endpoint_rows=list(range(6)),
            token_types=types,
            image_embeddings=embeddings,
        )
        torch.testing.assert_close(output.logits[[0, 5]], ordinary_bias.logits[[0, 5]])
        self.assertGreater(
            (output.logits[1:5] - ordinary_bias.logits[1:5]).abs().max().item(), 0.01
        )
        for changed_types, changed_embeddings in (
            (None, embeddings),
            (types, None),
            (types, embeddings.float()),
            (types, embeddings[:3]),
            (types, embeddings * float("nan")),
            (types.astype(float), embeddings),
            ([-1, 0, 1, 2, 4, -1], embeddings),
            ([0, 1, 2, 3, -1, -1], embeddings),
        ):
            with self.subTest(types=changed_types), self.assertRaises(ValueError):
                model.forward(
                    tokens,
                    token_types=changed_types,
                    image_embeddings=changed_embeddings,
                )
        with self.assertRaises(ValueError):
            model.forward(
                [2],
                previous=output.state,
                token_types=[1],
                image_embeddings=embeddings[:1],
            )


if __name__ == "__main__":
    unittest.main()
