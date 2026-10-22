import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import reference_long_case as case


class LongCaseReference(unittest.TestCase):
    def test_publication_rejects_partial_vectors_and_changed_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            output = root.parent / "long.npz"
            arrays = {
                name: np.arange(129280, dtype=np.float64) for name in case.ARRAY_SHAPES
            }
            metadata = {"test": "full-vocabulary"}
            invalid = [dict(arrays, prefill_16385=np.zeros(3))]
            invalid.append(
                {
                    name: value
                    for name, value in arrays.items()
                    if name != "decode_65537"
                }
            )
            invalid.append(dict(arrays, prefill_16385=np.full(129280, np.inf)))
            invalid.append(
                dict(arrays, prefill_16385=np.zeros(129280, dtype=np.float32))
            )
            for values in invalid:
                with self.assertRaises(ValueError):
                    case.write_case(
                        output, root, metadata, values, validate_inputs=lambda: None
                    )
                self.assertFalse(output.exists())

            serialized = False
            original = np.savez_compressed

            def archive(*args, **kwargs):
                nonlocal serialized
                original(*args, **kwargs)
                serialized = True

            def reject_changed():
                if serialized:
                    raise ValueError("input changed during serialization")

            with (
                patch.object(np, "savez_compressed", side_effect=archive),
                self.assertRaisesRegex(ValueError, "changed during"),
            ):
                case.write_case(
                    output, root, metadata, arrays, validate_inputs=reject_changed
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.parent.iterdir()), [root])
            case.write_case(
                output, root, metadata, arrays, validate_inputs=lambda: None
            )
            with np.load(output, allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), {"metadata", *arrays})
                self.assertEqual(json.loads(stored["metadata"].item()), metadata)
                for name, values in arrays.items():
                    np.testing.assert_array_equal(stored[name], values)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                case.write_case(
                    output, root, metadata, arrays, validate_inputs=lambda: None
                )
            self.assertEqual(output.read_bytes(), before)


class LongCaseMath(unittest.TestCase):
    def test_chunked_endpoints_do_not_accept_probe_decode_tokens(self):
        from .reference_model import Model, TorchWeights
        from .test_reference_model import ArrayWeights, add_engram, backbone_fixture

        text, arrays = backbone_fixture()
        text["max_position_embeddings"] = 16
        add_engram(text, arrays)
        # Nonzero attention makes a wrong continuation history affect logits.
        for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
            arrays[f"layers.0.attn.{name}.weight"] = np.eye(2)
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
            {"text_config": text},
            np.arange(3),
        )
        tokens = (0, 1, 2, 1, 0, 2, 0, 1, 2)
        checkpoints = (5, 9)
        expected = {}
        for length in checkpoints:
            prefill = model.forward(tokens[:length])
            token = int(prefill.logits[0].argmax())
            decode = model.forward([token], previous=prefill.state)
            expected[length] = (
                token,
                prefill.logits[0].numpy(),
                decode.logits[0].numpy(),
            )
        for chunk_size in (1, 4, 9):
            inputs, result = case.long_endpoints(
                model, tokens, checkpoints=checkpoints, chunk_size=chunk_size
            )
            self.assertEqual(inputs["tokens"], list(tokens))
            self.assertEqual(inputs["checkpoints"], [5, 9])
            for length, (token, prefill, decode) in expected.items():
                self.assertEqual(inputs["decode_tokens"][str(length)], token)
                np.testing.assert_allclose(
                    result[f"prefill_{length}"], prefill, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    result[f"decode_{length}"], decode, rtol=1e-12, atol=1e-12
                )
        for checkpoints, chunk_size in (
            ((9, 5), 4),
            ((5, 5), 4),
            ((0, 9), 4),
            ((5, 10), 4),
            ((5, 8), 4),
            ((True, 9), 4),
            ((5, 9), 0),
            ((5, 9), True),
            ((5, 9), 4097),
        ):
            with (
                self.subTest(checkpoints=checkpoints, chunk_size=chunk_size),
                self.assertRaises(ValueError),
            ):
                case.long_endpoints(
                    model, tokens, checkpoints=checkpoints, chunk_size=chunk_size
                )
