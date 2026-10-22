import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import identity, weights
from .test_identity import write_shard


def reference_fixture(root):
    arrays = {
        "embed.weight": (
            "BF16",
            np.array([[0x3F80, 0xC000], [0, 0x3F00]], dtype="<u2"),
        ),
        "layers.1.engram.embed.weight": (
            "F8_E4M3",
            np.array([[0x38] * 32, [0x3C] * 32], dtype=np.uint8),
        ),
        "layers.1.engram.embed.scale": (
            "F8_E8M0",
            np.array([[128], [127]], dtype=np.uint8),
        ),
        "layers.0.ffn.experts.0.w1.weight": (
            "I8",
            np.full((2, 16), -98, dtype=np.int8),
        ),
        "layers.0.ffn.experts.0.w1.scale": (
            "F8_E8M0",
            np.array([[127], [128]], dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((33, 64), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127, 128], [129, 130]], dtype=np.uint8),
        ),
    }
    layout, header, payload = {}, {}, bytearray()
    for name, (dtype, array) in arrays.items():
        layout[name] = {"dtype": dtype, "shape": list(array.shape)}
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = dict(layout[name], data_offsets=[start, len(payload)])
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(payload)},
                "weight_map": dict.fromkeys(layout, "model.safetensors"),
            }
        )
    )
    write_shard(root / "model.safetensors", header, payload)
    return layout, identity.scan_checkpoint(root, layout)


class ReferenceWeights(unittest.TestCase):
    def test_packed_and_dense_scales_survive_row_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            reader = weights.Weights(root, manifest, layout)
            # Packed 0x9e stores -4 in the low nibble and -0.5 in the high nibble.
            expert = "layers.0.ffn.experts.0.w1.weight"
            expected = np.array([[-4.0, -0.5] * 16, [-8.0, -1.0] * 16])
            np.testing.assert_array_equal(reader.tensor(expert), expected)
            np.testing.assert_array_equal(
                reader.tensor(expert, rows=[1, 0, 1]), expected[[1, 0, 1]]
            )
            dense = "layers.0.attn.wq_a.weight"
            expected = np.array(
                [[1.0] * 32 + [2.0] * 32] * 32 + [[4.0] * 32 + [8.0] * 32]
            )
            np.testing.assert_array_equal(reader.tensor(dense), expected)
            np.testing.assert_array_equal(
                reader.tensor(dense, rows=[32, 0, 31]), expected[[32, 0, 31]]
            )

    def test_row_lookup_decodes_only_requested_engram_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            reader = weights.Weights(root, manifest, layout)
            np.testing.assert_array_equal(
                reader.tensor("embed.weight"), [[1.0, -2.0], [0.0, 0.5]]
            )
            values = reader.tensor("layers.1.engram.embed.weight", rows=[1, 0, 1])
            self.assertEqual(values.dtype, np.float64)
            np.testing.assert_array_equal(values, [[1.5] * 32, [2.0] * 32, [1.5] * 32])
            values[0, 0] = 999
            self.assertEqual(
                reader.tensor("layers.1.engram.embed.weight", rows=[1])[0, 0], 1.5
            )
            with self.assertRaisesRegex(ValueError, "rows"):
                reader.tensor("layers.1.engram.embed.weight")
            for invalid in ([-1], [2], [False], [1.0], 1):
                with self.subTest(rows=invalid), self.assertRaises(ValueError):
                    reader.tensor("layers.1.engram.embed.weight", rows=invalid)

    def test_rejects_unverified_or_changed_weight_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            metadata = copy.deepcopy(manifest)
            metadata["verification"] = "metadata-only"
            with self.assertRaisesRegex(ValueError, "full"):
                weights.Weights(root, metadata, layout)
            for filename in ("config.json", "model.safetensors"):
                partial = copy.deepcopy(manifest)
                del partial["files"][filename]
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    weights.Weights(root, partial, layout)
            reader = weights.Weights(root, manifest, layout)
            reader.tensor("embed.weight")
            with (root / "model.safetensors").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.tensor("embed.weight")
            fresh = weights.Weights(root, manifest, layout)
            with self.assertRaisesRegex(ValueError, "identity|digest"):
                fresh.tensor("embed.weight")
