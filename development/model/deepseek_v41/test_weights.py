import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import identity, weights
from .test_identity import write_shard
from .test_layout import checkpoint_config


def reference_fixture(root, extra=None, config=None):
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
    arrays.update(extra or {})
    layout, header, payload = {}, {}, bytearray()
    for name, (dtype, array) in arrays.items():
        layout[name] = {"dtype": dtype, "shape": list(array.shape)}
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = dict(layout[name], data_offsets=[start, len(payload)])
    (root / "config.json").write_text(json.dumps(config or {}))
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
    def test_source_tiles_keep_column_shards_fp4_bytes_and_engram_tail(self):
        from . import source_weights

        with TemporaryDirectory() as directory:
            root, config = Path(directory), checkpoint_config()
            column = np.arange(512, dtype=np.uint16).astype(np.uint8).reshape(4, 128)
            packed = np.arange(-16, 16, dtype=np.int8).reshape(2, 16)
            extra = {
                "embed.weight": ("BF16", np.zeros((8, 2), dtype="<u2")),
                "layers.0.attn.wo_b.weight": ("F8_E4M3", column),
                "layers.0.attn.wo_b.scale": ("F8_E8M0", np.full((1, 4), 127, np.uint8)),
                "layers.0.ffn.experts.288.w1.weight": ("I8", packed),
                "layers.0.ffn.experts.288.w1.scale": (
                    "F8_E8M0",
                    np.full((2, 1), 127, np.uint8),
                ),
                "layers.1.engram.embed.weight": (
                    "F8_E4M3",
                    np.full((7, 32), 0x38, np.uint8),
                ),
                "layers.1.engram.embed.scale": (
                    "F8_E8M0",
                    np.full((7, 1), 129, np.uint8),
                ),
            }
            layout, manifest = reference_fixture(root, extra, config)
            with patch.object(source_weights, "tensor_layout", return_value=layout):
                reader = source_weights.SourceWeights(
                    root, manifest, config, 3, read_bytes=64
                )
            expected = {
                "layers.0.attn.wo_b.weight": column[:, 96:],
                "layers.0.ffn.experts.288.w1.weight": packed,
                "layers.1.engram.embed.weight": np.array(
                    [[0x38] * 32, [0] * 32], np.uint8
                ),
                "layers.1.engram.embed.scale": np.array([[129], [127]], np.uint8),
            }
            self.assertNotIn("layers.0.ffn.experts.0.w1.weight", reader.parameters)
            for name, wanted in expected.items():
                tiles = list(reader.tiles(name))
                self.assertTrue(all(tile.nbytes <= 64 for _, tile in tiles))
                np.testing.assert_array_equal(
                    np.concatenate([tile for _, tile in tiles]), wanted
                )

    def test_source_wo_a_uses_global_scale_rows_and_validates_destinations(self):
        from . import source_weights

        with TemporaryDirectory() as directory:
            root, config = Path(directory), checkpoint_config()
            extra = {
                "embed.weight": ("BF16", np.zeros((8, 2), dtype="<u2")),
                "layers.0.attn.wo_a.weight": (
                    "F8_E4M3",
                    np.full((128, 64), 0x38, np.uint8),
                ),
                "layers.0.attn.wo_a.scale": (
                    "F8_E8M0",
                    np.arange(127, 135, dtype=np.uint8).reshape(4, 2),
                ),
            }
            layout, manifest = reference_fixture(root, extra, config)
            with patch.object(source_weights, "tensor_layout", return_value=layout):
                reader = source_weights.SourceWeights(
                    root, manifest, config, 1, read_bytes=64
                )
            name = "layers.0.attn.wo_a.weight"
            tiles = list(reader.tiles(name))
            self.assertNotIn(name.replace("weight", "scale"), reader.parameters)
            np.testing.assert_array_equal(
                np.concatenate([tile for _, tile in tiles]),
                np.array([[0x4080] * 32 + [0x4100] * 32] * 32, dtype="<u2"),
            )
            self.assertEqual([row for row, _ in tiles], list(range(32)))
            with self.assertRaisesRegex(ValueError, "parameters"):
                reader.load_into(SimpleNamespace(named_parameters=lambda: []))
            destinations = {
                key: SimpleNamespace(
                    shape=spec["shape"],
                    dtype="torch." + source_weights.TORCH_DTYPES[spec["dtype"]],
                )
                for key, spec in reader.parameters.items()
            }
            model = SimpleNamespace(named_parameters=lambda: destinations.items())
            for field, invalid in (("shape", (1,)), ("dtype", "torch.float64")):
                original = getattr(destinations[name], field)
                setattr(destinations[name], field, invalid)
                with self.assertRaisesRegex(ValueError, "shape/dtype"):
                    reader.load_into(model)
                setattr(destinations[name], field, original)
            with (root / "model.safetensors").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "changed"):
                next(reader.tiles(name))

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
