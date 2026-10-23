import copy
import hashlib
import json
import os
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


def shared_fixture(root):
    config = checkpoint_config()
    config["text_config"]["engram_num_embeddings"] = [3, 5]
    extra = {
        "layers.1.engram.embed.weight": ("F8_E4M3", np.full((3, 256), 0x38, np.uint8)),
        "layers.1.engram.embed.scale": ("F8_E8M0", np.full((3, 8), 128, np.uint8)),
        "layers.14.engram.embed.weight": ("F8_E4M3", np.full((5, 256), 0xBC, np.uint8)),
        "layers.14.engram.embed.scale": ("F8_E8M0", np.full((5, 8), 127, np.uint8)),
    }
    _, manifest = reference_fixture(root, extra, config)
    return manifest["files"]


class SharedEngramWeights(unittest.TestCase):
    def test_load_keeps_encoded_ranges_and_seals_one_bounded_store(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram

        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = shared_fixture(root)
            with load_engram(root, files, limit=4096, read_bytes=17) as loaded:
                self.assertTrue(loaded.shared.sealed)
                self.assertEqual(loaded.shared.size, 2112)
                self.assertEqual(loaded.shared.mapped_bytes, 4096)
                self.assertEqual(set(loaded.tables), {1, 14})
                for layer, rows, weight, scale in (
                    (1, 3, 0x38, 128),
                    (14, 5, 0xBC, 127),
                ):
                    table = loaded.tables[layer]
                    self.assertEqual(table.rows, rows)
                    self.assertEqual(
                        os.pread(
                            loaded.shared.fileno(), rows * 256, table.table_offset
                        ),
                        bytes([weight]) * (rows * 256),
                    )
                    self.assertEqual(
                        os.pread(loaded.shared.fileno(), rows * 8, table.scale_offset),
                        bytes([scale]) * (rows * 8),
                    )
                with self.assertRaises(OSError):
                    os.pwrite(loaded.shared.fileno(), b"x", 0)
                descriptor = os.dup(loaded.shared.fileno())
            try:
                self.assertEqual(os.fstat(descriptor).st_size, 4096)
                with self.assertRaises(RuntimeError):
                    loaded.shared.fileno()
            finally:
                os.close(descriptor)

    def test_rejects_unverified_changed_and_invalid_checkpoint_bytes(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram

        for mode in ("digest", "payload", "truncated", "shape", "index", "link", "cap"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                files = shared_fixture(root)
                shard = root / "model.safetensors"
                if mode == "digest":
                    files[shard.name]["sha256"] = None
                elif mode == "payload":
                    with shard.open("r+b") as source:
                        source.seek(-1, 2)
                        source.write(b"\xff")
                elif mode == "truncated":
                    shard.write_bytes(shard.read_bytes()[:-1])
                elif mode in ("shape", "index"):
                    name = (
                        "config.json"
                        if mode == "shape"
                        else "model.safetensors.index.json"
                    )
                    path = root / name
                    data = json.loads(path.read_text())
                    if mode == "shape":
                        data["text_config"]["engram_num_embeddings"][0] = 4
                    else:
                        data["weight_map"]["layers.1.engram.embed.scale"] = "../outside"
                    path.write_text(json.dumps(data))
                    files[name] = {
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                elif mode == "link":
                    shard.rename(root / "renamed")
                    shard.symlink_to("renamed")
                with self.assertRaises((ValueError, OSError)):
                    load_engram(root, files, limit=2112 if mode == "cap" else 4096)

    def test_interrupted_or_changed_load_closes_partial_shared_storage(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram
        from oh_my_vllm.operators.engram.shared import SharedWeightFile

        append = SharedWeightFile.append
        for mode in ("interrupt", "change"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                files = shared_fixture(root)
                owners = []

                def inject(owner, data, owners=owners, mode=mode, root=root):
                    result = append(owner, data)
                    if not owners:
                        owners.append(owner)
                        if mode == "interrupt":
                            raise KeyboardInterrupt
                        path = root / "config.json"
                        path.write_bytes(path.read_bytes() + b" ")
                    return result

                with (
                    patch.object(SharedWeightFile, "append", new=inject),
                    self.assertRaises(
                        KeyboardInterrupt if mode == "interrupt" else ValueError
                    ),
                ):
                    load_engram(root, files, limit=4096, read_bytes=17)
                self.assertEqual(len(owners), 1)
                self.assertIsNone(owners[0].fd)


class EncodedCheckpointWeights(unittest.TestCase):
    def test_column_slices_and_vectors_keep_raw_bits_with_bounded_reads(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        with TemporaryDirectory() as directory:
            root = Path(directory)
            extra = {
                "columns": ("F8_E4M3", np.arange(32, dtype=np.uint8).reshape(4, 8)),
                "packed": ("I8", np.array([[-128, -1, 0, 127]], np.int8)),
                "bits": ("BF16", np.array([0, 0x8000, 0x7FC1, 0x3F80], "<u2")),
                "float": ("F32", np.array([1.25, -0.0, -2.5], "<f4")),
                "scale": ("F8_E8M0", np.array([0, 127, 255], np.uint8)),
            }
            _, manifest = reference_fixture(root, extra)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["nonowned"] = "absent.safetensors"
            index_path.write_text(json.dumps(index))
            manifest["files"][index_path.name] = {
                "size": index_path.stat().st_size,
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            }
            selections = {
                "columns": TensorSlice("F8_E4M3", (4, 8), (1, 2), (4, 4)),
                "packed": TensorSlice("I8", (1, 4), (0, 0), (1, 4)),
                "bits": TensorSlice("BF16", (4,), (1,), (4,)),
                "float": TensorSlice("F32", (3,), (0,), (3,)),
                "scale": TensorSlice("F8_E8M0", (3,), (0,), (3,)),
            }
            with EncodedReader(
                root, manifest["files"], selections, read_bytes=17
            ) as reader:
                output = np.full((4, 2), 90, np.uint8)
                self.assertEqual(reader.read_into("columns", 0, output), 3)
                np.testing.assert_array_equal(
                    output, [[10, 11], [18, 19], [26, 27], [90, 90]]
                )
                self.assertEqual(reader.read_into("columns", 1, output[:2]), 2)
                np.testing.assert_array_equal(output[:2], [[18, 19], [26, 27]])
                self.assertEqual(reader.read_into("columns", 3, output), 0)
                for name, wanted in (
                    ("packed", np.array([[-128, -1, 0, 127]], np.int8)),
                    ("bits", np.array([0x8000, 0x7FC1, 0x3F80], "<u2")),
                    ("float", np.array([1.25, -0.0, -2.5], "<f4")),
                    ("scale", np.array([0, 127, 255], np.uint8)),
                ):
                    actual = np.empty_like(wanted)
                    self.assertEqual(reader.read_into(name, 0, actual), len(wanted))
                    self.assertEqual(actual.tobytes(), wanted.tobytes())
                verified = reader.finish()
                self.assertEqual(
                    set(verified),
                    {"config.json", index_path.name, "model.safetensors"},
                )
                for name, digest in verified.items():
                    self.assertEqual(digest, manifest["files"][name]["sha256"])
            with self.assertRaises(RuntimeError):
                reader.finish()

    def test_rejects_invalid_selections_destinations_and_tile_limits(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        for args in (
            ("F16", (2,), (0,), (2,)),
            ("BF16", (True,), (0,), (1,)),
            ("BF16", (2,), (-1,), (1,)),
            ("BF16", (2,), (0,), (3,)),
            ("BF16", (2, 2), (0,), (2,)),
            ("BF16", (2, 2, 2), (0, 0, 0), (2, 2, 2)),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TensorSlice(*args)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = reference_fixture(root)
            selection = {"embed.weight": TensorSlice("BF16", (2, 2), (0, 1), (2, 2))}
            for cap in (0, True, (64 << 20) + 1):
                with self.assertRaises(ValueError):
                    EncodedReader(root, manifest["files"], selection, read_bytes=cap)
            with EncodedReader(
                root, manifest["files"], selection, read_bytes=3
            ) as reader:
                readonly = np.zeros((2, 1), "<u2")
                readonly.flags.writeable = False
                for destination in (
                    np.empty((2, 1), np.uint8),
                    np.empty((2, 3), "<u2"),
                    np.empty((2, 2), "<u2")[:, :1],
                    readonly,
                    np.empty((2,), "<u2"),
                ):
                    with self.assertRaises(ValueError):
                        reader.read_into("embed.weight", 0, destination)
                for row in (-1, 3, True):
                    with self.assertRaises(ValueError):
                        reader.read_into("embed.weight", row, np.empty((1, 1), "<u2"))
                with self.assertRaises(ValueError):
                    reader.read_into("nonowned", 0, np.empty((1, 1), "<u2"))
                raw = np.empty((2, 1), "<u2")
                self.assertEqual(reader.read_into("embed.weight", 0, raw), 2)
                np.testing.assert_array_equal(raw, [[0xC000], [0x3F00]])
            for kind, shape in (("BF16", (4, 2)), ("F32", (2, 2))):
                wrong = {"embed.weight": TensorSlice(kind, shape, (0, 0), (2, 2))}
                with (
                    EncodedReader(root, manifest["files"], wrong) as reader,
                    self.assertRaises(ValueError),
                ):
                    reader.read_into(
                        "embed.weight",
                        0,
                        np.empty((2, 2), "<f4" if kind == "F32" else "<u2"),
                    )

    def test_verifies_unselected_payload_before_copy_and_rechecks_controls(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        for mode in (
            "digest",
            "config.json",
            "model.safetensors.index.json",
            "model.safetensors",
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest = reference_fixture(root)
                selection = {
                    "embed.weight": TensorSlice("BF16", (2, 2), (0, 0), (2, 2))
                }
                if mode == "digest":
                    with (root / "model.safetensors").open("r+b") as stream:
                        stream.seek(-1, 2)
                        stream.write(b"\xff")
                with EncodedReader(
                    root, manifest["files"], selection, read_bytes=17
                ) as reader:
                    output = np.full((2, 2), 0x5A5A, "<u2")
                    if mode == "digest":
                        with self.assertRaisesRegex(ValueError, "SHA256"):
                            reader.read_into("embed.weight", 0, output)
                        self.assertTrue(np.all(output == 0x5A5A))
                        with self.assertRaisesRegex(ValueError, "verification"):
                            reader.finish()
                    else:
                        reader.read_into("embed.weight", 0, output)
                        path = root / mode
                        path.write_bytes(path.read_bytes() + b" ")
                        with self.assertRaisesRegex(ValueError, "changed"):
                            reader.read_into("embed.weight", 0, output)
                        with self.assertRaisesRegex(ValueError, "changed"):
                            reader.finish()


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
