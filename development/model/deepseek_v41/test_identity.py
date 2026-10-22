import hashlib
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from . import identity

LAYOUT = {
    "embed.weight": {"dtype": "BF16", "shape": [2, 2]},
    "layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [1, 2]},
    "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [1, 1]},
}


def write_shard(path, header, payload):
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def fixture(root):
    (root / "config.json").write_text('{"model_type":"fixture"}')
    (root / "tokenizer.json").write_text('{"fixture":true}')
    (root / "tokenizer_config.json").write_text("{}")
    header = {}
    offset = 0
    for name, description in LAYOUT.items():
        size = 8 if name == "embed.weight" else 1 if name.endswith("scale") else 2
        header[name] = dict(description, data_offsets=[offset, offset + size])
        offset += size
    index = {
        "metadata": {"total_size": 11},
        "weight_map": dict.fromkeys(LAYOUT, "model.safetensors"),
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    write_shard(root / "model.safetensors", header, bytes(range(11)))
    return header, index


class CheckpointIdentity(unittest.TestCase):
    def test_manifest_binds_payload_and_separates_host_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            full = identity.scan_checkpoint(root, LAYOUT)
            self.assertEqual(full["verification"], "full")
            self.assertEqual(full["tensor_count"], 3)
            self.assertEqual(full["tensor_bytes"], 11)
            self.assertEqual(full["component_bytes"], {"target": 8, "engram": 3})
            shard = root / "model.safetensors"
            self.assertEqual(
                full["files"][shard.name]["sha256"],
                hashlib.sha256(shard.read_bytes()).hexdigest(),
            )
            with shard.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\xff")
            changed = identity.scan_checkpoint(root, LAYOUT)
            self.assertNotEqual(
                full["files"][shard.name]["sha256"],
                changed["files"][shard.name]["sha256"],
            )
            metadata = identity.scan_checkpoint(root, LAYOUT, hash_weights=False)
            self.assertEqual(metadata["verification"], "metadata-only")
            self.assertIsNone(metadata["files"][shard.name]["sha256"])
            self.assertIsNotNone(metadata["files"]["config.json"]["sha256"])

    def test_rejects_invalid_tensor_storage_and_layout(self):
        changes = (
            ("dtype", "F32"),
            ("dtype", "UNKNOWN"),
            ("shape", [2, 3]),
            ("shape", [True, 4]),
            ("shape", [-1, 4]),
            ("data_offsets", [0, 7]),
            ("data_offsets", [1, 9]),
            ("data_offsets", [0, 999]),
            ("data_offsets", [False, 8]),
        )
        for field, value in changes:
            with (
                self.subTest(field=field, value=value),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                header, _ = fixture(root)
                header["embed.weight"][field] = value
                write_shard(root / "model.safetensors", header, bytes(range(11)))
                with self.assertRaises(ValueError):
                    identity.scan_checkpoint(root, LAYOUT)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            shard = root / "model.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload|length|size|offset"):
                identity.scan_checkpoint(root, LAYOUT)

    def test_rejects_incomplete_or_escaping_index(self):
        for fault in ("missing", "extra", "size", "escape", "absolute", "symlink"):
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                root = Path(directory)
                _, index = fixture(root)
                if fault == "missing":
                    del index["weight_map"]["embed.weight"]
                elif fault == "extra":
                    index["weight_map"]["unknown"] = "model.safetensors"
                elif fault == "size":
                    index["metadata"]["total_size"] = 12
                elif fault == "symlink":
                    (root / "alias.safetensors").symlink_to("model.safetensors")
                    index["weight_map"]["embed.weight"] = "alias.safetensors"
                else:
                    index["weight_map"]["embed.weight"] = (
                        "../outside.safetensors"
                        if fault == "escape"
                        else "/outside.safetensors"
                    )
                (root / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    identity.scan_checkpoint(root, LAYOUT)

    def test_rejects_duplicate_json_keys_and_trailing_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            index = root / "model.safetensors.index.json"
            index.write_text('{"weight_map":{},"weight_map":{}}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                identity.scan_checkpoint(root, LAYOUT)
            fixture(root)
            shard = root / "model.safetensors"
            with shard.open("ab") as stream:
                stream.write(b"unused")
            with self.assertRaisesRegex(ValueError, "payload|length|size|offset"):
                identity.scan_checkpoint(root, LAYOUT)
