import hashlib
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from oh_my_vllm.kernels.native import source_digest
from oh_my_vllm.kernels.specification import MODEL
from oh_my_vllm.worker.checkpoint import Checkpoint


def checkpoint_fixture(root):
    config = b'{"fixture":true}'
    (root / "config.json").write_bytes(config)
    weights = {name: f"{name}.safetensors" for name in ("weight", "embedding")}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    for name, filename in weights.items():
        header = json.dumps(
            {name: {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
        ).encode()
        header += b" " * (-len(header) % 8)
        (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")
    return hashlib.sha256(config).hexdigest()


class Identity(unittest.TestCase):
    def test_checkpoint_identity_tracks_loaded_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint_fixture(root)
            with patch.dict(MODEL, {"config_sha256": config}):
                loaded = Checkpoint(root)
                loaded.raw_tensor("weight")
                frozen = loaded.verify_identity()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "weight.safetensors",
                ):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    self.assertEqual(loaded.identity, frozen)
                    with (
                        self.subTest(checkpoint=name),
                        self.assertRaisesRegex(ValueError, "file bytes changed"),
                    ):
                        loaded.verify_identity(frozen)
                    path.write_bytes(original)
                    self.assertEqual(loaded.verify_identity(frozen), frozen)
                loaded.raw_tensor("embedding")
                with self.assertRaisesRegex(ValueError, "loaded checkpoint set"):
                    loaded.verify_identity(frozen)

    def test_native_identity_tracks_only_transitive_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, header, leaf, unrelated = (
                root / name for name in ("op.cu", "shared.cuh", "leaf.cuh", "other.cuh")
            )
            source.write_text('#include "shared.cuh"\n')
            header.write_text('#include "leaf.cuh"\n')
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 128\n')
            unrelated.write_text("unrelated\n")
            identity = source_digest(source)
            unrelated.write_text("changed\n")
            self.assertEqual(source_digest(source), identity)
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 256\n')
            self.assertNotEqual(source_digest(source), identity)
            leaf.unlink()
            with self.assertRaises(FileNotFoundError):
                source_digest(source)
