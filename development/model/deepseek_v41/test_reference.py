import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import reference
from .reference import write_reference


class ReferenceArtifact(unittest.TestCase):
    def test_control_bytes_are_verified_before_parser_or_tokenizer_consumption(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tokenizer.json"
            data = b'{"vocab":"original"}'
            path.write_bytes(data)
            manifest = {
                "files": {
                    "tokenizer.json": {
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                }
            }
            verified = reference.control_bytes(root, "tokenizer.json", manifest)
            self.assertEqual(verified, data)
            path.write_bytes(b'{"vocab":"replaced"}')
            self.assertEqual(verified, data)
            with self.assertRaisesRegex(ValueError, "control.*identity"):
                reference.control_bytes(root, "tokenizer.json", manifest)
            with self.assertRaises(ValueError):
                reference.control_bytes(root, "../tokenizer.json", manifest)

    def test_full_vectors_are_external_finite_and_never_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "reference.npz"
            vectors = {
                "prefill": np.arange(129280, dtype=np.float64),
                "decode": np.zeros(129280, dtype=np.float64),
            }
            metadata = {"schema_version": 1, "case": "short"}
            write_reference(output, checkpoint, metadata, vectors)
            with np.load(output, allow_pickle=False) as stored:
                self.assertEqual(set(stored), {"metadata", "prefill", "decode"})
                self.assertEqual(json.loads(str(stored["metadata"])), metadata)
                np.testing.assert_array_equal(stored["prefill"], vectors["prefill"])
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_reference(output, checkpoint, metadata, vectors)
            self.assertEqual(output.read_bytes(), original)
            dangling = root / "dangling.npz"
            destination = root / "missing.npz"
            dangling.symlink_to(destination)
            with self.assertRaises(FileExistsError):
                write_reference(dangling, checkpoint, metadata, vectors)
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(ValueError, "external"):
                write_reference(
                    checkpoint / "reference.npz", checkpoint, metadata, vectors
                )
            for bad in (
                {"prefill": vectors["prefill"]},
                dict(vectors, decode=np.zeros(5)),
                dict(vectors, decode=np.zeros(129280, dtype=np.float32)),
                dict(vectors, decode=np.full(129280, np.nan)),
            ):
                with self.assertRaises(ValueError):
                    write_reference(root / "bad.npz", checkpoint, metadata, bad)
                self.assertFalse((root / "bad.npz").exists())


if __name__ == "__main__":
    unittest.main()
