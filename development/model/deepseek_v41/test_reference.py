import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .reference import write_reference


class ReferenceArtifact(unittest.TestCase):
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
