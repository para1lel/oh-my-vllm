import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from . import reference
from .reference import write_reference


class ReferenceArtifact(unittest.TestCase):
    def test_source_arguments_reject_unbound_or_unsupported_config(self):
        from .layout import _SUPPORTED_CONFIG
        from .source_capture import cuda_uuid, local_identity, model_arguments

        identities = local_identity()
        repository = Path(__file__).resolve().parents[3]
        for name in ("placement.py", "checkpoint.py"):
            path = f"python/oh_my_vllm/models/deepseek_v41/{name}"
            self.assertEqual(
                identities.get(path),
                hashlib.sha256((repository / path).read_bytes()).hexdigest(),
            )
        identity = SimpleNamespace(bytes=list(range(16)))
        self.assertEqual(
            cuda_uuid(identity), "GPU-" + "00010203-0405-0607-0809-0a0b0c0d0e0f"
        )
        with self.assertRaises(ValueError):
            cuda_uuid(SimpleNamespace(bytes=[0] * 15))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(_SUPPORTED_CONFIG)
            data = json.dumps(config).encode()
            (root / "config.json").write_bytes(data)
            manifest = {
                "verification": "full",
                "files": {
                    "config.json": {
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                },
            }
            verified, values = model_arguments(root, manifest)
            self.assertEqual(verified, config)
            self.assertEqual(
                [values[key] for key in ("dim", "n_layers", "n_heads", "n_mtp_layers")],
                [5120, 40, 64, 3],
            )
            self.assertEqual(
                [
                    values[key]
                    for key in ("max_batch_size", "max_seq_len", "temperature")
                ],
                [1, 144, 0],
            )
            (root / "config.json").write_bytes(data + b" ")
            with self.assertRaisesRegex(ValueError, "identity"):
                model_arguments(root, manifest)
            config["text_config"]["head_dim"] = 128
            data = json.dumps(config).encode()
            (root / "config.json").write_bytes(data)
            manifest["files"]["config.json"] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "head_dim"):
                model_arguments(root, manifest)
            manifest["verification"] = "metadata-only"
            with self.assertRaisesRegex(ValueError, "full"):
                model_arguments(root, manifest)

    def test_source_capture_publishes_full_fp32_diagnostics_exclusively(self):
        from .source_capture import publish_short

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "source.npz"
            expected = {
                key: np.zeros(129280, dtype=np.float64) for key in ("prefill", "decode")
            }
            for value in expected.values():
                value[3] = 1
            vectors = {
                key: value.astype(np.float32) * 2 for key, value in expected.items()
            }
            publish_short(
                output,
                checkpoint,
                {"case": "short"},
                vectors,
                expected,
                verify_unchanged=lambda: None,
            )
            sidecar = json.loads(output.with_suffix(".json").read_bytes())
            with np.load(output, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["decode"], vectors["decode"])
                self.assertEqual(stored["prefill"].dtype, np.float32)
            self.assertEqual(sidecar["metrics"]["prefill"]["nrmse"], 1.0)
            self.assertEqual(sidecar["metrics"]["prefill"]["maxabs"], 1.0)
            self.assertEqual(sidecar["metrics"]["prefill"]["cosine"], 1.0)
            self.assertTrue(sidecar["metrics"]["prefill"]["top1_equal"])
            self.assertEqual(
                sidecar["npz_sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            with self.assertRaises(FileExistsError):
                publish_short(
                    output,
                    checkpoint,
                    {},
                    vectors,
                    expected,
                    verify_unchanged=lambda: None,
                )
            with self.assertRaises(ValueError):
                publish_short(
                    root / "bad.npz",
                    checkpoint,
                    {},
                    dict(vectors, decode=np.zeros(5, dtype=np.float32)),
                    expected,
                    verify_unchanged=lambda: None,
                )
            self.assertFalse((root / "bad.npz").exists())

            def changed():
                raise ValueError("source changed")

            with self.assertRaisesRegex(ValueError, "source changed"):
                publish_short(
                    root / "changed.npz",
                    checkpoint,
                    {},
                    vectors,
                    expected,
                    verify_unchanged=changed,
                )
            self.assertFalse((root / "changed.npz").exists())
            self.assertFalse((root / "changed.json").exists())

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
