import copy
import json
import unittest
import zipfile
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
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def resume_fixture(self, compressed):
        from .reference_model import Model, TorchWeights
        from .test_reference_model import ArrayWeights, add_engram, backbone_fixture

        text, arrays = backbone_fixture()
        text["max_position_embeddings"] = 16
        add_engram(text, arrays)
        if compressed:
            text.update(
                compress_ratios=[2],
                kv_source_layer_ids=[0],
                index_source_layer_ids=[0],
                index_head_dim=2,
                index_n_heads=1,
                index_topk=2,
            )
            for name in (
                "compressor.wkv",
                "compressor.wgate",
                "indexer.wk",
                "indexer.wq_b",
                "indexer.weights_proj",
            ):
                arrays[f"layers.0.attn.{name}.weight"] = (
                    np.ones((1, 2)) if name.endswith("weights_proj") else np.eye(2)
                )
            for name in ("compressor.norm", "indexer.k_norm"):
                arrays[f"layers.0.attn.{name}.weight"] = np.ones(2)
        for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
            arrays[f"layers.0.attn.{name}.weight"] = np.eye(2)
        config = {"text_config": text}
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu", row_limit=2),
            config,
            np.arange(3),
        )
        identity = {
            "config": config,
            "tokens": [0, 1, 2, 1, 0, 2, 0, 1, 2],
            "checkpoints": [5, 9],
            "checkpoint_manifest_sha256": "weights",
            "reference_sources": {"source.py": "source"},
            "dependencies": {"torch": "test"},
            "token_map_sha256": "token-map",
            "execution": {
                "device": "cpu",
                "threads": 1,
                "projection_row_limit": 2,
                "prefill_chunk_size": 3,
            },
        }
        return model, identity

    def test_resume_preserves_chunks_probe_branches_and_all_state_fields(self):
        from .reference_long_checkpoint import Checkpoint

        for compressed in (False, True):
            model, identity = self.resume_fixture(compressed)
            arguments = dict(checkpoints=(5, 9), chunk_size=3)
            expected_inputs, expected = case.long_endpoints(
                model, identity["tokens"], **arguments
            )
            for stop in range(1, 7):
                with (
                    self.subTest(compressed=compressed, stop=stop),
                    TemporaryDirectory() as directory,
                ):
                    store = Checkpoint(Path(directory) / "state.npz", identity)
                    calls = 0

                    def interrupt(point, store=store, stop=stop):
                        nonlocal calls
                        calls += 1
                        store.save(point)
                        if calls == stop:
                            raise InterruptedError("simulated process loss")

                    with self.assertRaises(InterruptedError):
                        case.long_endpoints(
                            model,
                            identity["tokens"],
                            save_checkpoint=interrupt,
                            **arguments,
                        )
                    point = store.load()
                    self.assertEqual(point["state"].length, point["cursor"])
                    layer = point["state"].layers[0]
                    self.assertEqual(layer.pending_values is None, not compressed)
                    self.assertEqual(store.previous.exists(), stop > 1)
                    with zipfile.ZipFile(store.path) as archive:
                        self.assertTrue(
                            all(
                                item.compress_type == zipfile.ZIP_STORED
                                for item in archive.infolist()
                            )
                        )
                    inputs, result = case.long_endpoints(
                        model,
                        identity["tokens"],
                        resume=point,
                        save_checkpoint=store.save,
                        **arguments,
                    )
                    self.assertEqual(inputs, expected_inputs)
                    for name in expected:
                        np.testing.assert_array_equal(result[name], expected[name])
                    completed = store.load()
                    self.assertEqual(completed["phase"], "complete")
                    self.assertEqual(completed["state"].length, 9)

    def test_checkpoint_rejects_identity_corruption_and_failed_publication(self):
        from . import reference_long_checkpoint as checkpoint

        model, identity = self.resume_fixture(True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            store = checkpoint.Checkpoint(path, identity)
            case.long_endpoints(
                model,
                identity["tokens"],
                checkpoints=(5, 9),
                chunk_size=3,
                save_checkpoint=store.save,
            )
            for key in identity:
                changed = copy.deepcopy(identity)
                changed[key] = "different"
                with (
                    self.subTest(identity=key),
                    self.assertRaisesRegex(ValueError, "identity"),
                ):
                    checkpoint.Checkpoint(path, changed).load()
            primary, previous = path.read_bytes(), store.previous.read_bytes()

            def reject():
                raise ValueError("source changed")

            point = store.load()
            with self.assertRaisesRegex(ValueError, "source changed"):
                checkpoint.Checkpoint(path, identity, validate_inputs=reject).save(
                    point
                )
            self.assertEqual(path.read_bytes(), primary)
            self.assertEqual(store.previous.read_bytes(), previous)
            with np.load(path, allow_pickle=False) as saved:
                values = {name: saved[name] for name in saved.files}
            damaged_logits = values["last_logits"].copy()
            damaged_logits[0] += 1
            corruptions = [
                {"last_logits": damaged_logits},
                {"metadata": np.asarray(1)},
                {"metadata_sha256": np.asarray("incorrect")},
                {"last_logits": np.asarray([object()], dtype=object)},
            ]
            for damaged in corruptions:
                with self.subTest(corruption=next(iter(damaged))):
                    with path.open("wb") as output:
                        np.savez(output, **dict(values, **damaged))
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        store.load()
                    with self.assertRaisesRegex(
                        ValueError, "changed before replacement"
                    ):
                        store.save(point)
                    self.assertEqual(store.previous.read_bytes(), previous)
            path.write_bytes(primary[: len(primary) // 2])
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                store.load()
            recovered = checkpoint.Checkpoint(store.previous, identity).load()
            self.assertEqual(recovered["state"].length, 9)
            path.write_bytes(primary)
            point = store.load()
            original_replace = checkpoint.os.replace

            def reject_install(source, destination):
                if destination == store.path:
                    raise OSError("interrupted replacement")
                return original_replace(source, destination)

            with (
                patch.object(checkpoint.os, "replace", side_effect=reject_install),
                self.assertRaisesRegex(OSError, "interrupted"),
            ):
                store.save(point)
            self.assertEqual(path.read_bytes(), primary)
            self.assertEqual(
                checkpoint.Checkpoint(store.previous, identity).load()["cursor"], 9
            )
            path.unlink()
            path.symlink_to(store.previous)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                store.load()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                store.save(point)
            self.assertEqual(
                set(Path(directory).iterdir()), {store.path, store.previous}
            )

    def test_chunked_endpoints_do_not_accept_probe_decode_tokens(self):
        model, identity = self.resume_fixture(False)
        tokens, checkpoints = identity["tokens"], identity["checkpoints"]
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
