import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import reference_cases as cases


class Target:
    def __init__(self):
        self.calls = []

    def forward(self, tokens, *, previous=None, endpoint_rows=None):
        start = 0 if previous is None else previous.length
        self.calls.append((tuple(tokens), start, endpoint_rows))
        features = np.arange(start, start + len(tokens), dtype=np.float64)[:, None]
        selected = [len(tokens) - 1] if endpoint_rows is None else endpoint_rows
        logits = np.zeros((len(selected), 8), dtype=np.float64)
        for row, endpoint in enumerate(selected):
            logits[row, (start + endpoint + 1) % 8] = 1
        return SimpleNamespace(
            features=features,
            logits=logits,
            state=SimpleNamespace(length=start + len(tokens)),
        )


class Predictor:
    def __init__(self):
        self.calls = []

    def append_main(self, features, *, previous=None, start_pos=0):
        if previous is not None:
            assert previous.length == start_pos
        self.calls.append(("append", features[:, 0].tolist(), start_pos))
        return SimpleNamespace(length=start_pos + len(features))

    def project_features(self, features):
        return features

    def forward(self, main, *, start_pos, input_token, previous):
        assert previous.length == start_pos
        assert main.shape == (1, 1) and main[0, 0] == start_pos
        self.calls.append(("proposal", start_pos, input_token))
        return SimpleNamespace(
            proposal_tokens=(1, 2, 3, 4, 5),
            logits=np.tile(np.arange(8, dtype=np.float64), (5, 1)),
            raw_confidence=np.arange(5, dtype=np.float64),
            state=SimpleNamespace(length=start_pos + 1),
        )


class ReferenceCases(unittest.TestCase):
    def test_generator_checks_startup_and_final_identities_before_publication(self):
        for fault in (None, "startup", "checkpoint", "source"):
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                root = Path(directory) / "model"
                root.mkdir()
                config = {"text_config": {"engram_compressed_vocab_size": 3}}
                (root / "config.json").write_text(json.dumps(config))
                (root / "tokenizer.json").write_text("{}")
                manifest = {
                    "verification": "full",
                    "files": {
                        name: {
                            "size": (root / name).stat().st_size,
                            "sha256": hashlib.sha256(
                                (root / name).read_bytes()
                            ).hexdigest(),
                        }
                        for name in ("config.json", "tokenizer.json")
                    },
                }
                manifest_path = root.parent / "manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                output = root.parent / "case.npz"
                sources = {"source": "original"}

                def compute(*args, fault=fault, root=root, sources=sources, **kwargs):
                    if fault == "checkpoint":
                        (root / "tokenizer.json").write_text('{"changed":true}')
                    if fault == "source":
                        sources["source"] = "changed"
                    return {}, {"prefill": np.zeros(1)}

                modules = {
                    "torch": SimpleNamespace(set_num_threads=lambda _: None),
                    "tokenizers": SimpleNamespace(
                        Tokenizer=SimpleNamespace(from_buffer=lambda _: object())
                    ),
                    "development.model.deepseek_v41.engram": SimpleNamespace(
                        compressed_token_map=lambda _: (np.arange(3), 3)
                    ),
                    "development.model.deepseek_v41.reference_model": SimpleNamespace(
                        Model=lambda *args: object(),
                        TorchWeights=lambda *args, **kwargs: object(),
                    ),
                    "development.model.deepseek_v41.reference_dspark": SimpleNamespace(
                        DSpark=lambda *args: object()
                    ),
                }
                versions = {"numpy": "2.5.4", "torch": "2.12.0", "tokenizers": "0.23.3"}
                with (
                    patch.dict("sys.modules", modules),
                    patch.object(
                        cases.importlib.metadata,
                        "version",
                        side_effect=versions.__getitem__,
                    ),
                    patch.object(
                        cases.importlib.metadata, "distributions", return_value=[]
                    ),
                    patch.object(
                        cases,
                        "case_sources",
                        side_effect=lambda sources=sources: dict(sources),
                    ),
                    patch.object(
                        cases,
                        "inspect_checkpoint",
                        return_value={} if fault == "startup" else manifest,
                    ),
                    patch.object(cases, "tensor_layout", return_value={}),
                    patch.object(cases, "Weights", return_value=object()),
                    patch.object(
                        cases, "dspark_restored", side_effect=compute
                    ) as computation,
                    patch.object(
                        cases,
                        "write_case",
                        side_effect=lambda path, *args: path.write_bytes(b"published"),
                    ) as publication,
                ):
                    if fault is None:
                        cases.generate(root, manifest_path, output)
                        publication.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "identity" if fault == "startup" else "changed"
                        ):
                            cases.generate(root, manifest_path, output)
                        publication.assert_not_called()
                        self.assertFalse(output.exists())
                    self.assertEqual(
                        computation.call_count, 0 if fault == "startup" else 1
                    )

    def test_writer_requires_complete_finite_fp64_endpoints_and_new_external_path(self):
        arrays = {
            "prefill": np.zeros(129280),
            "draft_logits": np.zeros((5, 129280)),
            "draft_confidence": np.zeros(5),
            "verify_logits": np.zeros((6, 129280)),
            "restored_logits": np.zeros(129280),
            "next_draft_logits": np.zeros((5, 129280)),
            "next_draft_confidence": np.zeros(5),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model"
            checkpoint.mkdir()
            output = root / "reference.npz"
            cases.write_case(output, checkpoint, {"case": "dspark-restored"}, arrays)
            with np.load(output, allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), {*arrays, "metadata"})
                self.assertEqual(stored["verify_logits"].shape, (6, 129280))
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                cases.write_case(output, checkpoint, {}, arrays)
            self.assertEqual(output.read_bytes(), original)
            dangling = root / "dangling.npz"
            missing = root / "missing.npz"
            dangling.symlink_to(missing)
            with self.assertRaises(FileExistsError):
                cases.write_case(dangling, checkpoint, {}, arrays)
            self.assertFalse(missing.exists())
            checked_path = cases.output_path

            def add_competing_link(path, model):
                destination = checked_path(path, model)
                destination.symlink_to(missing)
                return destination

            with (
                patch.object(cases, "output_path", side_effect=add_competing_link),
                self.assertRaises(FileExistsError),
            ):
                cases.write_case(root / "racing.npz", checkpoint, {}, arrays)
            self.assertFalse(missing.exists())
            with self.assertRaisesRegex(ValueError, "external"):
                cases.write_case(checkpoint / "reference.npz", checkpoint, {}, arrays)
            for bad in (
                {
                    key: value
                    for key, value in arrays.items()
                    if key != "draft_confidence"
                },
                dict(arrays, verify_logits=np.zeros((5, 129280))),
                dict(arrays, prefill=np.zeros(129280, dtype=np.float32)),
                dict(arrays, next_draft_confidence=np.full(5, np.nan)),
            ):
                with self.assertRaises(ValueError):
                    cases.write_case(root / "bad.npz", checkpoint, {}, bad)
                self.assertFalse((root / "bad.npz").exists())

    def test_verification_forks_two_inputs_and_keeps_pending_boundary_separate(self):
        target, predictor = Target(), Predictor()
        inputs, arrays = cases.dspark_restored(target, predictor, [0, 1, 2, 3])
        self.assertEqual(
            target.calls,
            [
                ((0, 1, 2, 3), 0, None),
                ((4, 1, 2, 3, 4, 5), 4, list(range(6))),
                ((4, 1), 4, [0, 1]),
                ((6,), 6, None),
            ],
        )
        self.assertEqual(
            predictor.calls,
            [
                ("append", [0, 1, 2], 0),
                ("proposal", 3, 4),
                ("append", [4], 4),
                ("proposal", 5, 6),
            ],
        )
        self.assertEqual(inputs["accepted_input_rows"], 2)
        self.assertEqual(inputs["accepted_drafts"], 1)
        self.assertEqual(inputs["correction_token"], 6)
        self.assertEqual(inputs["next_pending_position"], 6)
        self.assertEqual(inputs["verification_tokens"], [4, 1, 2, 3, 4, 5])
        self.assertEqual(arrays["verify_logits"].shape, (6, 8))
        self.assertEqual(int(arrays["restored_logits"].argmax()), 7)
        self.assertEqual(arrays["draft_logits"].shape, (5, 8))
        self.assertEqual(arrays["next_draft_logits"].shape, (5, 8))
        broken = Target()
        original = broken.forward

        def changed_prefix(tokens, **kwargs):
            result = original(tokens, **kwargs)
            if len(tokens) == 2:
                result.features[0, 0] += 1
            return result

        broken.forward = changed_prefix
        with self.assertRaisesRegex(ValueError, "accepted prefix"):
            cases.dspark_restored(broken, Predictor(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
