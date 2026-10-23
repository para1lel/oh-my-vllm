import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import fixtures


class FrozenFixtures(unittest.TestCase):
    def test_media_reference_keeps_complete_inputs_vectors_and_historical_provenance(
        self,
    ):
        self.assertTrue(
            callable(getattr(fixtures, "load_media", None)),
            "media fixture loader is missing",
        )
        metadata, arrays = fixtures.load_media()
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["case"], "still-and-video-frames")
        self.assertEqual(
            metadata["reference_kind"],
            "independent-pytorch-fp64-no-activation-quantization",
        )
        manifest = (
            json.dumps(metadata["checkpoint"], sort_keys=True, indent=2) + "\n"
        ).encode()
        self.assertEqual(
            hashlib.sha256(manifest).hexdigest(),
            "0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67",
        )
        self.assertEqual(metadata["checkpoint"]["verification"], "full")
        self.assertEqual(metadata["checkpoint"]["tensor_count"], 96085)
        self.assertEqual(metadata["checkpoint"]["shard_count"], 48)
        self.assertEqual(
            metadata["execution"],
            {"device": "cpu", "threads": 8, "projection_row_limit": 2048},
        )
        for key, expected in (
            (
                "inputs",
                "968e863392fe7ec3531e4df6a7d43851a295f070c486fec80565abc63a7449bc",
            ),
            (
                "reference_sources",
                "c099eb57db022052006567e726dbd3cd4f56c5bebd35eaf7bce45a3ed2383dd7",
            ),
        ):
            canonical = json.dumps(
                metadata[key], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected)
        self.assertEqual(len(metadata["reference_sources"]), 21)
        self.assertEqual(
            metadata["reference_sources"][
                "development/model/deepseek_v41/reference_media_case.py"
            ],
            "e3adcac187cee11c76e05a436d690ad8f92aef9b172da7b35b048624aec58038",
        )
        self.assertEqual(
            metadata["token_map_sha256"],
            "26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496",
        )
        inputs = metadata["inputs"]
        self.assertEqual(len(inputs["placeholder_tokens"]), 37)
        self.assertEqual(len(inputs["expanded_tokens"]), 624)
        self.assertEqual(len(inputs["token_types"]), 624)
        self.assertEqual(inputs["decode_token"], 9544)
        self.assertEqual(
            inputs["video"],
            {
                "fps": 2.0,
                "frame_count": 2,
                "selected_indices": [0, 1],
                "scope": "synthetic-decoded-frames; no video decoder",
            },
        )
        self.assertEqual(
            [value["shape"] for value in inputs["vision_embeddings"]],
            [[200, 5120], [206, 5120], [184, 5120]],
        )
        expected_hashes = {
            "prefill": (
                "7edd90ed9122b9dab48dffbcc231fd06a284591bc4f04cfbf65ecf2384edfc38"
            ),
            "decode": (
                "b97f1fd3e53a8e90145a5db261ffa44b78228beb427f337145836336422b6c31"
            ),
        }
        self.assertEqual(set(arrays), set(expected_hashes))
        self.assertEqual(set(metadata["arrays"]), set(expected_hashes))
        for name, expected in expected_hashes.items():
            value = arrays[name]
            self.assertEqual(value.shape, (129280,))
            self.assertEqual(value.dtype, np.float64)
            self.assertTrue(np.isfinite(value).all())
            self.assertEqual(
                hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(), expected
            )
            self.assertEqual(
                metadata["arrays"][name],
                {"shape": [129280], "dtype": "float64", "sha256": expected},
            )
        self.assertEqual(int(arrays["prefill"].argmax()), inputs["decode_token"])
        self.assertEqual(int(arrays["decode"].argmax()), 344)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "media.npz"
            original = fixtures.MEDIA_PATH.read_bytes()
            self.assertEqual(len(original), 2012626)
            self.assertEqual(
                hashlib.sha256(original).hexdigest(),
                "a847254bb8a1b858d67eeb7405edb4a5435c24f2280585ed18f6b4b0eee86223",
            )
            path.write_bytes(original)
            copied_metadata, copied_arrays = fixtures.load_media(path)
            self.assertEqual(copied_metadata, metadata)
            for name in arrays:
                np.testing.assert_array_equal(copied_arrays[name], arrays[name])
            for altered in (original[:-1], original + b"extra", b"not an artifact"):
                path.write_bytes(altered)
                with self.assertRaisesRegex(ValueError, "identity"):
                    fixtures.load_media(path)
            for label in (
                "missing-decode",
                "partial-decode",
                "float32",
                "nonfinite",
                "changed-value",
                "array-metadata",
                "checkpoint",
                "inputs",
                "sources",
                "token-map",
            ):
                with self.subTest(altered=label):
                    changed = json.loads(json.dumps(metadata))
                    values = {name: value.copy() for name, value in arrays.items()}
                    if label == "missing-decode":
                        del values["decode"]
                    elif label == "partial-decode":
                        values["decode"] = values["decode"][:128]
                    elif label == "float32":
                        values["decode"] = values["decode"].astype(np.float32)
                    elif label == "nonfinite":
                        values["decode"][0] = np.inf
                    elif label == "changed-value":
                        values["decode"][0] += 1.0
                    elif label == "array-metadata":
                        changed["arrays"]["decode"]["dtype"] = "float32"
                    elif label == "checkpoint":
                        changed["checkpoint"]["tensor_count"] = 1
                    elif label == "inputs":
                        changed["inputs"]["expanded_tokens"][0] += 1
                    elif label == "sources":
                        changed["reference_sources"].pop(
                            "development/model/deepseek_v41/reference_media_case.py"
                        )
                    elif label == "token-map":
                        changed["token_map_sha256"] = "0" * 64
                    np.savez(path, metadata=np.asarray(json.dumps(changed)), **values)
                    with self.assertRaises(ValueError):
                        fixtures.load_media(path)
                    # Exercise content checks after the whole-file identity gate.
                    with (
                        patch.object(
                            fixtures,
                            "MEDIA_SHA256",
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                        ),
                        self.assertRaises(ValueError),
                    ):
                        fixtures.load_media(path)

    def test_dspark_reference_keeps_full_vectors_fixed_state_and_artifact_identity(
        self,
    ):
        metadata, arrays = fixtures.load_dspark()
        self.assertEqual(metadata["case"], "dspark-restored")
        self.assertEqual(metadata["checkpoint"]["tensor_count"], 96085)
        self.assertEqual(metadata["checkpoint"]["shard_count"], 48)
        self.assertEqual(
            metadata["checkpoint_manifest_sha256"],
            "0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67",
        )
        self.assertEqual(
            metadata["reference_sources"][
                "development/model/deepseek_v41/reference_model.py"
            ],
            "ab57b03c36d2d3ffaf2304fb9cc772aad0b80fb673cffe7c4405593d540e7712",
        )
        self.assertEqual(
            metadata["state_selection"],
            "forced-two-input-prefix; correction is target argmax, "
            "not sampled rejection",
        )
        self.assertEqual(
            metadata["inputs"],
            {
                "tokens": [
                    671,
                    4787,
                    13769,
                    46012,
                    54994,
                    1060,
                    270,
                    41638,
                    6397,
                    16,
                    28231,
                    3939,
                    270,
                    12709,
                    344,
                    8295,
                ]
                * 8
                + [671],
                "pending_token": 4787,
                "proposal_tokens": [13769, 46012, 54994, 1060, 270],
                "verification_tokens": [4787, 13769, 46012, 54994, 1060, 270],
                "accepted_input_rows": 2,
                "accepted_drafts": 1,
                "correction_token": 46012,
                "next_pending_position": 131,
                "next_proposal_tokens": [54994, 1060, 270, 41638, 6397],
            },
        )
        shapes = {
            "prefill": (129280,),
            "draft_logits": (5, 129280),
            "draft_confidence": (5,),
            "verify_logits": (6, 129280),
            "restored_logits": (129280,),
            "next_draft_logits": (5, 129280),
            "next_draft_confidence": (5,),
        }
        self.assertEqual(set(arrays), set(shapes))
        self.assertEqual(set(metadata["arrays"]), set(shapes))
        for name, shape in shapes.items():
            value = arrays[name]
            self.assertEqual(value.shape, shape)
            self.assertEqual(value.dtype, np.float64)
            self.assertTrue(np.isfinite(value).all())
            self.assertEqual(
                metadata["arrays"][name],
                {
                    "shape": list(shape),
                    "dtype": "float64",
                    "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
                },
            )
        self.assertEqual(int(arrays["prefill"].argmax()), 4787)
        self.assertEqual(
            arrays["draft_logits"].argmax(axis=1).tolist(),
            [13769, 46012, 54994, 1060, 270],
        )
        self.assertEqual(int(arrays["verify_logits"][1].argmax()), 46012)
        self.assertEqual(
            arrays["next_draft_logits"].argmax(axis=1).tolist(),
            [54994, 1060, 270, 41638, 6397],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dspark.npz"
            original = fixtures.DSPARK_PATH.read_bytes()
            self.assertEqual(
                hashlib.sha256(original).hexdigest(),
                "c78d89d66d2ff38999af2aa44b845c6d2fea1702280c217a92e7a0d693ee6fa8",
            )
            path.write_bytes(original)
            copied, _ = fixtures.load_dspark(path)
            self.assertEqual(copied, metadata)
            for altered in (original[:-1], original + b"extra", b"not an artifact"):
                path.write_bytes(altered)
                with self.assertRaisesRegex(ValueError, "identity"):
                    fixtures.load_dspark(path)

    def test_codec_aliases_and_canonical_tool_images_keep_full_token_ids(self):
        artifact = fixtures.load_codec()
        cases = {case["name"]: case for case in artifact["cases"]}
        self.assertEqual(len(cases), 20)
        self.assertEqual(
            cases["plain-chat"]["token_ids"],
            [
                0,
                128799,
                19905,
                24752,
                16,
                128803,
                3085,
                344,
                223,
                20,
                940,
                223,
                20,
                33,
                128804,
                128822,
            ],
        )
        for alias, number in (
            ("low", "50"),
            ("high", "75"),
            ("max", "100"),
            ("default", "75"),
        ):
            self.assertEqual(
                cases[f"thinking-{alias}"]["token_ids"],
                cases[f"thinking-{number}"]["token_ids"],
            )
        self.assertNotIn("earlier-reasoning", cases["reasoning-drop"]["rendered_text"])
        self.assertIn("earlier-reasoning", cases["reasoning-keep"]["rendered_text"])
        self.assertIn("earlier-reasoning", cases["tools-ordered"]["rendered_text"])
        self.assertNotEqual(cases["context-continuation"]["token_ids"][0], 0)
        raw, canonical = (
            cases["tool-result-images"],
            cases["tool-result-images-canonical"],
        )
        self.assertEqual(raw["token_ids"], canonical["token_ids"])
        self.assertEqual(raw["rendered_text"], canonical["rendered_text"])
        self.assertLess(
            raw["rendered_text"].index("first-result"),
            raw["rendered_text"].index("second-result"),
        )
        self.assertEqual(
            [image["url"] for image in raw["media"]["images"]],
            ["fixture://second", "fixture://first"],
        )
        self.assertEqual(
            [image["url"] for image in canonical["media"]["images"]],
            ["fixture://first", "fixture://second"],
        )
        self.assertEqual(canonical["token_ids"].count(129264), 2)
        with TemporaryDirectory() as directory:
            replaced = Path(directory) / "codec.json"
            replaced.write_bytes(fixtures.CODEC_PATH.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "identity"):
                fixtures.load_codec(replaced)

    def test_short_and_exact_references_are_complete_bound_and_cannot_be_replaced(self):
        metadata, arrays = fixtures.load_short()
        self.assertEqual(metadata["decode_token"], 16)
        self.assertEqual(len(metadata["tokens"]), 16)
        self.assertEqual(metadata["checkpoint"]["tensor_count"], 96085)
        self.assertEqual(set(arrays), {"prefill", "decode"})
        self.assertEqual(int(arrays["prefill"].argmax()), 16)
        self.assertEqual(int(arrays["decode"].argmax()), 1)
        for array in arrays.values():
            self.assertEqual(array.shape, (129280,))
            self.assertEqual(array.dtype, np.float64)
            self.assertTrue(np.isfinite(array).all())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "replaced.npz"
            original = fixtures.SHORT_PATH.read_bytes()
            path.write_bytes(original)
            copied, _ = fixtures.load_short(path)
            self.assertEqual(copied, metadata)
            for altered in (original[:-1], original + b"extra", b"not an artifact"):
                path.write_bytes(altered)
                with self.assertRaisesRegex(ValueError, "identity"):
                    fixtures.load_short(path)
        loader = getattr(fixtures, "load_exact_operators", None)
        self.assertTrue(callable(loader), "frozen exact-operator loader is missing")
        exact_metadata, cases = loader()
        self.assertEqual(
            set(cases),
            {"activation_fp8", "indexer_fp4", "compressed_kv_fp4", "engram"},
        )
        self.assertEqual(
            exact_metadata["reference_kind"], "hand-checked-source-precision-literals"
        )
        from development.operators.numerics import compare

        from . import reference_quantization
        from .engram import HashLayout, hash_rows

        for name, case in cases.items():
            if name == "engram":
                options = case.manifest["case"]["options"]
                tokens, split = case.inputs["tokens"], options["split"]
                layout = HashLayout(**options["layout"])
                first, history = hash_rows(
                    tokens[:split],
                    layout,
                    pad_id=options["pad_id"],
                    history=case.state["history"],
                )
                second, tail = hash_rows(
                    tokens[split:], layout, pad_id=options["pad_id"], history=history
                )
                actual = {
                    "hashes": np.concatenate((first, second)),
                    "history_after_split": np.asarray(history, dtype=np.int64),
                    "history": np.asarray(tail, dtype=np.int64),
                }
            else:
                result = getattr(reference_quantization, name)(case.inputs["values"])
                actual = {
                    "encoded": result.encoded,
                    "scales": result.scales,
                    "dequantized_bf16_bits": (
                        result.dequantized.view(np.uint32) >> 16
                    ).astype(np.uint16),
                }
            limits = case.manifest["limits"]
            with self.subTest(case=name):
                self.assertTrue(compare(actual, case, limits).passed)
                self.assertTrue(
                    all(not value.flags.writeable for value in case.expected.values())
                )
                field = "hashes" if name == "engram" else "dequantized_bf16_bits"
                changed = {key: value.copy() for key, value in actual.items()}
                changed[field].flat[0] ^= 1 if name == "engram" else 0x8000
                self.assertFalse(compare(changed, case, limits).passed)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exact.npz"
            original = fixtures.EXACT_OPERATORS_PATH.read_bytes()
            path.write_bytes(original + b"extra")
            with self.assertRaisesRegex(ValueError, "identity"):
                loader(path)
            with np.load(fixtures.EXACT_OPERATORS_PATH, allow_pickle=False) as stored:
                original_values = {key: stored[key] for key in stored.files}
            for alteration in ("incomplete", "provenance", "tensor"):
                values = {key: value.copy() for key, value in original_values.items()}
                if alteration == "incomplete":
                    del values["activation_fp8.expected.encoded"]
                elif alteration == "provenance":
                    changed = json.loads(values["metadata"].item())
                    changed["reference_kind"] = "candidate-output"
                    values["metadata"] = np.asarray(json.dumps(changed))
                else:
                    values["activation_fp8.expected.encoded"].flat[0] ^= 1
                np.savez(path, **values)
                with (
                    patch.object(
                        fixtures,
                        "EXACT_OPERATORS_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    ),
                    self.assertRaisesRegex(ValueError, alteration),
                ):
                    loader(path)
        lookup_loader = getattr(fixtures, "load_engram_lookup", None)
        self.assertTrue(
            callable(lookup_loader), "frozen Engram lookup loader is missing"
        )
        lookup_metadata, lookup = lookup_loader()
        self.assertEqual(
            lookup_metadata["checkpoint_manifest_sha256"],
            fixtures.CHECKPOINT_MANIFEST_SHA256,
        )
        self.assertEqual(lookup["indices"].shape, (8, 24))
        self.assertEqual(lookup["expected_bf16_bits"].shape, (8, 24, 256))
        np.testing.assert_array_equal(
            lookup["expected_bf16_bits"][0, 0, :7],
            [0, 0x8000, 0, 2, 0x8000, 0x8002, 1],
        )
        self.assertTrue(all(not value.flags.writeable for value in lookup.values()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lookup.npz"
            original = fixtures.ENGRAM_LOOKUP_PATH.read_bytes()
            path.write_bytes(original + b"extra")
            with self.assertRaisesRegex(ValueError, "identity"):
                lookup_loader(path)
            for alteration in ("provenance", "tensor"):
                with np.load(fixtures.ENGRAM_LOOKUP_PATH, allow_pickle=False) as stored:
                    values = {key: stored[key].copy() for key in stored.files}
                if alteration == "provenance":
                    changed = json.loads(values["metadata"].item())
                    changed["reference_kind"] = "candidate-output"
                    values["metadata"] = np.asarray(json.dumps(changed))
                else:
                    values["expected_bf16_bits"].flat[0] ^= 0x8000
                np.savez(path, **values)
                with (
                    patch.object(
                        fixtures,
                        "ENGRAM_LOOKUP_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    ),
                    self.assertRaisesRegex(ValueError, alteration),
                ):
                    lookup_loader(path)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; sys.modules.update({name: None for name in "
                "('torch', 'development.model.deepseek_v41.engram', "
                "'development.model.deepseek_v41.quantization', "
                "'development.model.deepseek_v41.reference_quantization')}); "
                "from development.model.deepseek_v41.fixtures "
                "import load_exact_operators, load_engram_lookup; "
                "assert len(load_exact_operators()[1]) == 4; "
                "assert len(load_engram_lookup()[1]) == 4",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
