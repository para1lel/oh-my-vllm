import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import fixtures


class FrozenFixtures(unittest.TestCase):
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

    def test_short_reference_is_complete_bound_and_cannot_be_replaced(self):
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


if __name__ == "__main__":
    unittest.main()
