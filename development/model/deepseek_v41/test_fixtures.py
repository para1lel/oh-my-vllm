import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import fixtures


class FrozenFixtures(unittest.TestCase):
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
