import hashlib
import importlib
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer

from .test_codec_reference import fixture

SOURCE = r"""
from pathlib import Path

FAULT = "normal"
IMAGE = "<\uff5cdeepseek_image\uff5c>"


def encode_messages(messages, thinking_mode, context=None, **options):
    assert options["return_multi_modal_data"]
    parts, images = [], []
    for message in messages:
        for block in message["content"]:
            if block["type"] == "image":
                images.append(dict(block))
                parts.append(IMAGE)
            else:
                parts.append(block["text"])
    text = ("" if context else "<BOS>") + "\n\n".join(parts)
    messages.clear()
    directory = Path(__file__).parents[2]
    changed = {
        "encoder-mutation": Path(__file__),
        "tokenizer-mutation": Path(__file__).parents[1] / "tokenizer.json",
        "manifest-mutation": directory / "manifest.json",
    }.get(FAULT)
    if changed:
        changed.write_bytes(changed.read_bytes() + b"\n")
    if FAULT == "competing-output":
        (directory / "codec.json").write_text("other owner")
    if FAULT == "reordered-media":
        images.reverse()
    if FAULT == "text-bound":
        text += "x" * 65537
    if FAULT == "token-bound":
        text += " x" * 4097
    if FAULT == "encoder-failure":
        raise ValueError("encoder failure")
    return text, {"images": images}
"""


def model_fixture(directory, fault="normal"):
    root, manifest_path, output = fixture(directory)
    source = root / "encoding/encoding.py"
    source.write_text(SOURCE.replace('FAULT = "normal"', f"FAULT = {fault!r}"))
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["encoding/encoding.py"] = {
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, output


class VideoCodec(unittest.TestCase):
    def setUp(self):
        name = f"{__package__}.video_codec"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "supplemental video codec is missing"
        )
        self.codec = importlib.import_module(name)

    def test_frozen_cases_keep_interleaved_tied_and_context_image_order(self):
        artifact = self.codec.load_video_codec()
        cases = {case["name"]: case for case in artifact["cases"]}
        self.assertEqual(
            list(cases),
            [
                "mixed-still-video",
                "rounded-timestamp-ties",
                "context-video-continuation",
            ],
        )
        expected_urls = {
            "mixed-still-video": [
                "fixture://still-before",
                "fixture://mixed-frame-0",
                "fixture://mixed-frame-1",
                "fixture://still-after",
            ],
            "rounded-timestamp-ties": [
                f"fixture://tied-frame-{index}" for index in range(4)
            ],
            "context-video-continuation": [
                f"fixture://tied-frame-{index}" for index in range(4)
            ],
        }
        token_hashes = {
            "mixed-still-video": (
                "6dc70a9c741105772bbe7ab817830fd56ba3f5efd2288ad37f46aee946d2b6b9"
            ),
            "rounded-timestamp-ties": (
                "4db564650981790880b61da2bf738fec31d9c3694436ce579869dad3204bc720"
            ),
            "context-video-continuation": (
                "dc3d83e68738900a1b7d7ccdce7c3aa31b01aed46848d518b54854be80fc021b"
            ),
        }
        for name, case in cases.items():
            self.assertEqual(
                case["media"],
                {
                    "images": [
                        {"type": "image", "url": url} for url in expected_urls[name]
                    ]
                },
            )
            self.assertEqual(case["rendered_text"].count(self.codec.IMAGE), 4)
            self.assertEqual(case["token_ids"].count(129264), 4)
            self.assertGreater(len(case["token_ids"]), 4)
            encoded = json.dumps(case["token_ids"], separators=(",", ":")).encode()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), token_hashes[name])
        mixed = cases["mixed-still-video"]
        fragments = [
            "Before the still.",
            self.codec.IMAGE,
            "Before the video.",
            "<0.0 seconds>",
            self.codec.IMAGE,
            "<0.5 seconds>",
            self.codec.IMAGE,
            "After the video.",
            self.codec.IMAGE,
            "After the still.",
        ]
        remaining = mixed["rendered_text"]
        for fragment in fragments:
            _, separator, remaining = remaining.partition(fragment)
            self.assertTrue(separator, fragment)
        tied = cases["rounded-timestamp-ties"]
        self.assertEqual(tied["video"]["selected_indices"], [0, 1, 2, 3])
        self.assertEqual(tied["video"]["fps"], 100.0)
        self.assertEqual(tied["rendered_text"].count("<0.0 seconds>"), 4)
        continued = cases["context-video-continuation"]
        self.assertEqual(
            continued["inputs"]["context"][0], mixed["inputs"]["messages"][0]
        )
        self.assertEqual(continued["media"], tied["media"])
        self.assertNotIn("Before the still.", continued["rendered_text"])
        self.assertEqual(continued["token_ids"][0], 128803)
        self.assertEqual(tied["token_ids"][0], 0)
        self.assertEqual(
            artifact["checkpoint_manifest_sha256"],
            "0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67",
        )
        self.assertEqual(artifact["dependencies"]["tokenizers"], "0.23.3")
        self.assertIn(
            "development/model/deepseek_v41/video_codec.py",
            artifact["reference_sources"],
        )
        old = Path(__file__).with_name("codec.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(old).hexdigest(),
            "30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75",
        )
        original = self.codec.VIDEO_CODEC_PATH.read_bytes()
        self.assertEqual(len(original), 24704)
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "21553a1d0c51f5361a202ed34048c8d4b39fab2cb8a64856699c653439975f7b",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "copied.json"
            path.write_bytes(original)
            self.assertEqual(self.codec.load_video_codec(path), artifact)
            altered = json.loads(original)
            altered["cases"][1]["media"]["images"].reverse()
            for value in (
                original[:-1],
                original + b" ",
                b"not JSON",
                json.dumps(altered).encode(),
            ):
                path.write_bytes(value)
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.codec.load_video_codec(path)

    def test_generation_binds_verified_bytes_complete_tokens_and_case_inputs(self):
        with TemporaryDirectory() as directory:
            root, manifest, output = model_fixture(Path(directory))
            result = self.codec.generate(root, manifest, output)
            self.assertEqual(json.loads(output.read_bytes()), result)
            self.assertEqual(
                result["verified_checkpoint_files"],
                json.loads(manifest.read_bytes())["files"],
            )
            self.assertEqual(
                result["checkpoint_manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            tokenizer = Tokenizer.from_buffer((root / "tokenizer.json").read_bytes())
            tokenizer.no_padding()
            tokenizer.no_truncation()
            for planned, actual in zip(
                self.codec.fixture_cases(), result["cases"], strict=True
            ):
                self.assertEqual(actual["inputs"], planned["inputs"])
                self.assertEqual(actual["video"], planned["video"])
                self.assertEqual(actual["media"]["images"], planned["expected_images"])
                expected = tokenizer.encode(
                    actual["rendered_text"], add_special_tokens=False
                ).ids
                self.assertEqual(actual["token_ids"], expected)
                self.assertEqual(
                    tokenizer.decode(expected, skip_special_tokens=False),
                    actual["rendered_text"],
                )
            self.assertFalse((root / "encoding/__pycache__").exists())
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()},
                {"model", "manifest.json", "codec.json"},
            )

    def test_generation_failure_keeps_existing_ownership_and_removes_temporary_files(
        self,
    ):
        for fault in (
            "existing-output",
            "competing-output",
            "encoder-mutation",
            "tokenizer-mutation",
            "manifest-mutation",
            "wrong-bytes",
            "reordered-media",
            "text-bound",
            "token-bound",
            "encoder-failure",
        ):
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                root, manifest, output = model_fixture(Path(directory), fault)
                if fault == "existing-output":
                    output.write_text("other owner")
                elif fault == "wrong-bytes":
                    (root / "encoding/encoding.py").write_text(
                        'raise AssertionError("unverified source executed")'
                    )
                with self.assertRaises((ValueError, FileExistsError)):
                    self.codec.generate(root, manifest, output)
                owned = fault in ("existing-output", "competing-output")
                self.assertEqual(output.exists(), owned)
                if owned:
                    self.assertEqual(output.read_text(), "other owner")
                self.assertEqual(
                    {path.name for path in Path(directory).iterdir()},
                    {"model", "manifest.json"} | ({"codec.json"} if owned else set()),
                )


if __name__ == "__main__":
    unittest.main()
