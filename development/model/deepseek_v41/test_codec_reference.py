import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

from . import codec_reference as codec
from .identity import VERSION

SOURCE = """import json


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None,
                    return_multi_modal_data=False):
    assert return_multi_modal_data is True
    inputs = dict(messages=messages, thinking_mode=thinking_mode, context=context,
                  drop_thinking=drop_thinking,
                  add_default_bos_token=add_default_bos_token,
                  reasoning_effort=reasoning_effort)
    text = ("[BOS]" if not context and add_default_bos_token else "")
    text += json.dumps(inputs, sort_keys=True)
    images = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if value.get("type") == "image_url":
                images.append(dict(type="image", url=value["image_url"]["url"]))
            else:
                for item in value.values():
                    collect(item)

    collect(messages)
    messages.clear()
    return text, dict(images=images)
"""


def fixture(directory):
    root = directory / "model"
    (root / "encoding").mkdir(parents=True)
    (root / "encoding/encoding.py").write_text(SOURCE)
    tokenizer = Tokenizer(
        models.BPE(
            {
                word: index
                for index, word in enumerate(
                    sorted(pre_tokenizers.ByteLevel.alphabet())
                )
            },
            [],
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[AUTO_BOS]", "[AUTO_EOS]"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[AUTO_BOS] $A [AUTO_EOS]",
        special_tokens=[
            (name, tokenizer.token_to_id(name)) for name in ("[AUTO_BOS]", "[AUTO_EOS]")
        ],
    )
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=12)
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        '{"add_bos_token":false,"add_eos_token":false}'
    )
    manifest = {
        "version": VERSION,
        "verification": "full",
        "files": {
            name: {
                "size": (root / name).stat().st_size,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name in codec.CHECKPOINT_FILES
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, directory / "codec.json"


class CodecReference(unittest.TestCase):
    def test_fixed_cases_preserve_authority_inputs_tokens_and_media(self):
        with TemporaryDirectory() as directory:
            root, manifest, output = fixture(Path(directory))
            result = codec.generate(root, manifest, output)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(
                result["checkpoint_manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["verified_checkpoint_files"],
                json.loads(manifest.read_text())["files"],
            )
            self.assertEqual(result["dependencies"]["tokenizers"], "0.23.3")
            cases = {case["name"]: case for case in result["cases"]}
            self.assertEqual(len(cases), len(codec.fixture_cases()))
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            for original, case in zip(
                codec.fixture_cases(), result["cases"], strict=True
            ):
                self.assertEqual(case["inputs"], original["inputs"])
                self.assertEqual(
                    case["token_ids"],
                    tokenizer.encode(
                        case["rendered_text"], add_special_tokens=False
                    ).ids,
                )
                self.assertEqual(
                    tokenizer.decode(case["token_ids"], skip_special_tokens=False),
                    case["rendered_text"],
                )
                self.assertNotIn(tokenizer.token_to_id("[AUTO_BOS]"), case["token_ids"])
                self.assertNotIn(tokenizer.token_to_id("[AUTO_EOS]"), case["token_ids"])
            self.assertEqual(cases["thinking-low"]["inputs"]["reasoning_effort"], "low")
            self.assertEqual(cases["thinking-50"]["inputs"]["reasoning_effort"], 50)
            self.assertFalse(cases["reasoning-keep"]["inputs"]["drop_thinking"])
            self.assertIn(
                '"tool_call_id": "second"', cases["tools-ordered"]["rendered_text"]
            )
            self.assertTrue(
                cases["assistant-continuation"]["inputs"]["messages"][-1]["wo_eos"]
            )
            self.assertFalse(
                cases["context-continuation"]["rendered_text"].startswith("[BOS]")
            )
            self.assertEqual(
                cases["ordered-images"]["media"],
                {
                    "images": [
                        {"type": "image", "url": "fixture://first"},
                        {"type": "image", "url": "fixture://second"},
                    ]
                },
            )
            self.assertEqual(
                cases["context-images"]["media"],
                {"images": [{"type": "image", "url": "fixture://current"}]},
            )
            self.assertEqual(
                cases["tool-result-images"]["media"],
                {
                    "images": [
                        {"type": "image", "url": "fixture://second"},
                        {"type": "image", "url": "fixture://first"},
                    ]
                },
            )
            self.assertFalse((root / "encoding/__pycache__").exists())
            for limit in (
                "MAX_CASES",
                "MAX_CASE_INPUT_BYTES",
                "MAX_CASE_TEXT_BYTES",
                "MAX_CASE_TOKENS",
                "MAX_TOTAL_TOKENS",
            ):
                bounded = output.with_name("bounded.json")
                with (
                    self.subTest(limit=limit),
                    patch.object(codec, limit, 1),
                    self.assertRaisesRegex(ValueError, "bound"),
                ):
                    codec.generate(root, manifest, bounded)
                self.assertFalse(bounded.exists())

    def test_rejects_unbound_identity_and_unsafe_or_existing_output(self):
        faults = (
            "metadata-only",
            "version",
            "missing-source",
            "source",
            "tokenizer",
            "config",
            "automatic-bos",
            "automatic-eos",
            "symlink",
            "dependency",
            "existing",
            "competing-output",
            "checkpoint-output",
            "repository-output",
            "suffix",
        )
        for fault in faults:
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                root, manifest_path, output = fixture(Path(directory))
                manifest = json.loads(manifest_path.read_text())
                if fault == "metadata-only":
                    manifest["verification"] = fault
                elif fault == "version":
                    manifest["version"] = "unknown"
                elif fault == "missing-source":
                    del manifest["files"]["encoding/encoding.py"]
                elif fault in ("source", "tokenizer", "config"):
                    name = {
                        "source": "encoding/encoding.py",
                        "tokenizer": "tokenizer.json",
                        "config": "tokenizer_config.json",
                    }[fault]
                    (root / name).write_text(
                        'raise AssertionError("unverified source")'
                    )
                elif fault in ("automatic-bos", "automatic-eos"):
                    path = root / "tokenizer_config.json"
                    config = json.loads(path.read_text())
                    config[f"add_{fault.removeprefix('automatic-')}_token"] = True
                    path.write_text(json.dumps(config))
                    manifest["files"]["tokenizer_config.json"] = {
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                elif fault == "symlink":
                    source = root / "encoding/encoding.py"
                    source.rename(root / "copy.py")
                    source.symlink_to(root / "copy.py")
                elif fault == "existing":
                    output.write_text("keep")
                elif fault == "checkpoint-output":
                    output = root / "codec.json"
                elif fault == "repository-output":
                    output = Path(codec.__file__).parent / "codec-output.json"
                elif fault == "suffix":
                    output = output.with_suffix(".npz")
                manifest_path.write_text(json.dumps(manifest))
                link = codec.os.link

                def competing_link(source, target, link=link):
                    Path(target).write_text("keep")
                    return link(source, target)

                with (
                    patch(
                        "importlib.metadata.version",
                        return_value="0.0.0" if fault == "dependency" else "0.23.3",
                    ),
                    patch.object(
                        codec.os,
                        "link",
                        side_effect=competing_link
                        if fault == "competing-output"
                        else link,
                    ),
                    self.assertRaises((ValueError, FileExistsError)),
                ):
                    codec.generate(root, manifest_path, output)
                if fault in ("existing", "competing-output"):
                    self.assertEqual(output.read_text(), "keep")
                else:
                    self.assertFalse(output.exists())
                self.assertEqual(
                    {path.name for path in Path(directory).iterdir()},
                    {"model", "manifest.json"}
                    | (
                        {"codec.json"}
                        if fault in ("existing", "competing-output")
                        else set()
                    ),
                )

    def test_rejects_mutation_before_publication(self):
        for name in (*codec.CHECKPOINT_FILES, "manifest.json"):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root, manifest, output = fixture(Path(directory))
                render = codec.render_cases

                path = manifest if name == "manifest.json" else root / name

                def mutate(encoder, tokenizer, render=render, path=path):
                    result = render(encoder, tokenizer)
                    path.write_bytes(path.read_bytes() + b"\n")
                    return result

                with (
                    patch.object(codec, "render_cases", side_effect=mutate),
                    self.assertRaisesRegex(ValueError, "changed"),
                ):
                    codec.generate(root, manifest, output)
                self.assertFalse(output.exists())
        with TemporaryDirectory() as directory:
            root, manifest, output = fixture(Path(directory))
            expected = codec.generate(root, manifest, output)
            replacement = root.with_name("replacement")
            replacement.mkdir()
            Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(
                str(replacement / "tokenizer.json")
            )
            original_root = root.with_name("original")
            loader = Tokenizer.from_file

            def temporarily_replace(path):
                root.rename(original_root)
                replacement.rename(root)
                try:
                    return loader(path)
                finally:
                    root.rename(replacement)
                    original_root.rename(root)

            with patch.object(Tokenizer, "from_file", side_effect=temporarily_replace):
                actual = codec.generate(root, manifest, output.with_name("second.json"))
            self.assertEqual(actual["cases"], expected["cases"])


if __name__ == "__main__":
    unittest.main()
