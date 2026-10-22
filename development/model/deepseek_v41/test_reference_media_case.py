import base64
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import reference_media_case as case
from .reference_media import expand


class MediaCaseReference(unittest.TestCase):
    def test_generation_checks_identity_before_and_after_computation(self):
        versions = {
            "numpy": "2.5.4",
            "torch": "2.12.0",
            "tokenizers": "0.23.3",
            "pillow": "12.3.0",
        }
        for fault in (
            None,
            "startup",
            "checkpoint",
            "source",
            "publication_checkpoint",
            "publication_source",
        ):
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                root = Path(directory) / "model"
                (root / "encoding").mkdir(parents=True)
                config = {
                    "text_config": {"engram_compressed_vocab_size": 3},
                    "vision_config": {},
                }
                contents = {
                    "config.json": json.dumps(config),
                    "tokenizer.json": "{}",
                    "tokenizer_config.json": (
                        '{"add_bos_token":false,"add_eos_token":false}'
                    ),
                    "encoding/encoding.py": (
                        "def encode_messages(**kwargs): return '', {'images': []}"
                    ),
                }
                for name, content in contents.items():
                    (root / name).write_text(content)
                manifest = {
                    "verification": "full",
                    "files": {
                        name: {
                            "size": len(content.encode()),
                            "sha256": hashlib.sha256(content.encode()).hexdigest(),
                        }
                        for name, content in contents.items()
                    },
                }
                manifest_path = root.parent / "manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                output = root.parent / "media.npz"
                sources = {"source": "original"}

                def compute(*args, fault=fault, sources=sources, root=root, **kwargs):
                    if fault == "checkpoint":
                        (root / "tokenizer.json").write_text('{"changed":true}')
                    if fault == "source":
                        sources["source"] = "changed"
                    return {"decode_token": 1}, {
                        "prefill": np.zeros(129280),
                        "decode": np.zeros(129280),
                    }

                def vision(weights, supplied, config=config):
                    self.assertEqual(supplied, config)
                    return object()

                serialize = np.savez_compressed

                def archive(
                    *args,
                    fault=fault,
                    sources=sources,
                    root=root,
                    serialize=serialize,
                    **kwargs,
                ):
                    serialize(*args, **kwargs)
                    if fault == "publication_checkpoint":
                        (root / "tokenizer.json").write_text('{"late_change":true}')
                    if fault == "publication_source":
                        sources["source"] = "late_change"

                tokenizer = SimpleNamespace(
                    no_padding=lambda: None, no_truncation=lambda: None
                )
                modules = {
                    "torch": SimpleNamespace(set_num_threads=lambda _: None),
                    "tokenizers": SimpleNamespace(
                        Tokenizer=SimpleNamespace(
                            from_buffer=lambda _, tokenizer=tokenizer: tokenizer
                        )
                    ),
                    "development.model.deepseek_v41.engram": SimpleNamespace(
                        compressed_token_map=lambda _: (np.arange(3), 3)
                    ),
                    "development.model.deepseek_v41.reference_model": SimpleNamespace(
                        Model=lambda *args: object(),
                        TorchWeights=lambda *args, **kwargs: object(),
                    ),
                    "development.model.deepseek_v41.reference_vision": SimpleNamespace(
                        Vision=vision
                    ),
                }
                with (
                    patch.dict("sys.modules", modules),
                    patch.object(
                        case.importlib.metadata,
                        "version",
                        side_effect=versions.__getitem__,
                    ),
                    patch.object(
                        case,
                        "case_sources",
                        side_effect=lambda sources=sources: dict(sources),
                    ),
                    patch.object(
                        case,
                        "inspect_checkpoint",
                        return_value={} if fault == "startup" else manifest,
                    ),
                    patch.object(
                        case,
                        "render_prompt",
                        return_value=(object(), {"input": "fixed"}),
                    ),
                    patch.object(case, "tensor_layout", return_value={}),
                    patch.object(case, "Weights", return_value=object()),
                    patch.object(case.np, "savez_compressed", side_effect=archive),
                    patch.object(
                        case, "media_endpoints", side_effect=compute
                    ) as computation,
                ):
                    if fault is None:
                        case.generate(root, manifest_path, output)
                        with np.load(output, allow_pickle=False) as artifact:
                            metadata = json.loads(str(artifact["metadata"]))
                            self.assertEqual(metadata["checkpoint"], manifest)
                            self.assertEqual(metadata["reference_sources"], sources)
                            self.assertEqual(
                                metadata["inputs"],
                                {"input": "fixed", "decode_token": 1},
                            )
                            self.assertEqual(
                                metadata["arrays"]["decode"]["shape"], [129280]
                            )
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "identity" if fault == "startup" else "changed"
                        ):
                            case.generate(root, manifest_path, output)
                        self.assertFalse(output.exists())
                    self.assertEqual(
                        computation.call_count, 0 if fault == "startup" else 1
                    )

    def test_prompt_keeps_still_and_frame_order_and_reserves_decode_position(self):
        inputs = case.case_inputs()
        records = [{"type": "image", "url": item["url"]} for item in inputs["images"]]
        config = {
            "image_token_id": 2,
            "text_config": {"vocab_size": 3, "max_position_embeddings": 128},
            "vision_config": {
                "num_hidden_layers": 1,
                "patch_size": 14,
                "downsample_ratio": 3,
                "max_image_tokens": 16,
                "min_pixels": 0,
                "max_wh_ratio": None,
            },
        }
        encoder = SimpleNamespace(
            encode_messages=lambda **kwargs: ("verified text", {"images": records})
        )
        tokenizer = SimpleNamespace(
            encode=lambda *args, **kwargs: SimpleNamespace(ids=[0, 2, 1, 2, 2, 1])
        )
        prompt, metadata = case.render_prompt(encoder, tokenizer, config)
        self.assertEqual(len(prompt.spans), 3)
        self.assertEqual([span.start for span in prompt.spans], [1, 6, 10])
        self.assertEqual(metadata["placeholder_tokens"], [0, 2, 1, 2, 2, 1])
        self.assertEqual(metadata["expanded_tokens"], prompt.tokens.tolist())
        self.assertEqual(metadata["token_types"], prompt.token_types.tolist())
        self.assertEqual(metadata["images"], inputs["images"])
        blocks = inputs["messages"][0]["content"]
        self.assertEqual(
            [block["text"] for block in blocks if block["type"] == "text"],
            [
                "Compare this still image and these video frames.",
                "<0.0 seconds>",
                "<0.5 seconds>",
                "Describe the changes in order.",
            ],
        )
        self.assertEqual(len({image["sha256"] for image in inputs["images"]}), 3)
        self.assertEqual(case.case_inputs(), inputs)
        records.reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            case.render_prompt(encoder, tokenizer, config)
        records.reverse()
        config["text_config"]["max_position_embeddings"] = len(prompt.tokens)
        with self.assertRaisesRegex(ValueError, "decode"):
            case.render_prompt(encoder, tokenizer, config)

    def test_publication_requires_full_endpoints_and_never_replaces_an_output(self):
        arrays = {name: np.zeros(129280) for name in ("prefill", "decode")}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "media.npz"
            case.write_case(output, checkpoint, {"case": "media"}, arrays)
            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(set(artifact.files), {"metadata", "prefill", "decode"})
                self.assertEqual(json.loads(str(artifact["metadata"]))["case"], "media")
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                case.write_case(output, checkpoint, {}, arrays)
            self.assertEqual(output.read_bytes(), original)
            checked = case.output_path
            missing = root / "missing.npz"

            def race(path, model):
                destination = checked(path, model)
                destination.symlink_to(missing)
                return destination

            with (
                patch.object(case, "output_path", side_effect=race),
                self.assertRaises(FileExistsError),
            ):
                case.write_case(root / "racing.npz", checkpoint, {}, arrays)
            self.assertFalse(missing.exists())
            for replacement in (
                {},
                {"prefill": np.zeros(3), "decode": arrays["decode"]},
                dict(arrays, decode=np.full(129280, np.nan)),
            ):
                with self.assertRaises(ValueError):
                    case.write_case(root / "invalid.npz", checkpoint, {}, replacement)
                self.assertFalse((root / "invalid.npz").exists())


class MediaCaseMath(unittest.TestCase):
    def test_computation_inserts_all_spans_then_continues_from_prefill_state(self):
        import torch

        from .reference_model import Model, TorchWeights
        from .test_reference_model import ArrayWeights, backbone_fixture

        text, arrays = backbone_fixture()
        text["max_position_embeddings"] = 32
        config = {
            "text_config": text,
            "image_token_id": 2,
            "vision_config": {
                "num_hidden_layers": 1,
                "patch_size": 14,
                "downsample_ratio": 3,
                "max_image_tokens": 16,
                "min_pixels": 0,
                "max_wh_ratio": None,
            },
        }
        arrays["layers.0.attn.wq_a.weight"] = np.eye(2)
        arrays["layers.0.attn.wq_b.weight"] = np.eye(2)
        arrays["layers.0.attn.wo_a.weight"] = np.eye(2)
        arrays["layers.0.attn.wo_b.weight"] = np.eye(2)
        arrays["layers.0.ffn.gate.bias_vl"] = np.array([1.0, 0.0])
        model = Model(
            TorchWeights(ArrayWeights(arrays), device="cpu"), config, np.arange(3)
        )
        images = [
            base64.b64decode(item["encoded_base64"])
            for item in case.case_inputs()["images"]
        ]
        prompt = expand([0, 2, 1, 2, 2, 1], images, config)

        def vision_forward(patches, n_h, n_w):
            value = patches.mean().item()
            return torch.tensor([[value, 1.0]] * 4, dtype=torch.float64)

        vision = SimpleNamespace(forward=vision_forward)
        inputs, result = case.media_endpoints(model, vision, prompt)
        embeddings = torch.cat(
            [
                vision_forward(
                    torch.from_numpy(span.image.patches.copy()),
                    span.image.plan.patch_h,
                    span.image.plan.patch_w,
                )
                for span in prompt.spans
            ]
        )
        prefill = model.forward(
            prompt.tokens, token_types=prompt.token_types, image_embeddings=embeddings
        )
        token = int(prefill.logits[0].argmax())
        decoded = model.forward([token], previous=prefill.state)
        np.testing.assert_array_equal(result["prefill"], prefill.logits[0].numpy())
        np.testing.assert_array_equal(result["decode"], decoded.logits[0].numpy())
        self.assertEqual(inputs["decode_token"], token)
        self.assertEqual(len(inputs["vision_embeddings"]), 3)
        for index, identity in enumerate(inputs["vision_embeddings"]):
            self.assertEqual(identity["shape"], [4, 2])
            self.assertEqual(
                identity["sha256"],
                hashlib.sha256(
                    embeddings[index * 4 : index * 4 + 4]
                    .numpy()
                    .astype("<f8")
                    .tobytes()
                ).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
