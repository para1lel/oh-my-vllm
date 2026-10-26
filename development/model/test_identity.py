import ast
import hashlib
import json
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from oh_my_vllm.models.qwen38.checkpoint import Checkpoint
from oh_my_vllm.operators.runtime.native import source_digest
from oh_my_vllm.operators.specification import MODEL

from .capture import PROMPT
from .compare import REFERENCE_PATH, load_reference


def checkpoint_fixture(root):
    config = b'{"fixture":true}'
    (root / "config.json").write_bytes(config)
    weights = {name: f"{name}.safetensors" for name in ("weight", "embedding")}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    for name, filename in weights.items():
        header = json.dumps(
            {name: {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
        ).encode()
        header += b" " * (-len(header) % 8)
        (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")
    return hashlib.sha256(config).hexdigest()


class Identity(unittest.TestCase):
    def test_frozen_reference_is_complete_and_bound_to_its_source(self):
        metadata, outputs = load_reference()
        self.assertEqual(metadata["reference"]["precision"], "torch.float64")
        self.assertEqual(metadata["case"]["tokens"], list(PROMPT))
        source = Path(__file__).with_name("reference.py").read_bytes()
        self.assertEqual(
            metadata["reference"]["source"], hashlib.sha256(source).hexdigest()
        )
        for value in outputs.values():
            self.assertEqual(value.shape, (248320,))
            self.assertEqual(str(value.dtype), "float64")
        with TemporaryDirectory() as directory:
            changed = Path(directory) / "reference.npz"
            changed.write_bytes(REFERENCE_PATH.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                load_reference(changed)

    def test_checkpoint_identity_tracks_loaded_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint_fixture(root)
            with patch.dict(MODEL, {"config_sha256": config}):
                loaded = Checkpoint(root)
                loaded.raw_tensor("weight")
                frozen = loaded.verify_identity()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "weight.safetensors",
                ):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    self.assertEqual(loaded.identity, frozen)
                    with (
                        self.subTest(checkpoint=name),
                        self.assertRaisesRegex(ValueError, "file bytes changed"),
                    ):
                        loaded.verify_identity(frozen)
                    path.write_bytes(original)
                    self.assertEqual(loaded.verify_identity(frozen), frozen)
                loaded.raw_tensor("embedding")
                with self.assertRaisesRegex(ValueError, "loaded checkpoint set"):
                    loaded.verify_identity(frozen)

    def test_native_identity_tracks_only_transitive_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, header, leaf, unrelated = (
                root / name
                for name in (
                    "fa/op.cu",
                    "shared/common.cuh",
                    "gdn/leaf.cuh",
                    "other.cuh",
                )
            )
            for path in (source, header, leaf):
                path.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('#include "../shared/common.cuh"\n')
            header.write_text('#include "../gdn/leaf.cuh"\n')
            leaf.write_text('#include "../shared/common.cuh"\n#define WIDTH 128\n')
            unrelated.write_text("unrelated\n")
            identity = source_digest(source, root)
            unrelated.write_text("changed\n")
            self.assertEqual(source_digest(source, root), identity)
            leaf.write_text('#include "../shared/common.cuh"\n#define WIDTH 256\n')
            self.assertNotEqual(source_digest(source, root), identity)
            leaf.unlink()
            with self.assertRaises(FileNotFoundError):
                source_digest(source, root)

    def test_reference_identity_binds_checkpoint_import_chain(self):
        root = Path(__file__).resolve().parents[2]
        read_bytes = Path.read_bytes
        for module in (
            "development/model/reference_cases.py",
            "development/operators/reference/__main__.py",
        ):
            path = root / module
            tree = ast.parse(path.read_text())
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "source_identity"
            )
            # Exercise provenance without importing the separate Torch environment.
            namespace = {"__file__": str(path), "Path": Path, "hashlib": hashlib}
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]), str(path), "exec"
                ),
                namespace,
            )
            identity = namespace["source_identity"]
            original = identity()
            for relative in (
                "python/oh_my_vllm/worker/checkpoint.py",
                "python/oh_my_vllm/models/qwen38/checkpoint.py",
            ):
                target = root / relative

                def changed_bytes(path, target=target):
                    return read_bytes(path) + (
                        b"\n# changed\n" if path == target else b""
                    )

                with (
                    self.subTest(module=module, dependency=relative),
                    patch.object(Path, "read_bytes", changed_bytes),
                ):
                    self.assertNotEqual(identity(), original)

    def test_ready_identity_includes_shards_before_lazy_media_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint_fixture(root)
            with patch.dict(MODEL, {"config_sha256": config}):
                loaded = Checkpoint(root)
                loaded.raw_tensor("weight")
                complete = loaded.complete_identity()
                self.assertEqual(
                    set(complete),
                    {"config", "index", "weight.safetensors", "embedding.safetensors"},
                )
                self.assertEqual(
                    complete["embedding.safetensors"],
                    hashlib.sha256(
                        (root / "embedding.safetensors").read_bytes()
                    ).hexdigest(),
                )
                path = root / "embedding.safetensors"
                original = path.read_bytes()
                path.write_bytes(original[:-1] + b"\x7f")
                with self.assertRaisesRegex(ValueError, "file bytes changed"):
                    loaded.raw_tensor("embedding")
                self.assertEqual(loaded.identity, complete)
                path.write_bytes(original)
                loaded = Checkpoint(root)
                complete = loaded.complete_identity()
                loaded.raw_tensor("embedding")
                self.assertEqual(loaded.verify_identity(complete), complete)
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "file bytes changed"):
                    loaded.raw_tensor("embedding")
                with self.assertRaisesRegex(ValueError, "file bytes changed"):
                    loaded.complete_identity()

    def test_lazy_vision_publishes_only_after_final_identity_verification(self):
        from oh_my_vllm.models.qwen38.weights import Weights

        for changed in (False, True):
            with self.subTest(changed=changed), TemporaryDirectory() as directory:
                root = Path(directory)
                digest = checkpoint_fixture(root)
                with patch.dict(MODEL, {"config_sha256": digest}):
                    weights = Weights.__new__(Weights)
                    weights.checkpoint = Checkpoint(root)
                    weights.checkpoint.complete_identity()
                    weights.visual = None
                    weights.device = SimpleNamespace(view=lambda *args: object())

                    def upload(name, changed=changed, root=root):
                        if changed and name == "model.visual.merger.linear_fc2.bias":
                            path = root / "embedding.safetensors"
                            original = path.read_bytes()
                            path.write_bytes(original[:-1] + b"\x7f")
                        return object()

                    weights.upload = upload
                    if changed:
                        with self.assertRaisesRegex(ValueError, "file bytes changed"):
                            weights.vision()
                        self.assertIsNone(weights.visual)
                    else:
                        self.assertIs(weights.vision(), weights.visual)

    def test_added_references_bind_complete_inputs_and_endpoints(self):
        from development.operators.reference.shards import write_fixture

        from .cases import load_case_fixture, load_cases

        cases = {case.id: case for case in load_cases()}
        self.assertEqual(set(cases), {"mtp-restored", "image-video", "context-32769"})
        case = cases["context-32769"]
        stored = {
            "inputs.tokens": np.arange(32768, dtype=np.int64),
            "inputs.positions": np.tile(np.arange(32768, dtype=np.int64), (3, 1)),
            "inputs.continuation": np.array([271], dtype=np.int64),
            "expected.prefill": np.ones(248320, dtype=np.float64),
            "expected.decode": np.ones(248320, dtype=np.float64),
        }
        provenance = dict(precision="torch.float64", source="fixed", torch="fixed")
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, values):
                location = root / name
                identity = write_fixture(
                    location,
                    case,
                    values,
                    {key: str(value.dtype) for key, value in values.items()},
                    {},
                    {"config": "fixed"},
                    provenance,
                )
                return replace(case, fixture_sha256=identity), location

            approved, location = write("valid", stored)
            fixture = load_case_fixture(approved, location)
            self.assertEqual(fixture.expected["decode"].shape, (248320,))
            with self.assertRaisesRegex(ValueError, "approved"):
                load_case_fixture(replace(approved, fixture_sha256=None), location)
            malformed, location = write(
                "truncated",
                {**stored, "expected.decode": stored["expected.decode"][:1]},
            )
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                load_case_fixture(malformed, location)
            malformed, location = write(
                "wrong-input", {**stored, "inputs.tokens": stored["inputs.tokens"][:-1]}
            )
            with self.assertRaisesRegex(ValueError, "input"):
                load_case_fixture(malformed, location)

            case = cases["mtp-restored"]
            stored = {
                "inputs." + name: np.zeros(shape, dtype=dtype)
                for name, (dtype, shape) in case.input_shapes.items()
            }
            stored.update(
                {
                    "expected." + name: np.ones(
                        (5, 248320) if name == "verified" else (248320,),
                        dtype=np.float64,
                    )
                    for name in case.outputs
                }
            )
            approved, location = write("verification", stored)
            fixture = load_case_fixture(approved, location)
            self.assertEqual(fixture.expected["verified"].shape, (5, 248320))
            malformed, location = write(
                "last-verification-row",
                {**stored, "expected.verified": stored["expected.verified"][-1]},
            )
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                load_case_fixture(malformed, location)
