"""The accepted TileLang implementation is an immutable comparison artifact."""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTest(unittest.TestCase):
    def test_backend_identity_matches_process_selection(self):
        import os

        for selected in (None, "cuda", "tilelang"):
            with self.subTest(selected=selected):
                environment = dict(os.environ)
                environment.pop("OH_MY_VLLM_KERNEL_BACKEND", None)
                if selected is not None:
                    environment["OH_MY_VLLM_KERNEL_BACKEND"] = selected
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import json, os; "
                        "from oh_my_vllm.kernels.backend import NAME; "
                        "from oh_my_vllm.worker.runtime import identity; "
                        "os.environ['OH_MY_VLLM_KERNEL_BACKEND'] = 'typo'; "
                        "print(json.dumps([NAME, identity()['kernel_backend']]))",
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                expected = selected or "cuda"
                self.assertEqual(json.loads(result.stdout), [expected, expected])

    def test_reference_sources_match_frozen_hashes(self):
        manifest = json.loads(
            (ROOT / "development/kernels/tilelang-reference.json").read_text()
        )
        source = ROOT / "python/oh_my_vllm/kernels/tilelang_reference"
        for name, digest in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((source / name).read_bytes()).hexdigest(), digest
            )

    def test_invalid_backend_is_not_silently_accepted(self):
        import os

        result = subprocess.run(
            [sys.executable, "-c", "import oh_my_vllm.kernels.backend"],
            env={**os.environ, "OH_MY_VLLM_KERNEL_BACKEND": "typo"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown OH_MY_VLLM_KERNEL_BACKEND", result.stderr)

    def test_formal_collector_rejects_changed_reference_or_lock(self):
        import shutil
        import tempfile

        from development.kernels.reference import verify_reference

        verify_reference()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "development/kernels/tilelang-reference.json",
                "requirements/runtime.txt",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            relative = "python/oh_my_vllm/kernels/tilelang_reference"
            shutil.copytree(ROOT / relative, root / relative)
            verify_reference(root)
            for relative in (
                "requirements/runtime.txt",
                "python/oh_my_vllm/kernels/tilelang_reference/gdn.py",
            ):
                path = root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n# changed\n")
                with self.assertRaisesRegex(ValueError, "frozen comparison changed"):
                    verify_reference(root)
                path.write_bytes(original)
