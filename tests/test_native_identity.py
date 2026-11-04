import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_my_vllm.operators.runtime import native


class NativeIdentityTests(unittest.TestCase):
    def make_cuda_home(self, version="release 13.4, V13.4.99"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        compiler = root / "bin/nvcc"
        compiler.parent.mkdir()
        compiler.write_text(f"#!/bin/sh\necho '{version}'\n")
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        return root

    @staticmethod
    def environment():
        return {
            "python_version": "3.15.2",
            "torch_version": "2.15.0+cu134",
            "torch_cuda_version": "13.4",
            "cuda_home": "/opt/cuda-13.4",
            "cuda_includes": [
                "/opt/cuda-13.4/targets/x86_64-linux/include",
                "/opt/cuda-13.4/include",
            ],
            "cuda_compiler": "nvcc 13.4",
        }

    def build_digest(self, environment):
        return native.build_digest(
            "fingerprint",
            "nvcc 13.4",
            ("-O3", "--generate-code=arch=compute_100a,code=sm_100a", "-lineinfo"),
            "0.1.14",
            None,
            ("allocate",),
            environment=environment,
        )

    def test_collects_the_python_torch_and_cuda_compiler_identity(self):
        cuda_home = self.make_cuda_home()
        with patch.dict(os.environ, {"CUDA_HOME": str(cuda_home)}, clear=False):
            identity = native.build_environment_identity()

        self.assertEqual(identity["python_version"], "3.15.2")
        self.assertIn("2.15.0+cu134", identity["torch_version"])
        self.assertEqual(identity["torch_cuda_version"], "13.4")
        self.assertEqual(identity["cuda_home"], str(cuda_home.resolve()))
        self.assertEqual(
            identity["cuda_includes"], [str((cuda_home / "include").resolve())]
        )
        self.assertIn("13.4", identity["cuda_compiler"])

    def test_rejects_a_manifest_with_a_different_environment_identity(self):
        expected = self.environment()
        for field in native.ENVIRONMENT_IDENTITY_FIELDS:
            with self.subTest(field=field):
                actual = dict(expected)
                actual[field] = f"different {field}"
                with self.assertRaisesRegex(ValueError, field):
                    native.validate_environment_identity(actual, expected)

    def test_accepts_a_matching_environment_identity(self):
        expected = self.environment()
        native.validate_environment_identity(dict(expected), expected)

    def test_missing_environment_identity_is_stale_with_rebuild_advice(self):
        expected = self.environment()
        metadata = dict(expected)
        del metadata["cuda_includes"]

        with self.assertRaisesRegex(RuntimeError, "stale.*native_build"):
            native.require_current_environment(metadata, expected)

    def test_environment_change_uses_a_new_build_directory(self):
        expected = self.environment()
        for field in native.ENVIRONMENT_IDENTITY_FIELDS:
            with self.subTest(field=field):
                changed = dict(expected)
                changed[field] = f"different {field}"
                self.assertNotEqual(
                    self.build_digest(expected), self.build_digest(changed)
                )

    def test_same_environment_keeps_a_stable_build_directory(self):
        expected = self.environment()
        reordered = {
            field: expected[field]
            for field in reversed(native.ENVIRONMENT_IDENTITY_FIELDS)
        }

        self.assertEqual(self.build_digest(expected), self.build_digest(reordered))
        self.assertEqual(self.build_digest(expected), self.build_digest(dict(expected)))

    def test_rejects_a_missing_cuda_home_compiler(self):
        with (
            patch.dict(os.environ, {"CUDA_HOME": "/opt/missing-cuda"}, clear=False),
            self.assertRaisesRegex(RuntimeError, "nvcc"),
        ):
            native.build_environment_identity()

    def test_cuda_include_directory_falls_back_to_the_toolkit_include(self):
        cuda_home = self.make_cuda_home()

        self.assertEqual(
            native.cuda_include_directory(cuda_home), cuda_home / "include"
        )

    def test_cuda_include_directory_prefers_complete_target_headers(self):
        cuda_home = self.make_cuda_home()
        target = cuda_home / "targets/x86_64-linux/include"
        (target / "crt").mkdir(parents=True)
        (target / "cuda_runtime.h").write_text("")
        (target / "crt/host_config.h").write_text("")

        self.assertEqual(native.cuda_include_directory(cuda_home), target)
        self.assertEqual(
            native.cuda_include_directories(cuda_home),
            (target, cuda_home / "include"),
        )

        (target / "crt/host_config.h").unlink()
        self.assertEqual(
            native.cuda_include_directory(cuda_home), cuda_home / "include"
        )
        self.assertEqual(
            native.cuda_include_directories(cuda_home), (cuda_home / "include",)
        )

    def test_vision_vendor_uses_the_torch_runtime_images(self):
        vendor = native.vision_vendor_dependencies()

        self.assertIn("site-packages/nvidia", vendor["images"][0]["path"])
        self.assertIn("site-packages/nvidia", vendor["images"][2]["path"])

    def test_collectives_vendor_uses_the_torch_nccl_image(self):
        vendor = native.collectives_vendor_dependencies()

        self.assertIn("site-packages/nvidia/nccl", vendor["images"][0]["path"])


if __name__ == "__main__":
    unittest.main()
