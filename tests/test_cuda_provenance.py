"""The build identity must describe the loaded module, not a newer cache entry."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

BACKEND = (
    Path(__file__).resolve().parents[1]
    / "python/oh_my_vllm/kernels/cuda_backend/__init__.py"
)


def isolated_backend():
    """Give every test a fresh compiled() cache without touching production state."""
    spec = importlib.util.spec_from_file_location("isolated_cuda_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProvenanceTest(unittest.TestCase):
    def test_compiled_records_exact_library_and_actual_compiler(self):
        backend = isolated_backend()
        with tempfile.TemporaryDirectory() as directory:
            loaded = Path(directory) / "loaded.so"
            loaded.write_bytes(b"loaded CUDA module")
            newer = Path(directory) / "newer.so"
            newer.write_bytes(b"not loaded")
            expected = hashlib.sha256(loaded.read_bytes()).hexdigest()
            captured = {}

            def build(*args, **kwargs):
                captured.update(kwargs)
                return str(loaded)

            output = io.StringIO()
            with (
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("tvm_ffi.cpp.build_inline", side_effect=build),
                patch("tvm_ffi.load_module", return_value=object()),
                patch.object(
                    backend,
                    "_nvcc_identity",
                    return_value=(Path("/cuda/bin/nvcc"), "V12.8.1"),
                ),
                redirect_stderr(output),
            ):
                backend.compiled()
            reported = backend.provenance(require_loaded=True)
            self.assertEqual(reported["so_path"], str(loaded))
            self.assertEqual(reported["so_sha256"], expected)
            self.assertEqual(reported["nvcc_path"], "/cuda/bin/nvcc")
            self.assertEqual(reported["nvcc_version"], "V12.8.1")
            self.assertTrue(
                any(
                    flag.startswith("-DOH_MY_VLLM_NVCC_ID_")
                    for flag in captured["extra_cuda_cflags"]
                )
            )
            marker = output.getvalue().strip().removeprefix("CUDA_BUILD_PROVENANCE ")
            self.assertEqual(json.loads(marker), reported)

    def test_cuda_home_compiler_wins_over_path(self):
        backend = isolated_backend()
        with (
            patch("tvm_ffi.cpp.extension._find_cuda_home", return_value="/cuda/home"),
            patch(
                "subprocess.check_output", return_value="nvcc release 12.8, V12.8.1\n"
            ) as check,
        ):
            path, version = backend._nvcc_identity()
        self.assertEqual(path, Path("/cuda/home/bin/nvcc"))
        self.assertEqual(version, "nvcc release 12.8, V12.8.1")
        check.assert_called_once_with(["/cuda/home/bin/nvcc", "--version"], text=True)

    def test_compiler_version_changes_build_cache_key(self):
        flags = []
        with tempfile.TemporaryDirectory() as directory:
            loaded = Path(directory) / "loaded.so"
            loaded.write_bytes(b"module")

            def build(*args, **kwargs):
                flags.append(kwargs["extra_cuda_cflags"])
                return str(loaded)

            for version in ("V12.8.1", "V12.9.0"):
                backend = isolated_backend()
                with (
                    patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                    patch("tvm_ffi.cpp.build_inline", side_effect=build),
                    patch("tvm_ffi.load_module", return_value=object()),
                    patch.object(
                        backend,
                        "_nvcc_identity",
                        return_value=(Path("/cuda/bin/nvcc"), version),
                    ),
                    redirect_stderr(io.StringIO()),
                ):
                    backend.compiled()
        self.assertNotEqual(flags[0], flags[1])

    def test_nvcc_disappearing_after_build_does_not_break_runtime(self):
        backend = isolated_backend()
        with tempfile.TemporaryDirectory() as directory:
            loaded = Path(directory) / "cached.so"
            loaded.write_bytes(b"cached module")
            with (
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("tvm_ffi.cpp.build_inline", return_value=str(loaded)),
                patch("tvm_ffi.load_module", return_value=object()),
                patch.object(
                    backend,
                    "_nvcc_identity",
                    side_effect=[
                        (Path("/cuda/bin/nvcc"), "V12.8.1"),
                        (Path("/cuda/bin/nvcc"), "nvcc-unavailable: missing"),
                    ],
                ),
                redirect_stderr(io.StringIO()),
            ):
                backend.compiled()
            self.assertEqual(
                backend.provenance(require_loaded=True)["nvcc_version"],
                "nvcc-changed-during-build",
            )
            with self.assertRaisesRegex(RuntimeError, "compiler identity"):
                backend.provenance(require_loaded=True, require_compiler=True)

    def test_unloaded_process_never_claims_a_cached_shared_library(self):
        backend = isolated_backend()
        self.assertIsNone(backend.provenance()["so_sha256"])
        with self.assertRaisesRegex(RuntimeError, "loaded CUDA module"):
            backend.provenance(require_loaded=True)


if __name__ == "__main__":
    unittest.main()
